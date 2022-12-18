import argparse
import math
from collections import namedtuple
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.positional_embedding import PositionalEmbedding
from torch import Tensor

from dictionary import Dictionary
from models import register_model
from models.model_utils import init_bert_params
from modules.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder

register_name = "nat_transformer"

INF = 1e10


def find_bin_tree_level(seq_len):
    index = 1
    while True:
        lower_bound = math.pow(2, index) - 1
        high_bound = math.pow(2, index + 1) - 1
        if lower_bound < seq_len <= high_bound:
            return index
        index += 1


def softmax(x, T=1):
    return F.softmax(x / T, dim=-1)

    # if x.dim() == 3:
    #     return F.softmax(x.transpose(0, 2)).transpose(0, 2)
    # return F.softmax(x)


def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-1)).squeeze(-1)


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = src_lens.float() / trg_lens.float()  # step-size
    # max_trg_len
    index_t = new_arange(trg_lens, max_trg_len).float() + 1
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach() - 1

    src_lens_expand = src_lens[:, None].expand(src_lens.size(0), max_trg_len)

    index_t[index_t < 0] = 0
    index_t[index_t > src_lens_expand] = (src_lens_expand[index_t > src_lens_expand] - 1)
    return index_t


def interplote(source_masks, decoder_masks):
    max_src_len = source_masks.size(1)
    max_trg_len = decoder_masks.size(1)
    src_lens = source_masks.sum(-1).float()  # batchsize
    trg_lens = decoder_masks.sum(-1).float()  # batchsize
    steps = src_lens / trg_lens  # batchsize
    index_t = torch.arange(0, max_trg_len)  # max_trg_len
    if decoder_masks.is_cuda:
        index_t = index_t.cuda(decoder_masks.get_device())

    index_t = steps[:, None] @ index_t[None, :].float()  # batch x max_trg_len
    index_s = torch.arange(0, max_src_len)  # max_src_len
    if decoder_masks.is_cuda:
        index_s = index_s.cuda(decoder_masks.get_device())

    indexxx_ = (index_s[None, None, :] - index_t[:, :, None]) ** 2  # batch x max_trg x max_src
    indexxx = softmax(torch.tensor(
        -indexxx_.float() / 0.3 - INF * (1 - source_masks[:, None, :].float())))  # batch x max_trg x max_src
    return indexxx


# set arguments based on data flow order, data first pass through encoder, then decoder.
# encoder and decoder including embedding layer, multihead attention layer, feedforward layer, so we set
# arguments based on layer order.
# although most of arguments keep default values, these grouped arguments and their order could help you
# understand the model architecture from the top level and inspect the details from the bottom.
default_dict: Dict[str, Dict] = {

    "apply_bert_init": {"action": 'store_true'},
    "length_beam_size": {"type": int, "default": 1, "help": "length parallel decoding"},
    # non-autoregressive transformer setting
    "decoder_input_how": {"type": str, "default": None, "choices": ['copy', 'interpolate', 'pad', 'wrap'],
                          "help": "copy encoder word embeddings as the initial input of the decoder"},
    "pred_length": {"action": 'store_true',
                    "help": "predicting the target length"},
    "max_length": {"type": int, "default": 1024, "help": "length value"},
    "use_ground_truth_length": {"action": 'store_true',
                                "help": "use ground truth length in predicting length"},
    "use_ground_truth_target": {"action": 'store_true',
                                "help": "use ground truth target in predicting target tokens, this is really not regular"},
    "use_bridge": {"action": 'store_true',
                   "help": "bridge module"},

    # general arguments setting
    "share_all_embeddings": {"default": False,
                             "action": 'store_true',
                             "help": "share encoder, decoder and output embeddings "
                                     "(requires shared dictionary and embed dim)"},
    # this dropout uses after token embedding + position embedding, self attention, second feedforward network
    "dropout": {"type": float, "default": 0.1, "help": "dropout probability"},

    ##############################
    # encoder arguments setting  #
    ##############################
    # token embedding layer
    "encoder_embed_dim": {"type": int, "default": 512, "help": "source embedding dimension"},
    # position embedding layer
    "encoder_learned_pos": {"default": False, "action": 'store_true'},
    "encoder_max_source_positions": {"type": int, "default": 1024,
                                     "help": "Maximum input length supported by the encoder"},
    # encoder layer setting
    "encoder_layers": {"type": int, "default": 6, "help": "number of layers"},
    # encoder multiheads attention setting
    "encoder_attention_heads": {"type": int, "default": 8, "help": "number of attention heads"},
    "encoder_attention_dropout": {"type": float, "default": 0.0, "help": "dropout probability for attention weights"},

    # encoder feedforward network setting
    "encoder_ffn_embed_dim": {"type": int, "default": 2048, "help": "embedding dimension for FFN"},
    "encoder_activate_fn": {"type": str, "default": "relu", "help": "activation function to use"},
    "encoder_activation_dropout": {"type": float, "default": 0.0,
                                   "help": "dropout probability after activation in FFN."},

    # ##############################
    # # decoder arguments setting  #
    # ##############################
    # token embedding layer
    "share_decoder_input_output_embed": {"default": True, "action": "store_true",
                                         "help": "share decoder input and output embeddings"},
    "decoder_embed_dim": {"type": int, "default": 512, "help": "target embedding dimension"},
    # position embedding layer
    "decoder_learned_pos": {"default": False, "action": "store_true", "help": " "},
    "decoder_max_source_positions": {"type": int, "default": 1024,
                                     "help": "Maximum input length supported by the decoder"},
    # decoder layer setting
    "decoder_layers": {"type": int, "default": 6, "help": "number of layers"},
    # decoder multiheads attention setting
    "decoder_attention_heads": {"type": int, "default": 8, "help": "number of attention heads"},
    "decoder_attention_dropout": {"type": float, "default": 0.0, "help": "dropout probability for attention weights"},

    # decoder feedforward network setting
    "decoder_ffn_embed_dim": {"type": int, "default": 2048, "help": "embedding dimension for FFN"},
    "decoder_activate_fn": {"type": str, "default": "relu", "help": "activation function to use"},
    "decoder_activation_dropout": {"type": float, "default": 0.0,
                                   "help": "dropout probability after activation in FFN."},
}


@register_model(register_name)
class NAT(nn.Module):
    config = default_dict

    def __init__(self, config: namedtuple, src_dict: Dictionary, tgt_dict: Dictionary):
        super(NAT, self).__init__()
        self.config = config
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        src_token_emb = nn.Embedding(len(src_dict), self.config.encoder_embed_dim, padding_idx=src_dict.padding_id)
        if config.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if config.encoder_embed_dim != config.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )

            tgt_token_emb = src_token_emb
        else:
            tgt_token_emb = nn.Embedding(len(tgt_dict), self.config.decoder_embed_dim, padding_idx=tgt_dict.padding_id)

        self.encoder = NATencoder(self.config, src_dict, src_token_emb)
        self.decoder = NATdecoder(self.config, tgt_dict, tgt_token_emb)
        if config.apply_bert_init:
            self.encoder.apply(init_bert_params)
            self.decoder.apply(init_bert_params)

            # in decoder initialization, output_layer.weight = self.token_emb.weight; while in function init_bert_params,
            # it first initializes token_emb, then initializes output_layer, which leads to token_emb padding_idx
            # embedding value not zero, so we must initialize token_emb again.
            def init_embedding_params(module):
                def normal_(data):
                    # with FSDP, module params will be on CUDA, so we cast them back to CPU
                    # so that the RNG is consistent with and without FSDP
                    data.copy_(
                        data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
                    )

                if isinstance(module, nn.Embedding):
                    normal_(module.weight.data)
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()

            self.decoder.apply(init_embedding_params)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """
        :param parser:
        """
        for param_name, param_attr in default_dict.items():
            parser.add_argument("--" + param_name, **param_attr)

    def forward(self, src_tokens: Tensor, prev_tgt_tokens: Tensor) -> Dict:
        """

        :param tgt_masks:
        :param prev_tgt_tokens:
        :param src_masks:
        :param src_tokens:
        :param batch: batch is dict type, detailed key-value definition is in the dataset class collate_fn return
                     value. thus we define the data interface between model and dataset
        """

        encoder_outputs: dict = self.encoder(src_tokens)

        decoder_outputs = self.decoder(prev_tgt_tokens, encoder_outputs)

        return decoder_outputs

    def generate(self, src_tokens: Tensor) -> Dict:
        encoder_outputs: dict = self.encoder(src_tokens)

        if self.config.pred_length:
            length_logits = self.decoder.predict_length(encoder_outputs["encoder_features"],
                                                        encoder_outputs["src_masks"])

            predicted_lengths = F.log_softmax(length_logits, dim=-1)
            beam = predicted_lengths.topk(self.config.length_beam_size, dim=1)[1]  # [batch_size, length_beam_size]
            beam[beam < 2] = 2
            pred_max_length = beam.max().item()
            max_iter = find_bin_tree_level(pred_max_length)

            if max_iter >= 8:
                max_iter = 8
        else:
            max_iter = 7 # max length 255

        output_tokens, start_pos, end_pos = self.initialize_output_tokens(src_tokens)
        new_decoder_outputs = self.decoder.generate(output_tokens, encoder_outputs)
        new_output_tokens = new_decoder_outputs["hypo_tokens"]

        output_tokens[:, start_pos: end_pos] = new_output_tokens[:, start_pos: end_pos]

        for i in range(max_iter - 1):
            output_tokens, start_pos, end_pos = self.prepare_next_input(output_tokens)

            new_decoder_outputs = self.decoder.generate(output_tokens, encoder_outputs)
            new_output_tokens = new_decoder_outputs["hypo_tokens"]

            output_tokens[:, start_pos: end_pos] = new_output_tokens[:, start_pos: end_pos]

        return {"hypo_tokens": output_tokens}

    def initialize_output_tokens(self, src_tokens):
        initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 3)
        initial_output_tokens[:, 0] = self.decoder.dictionary.bos_id
        initial_output_tokens[:, 1] = self.decoder.dictionary.mask_id
        initial_output_tokens[:, 2] = self.decoder.dictionary.mask_id

        start_pos, end_pos = 1, 3

        return initial_output_tokens, start_pos, end_pos

    def prepare_next_input(self, output_tokens):
        try:
            seq_len = torch.tensor(output_tokens.size(1)).to(torch.float32)
            level = torch.log2(seq_len + 1).to(torch.int64).to(torch.float32)
            mask_num = torch.exp2(level).to(torch.int64)
            append = output_tokens.new(size=(output_tokens.size(0), mask_num)).fill_(self.decoder.dictionary.mask_id)

            output_tokens = torch.cat((output_tokens, append), dim=1)  # noqa

            start_pos = seq_len.to(torch.int64)
            end_pos = start_pos + mask_num
        except:
            print("error")

        return output_tokens, start_pos, end_pos


class NATencoder(nn.Module):
    def __init__(self, config, dictionary: Dictionary, token_emb):
        super(NATencoder, self).__init__()
        self.config = config

        self.token_emb = token_emb
        self.dropout = nn.Dropout(self.config.dropout)
        self.dictionary = dictionary
        self.embed_scale = math.sqrt(self.config.encoder_embed_dim)

        self.position_emb = PositionalEmbedding(num_embeddings=self.config.encoder_max_source_positions,
                                                embedding_dim=self.config.encoder_embed_dim,
                                                padding_idx=dictionary.padding_id,
                                                learned=self.config.encoder_learned_pos)

        self.layer = TransformerEncoderLayer(d_model=self.config.encoder_embed_dim,
                                                nhead=self.config.encoder_attention_heads,
                                                batch_first=True,
                                                dropout=self.config.dropout,
                                                dim_feedforward=self.config.encoder_ffn_embed_dim,
                                                activation=self.config.encoder_activate_fn)

        self.encoder = TransformerEncoder(encoder_layer=self.layer, num_layers=self.config.encoder_layers)

    def forward(self, src_tokens: Tensor) -> Dict:
        src_masks = (src_tokens == self.dictionary.padding_id)
        src_emb = self.embed_scale * self.token_emb(src_tokens) + self.position_emb(src_tokens)
        src_emb = self.dropout(src_emb)

        encoder_features = self.encoder(src=src_emb, src_key_padding_mask=src_masks)

        encoder_output = {"src_masks": src_masks, "encoder_features": encoder_features}

        return encoder_output


class NATdecoder(nn.Module):
    def __init__(self, config, dictionary: Dictionary, token_emb):
        super(NATdecoder, self).__init__()
        self.config = config

        self.token_emb = token_emb
        self.dropout = nn.Dropout(self.config.dropout)
        self.dictionary = dictionary
        self.embed_scale = math.sqrt(self.config.encoder_embed_dim)

        self.position_emb = PositionalEmbedding(num_embeddings=self.config.decoder_max_source_positions,
                                                embedding_dim=self.config.decoder_embed_dim,
                                                padding_idx=dictionary.padding_id,
                                                learned=self.config.decoder_learned_pos)

        self.layer = TransformerDecoderLayer(d_model=self.config.decoder_embed_dim,
                                                nhead=self.config.decoder_attention_heads,
                                                batch_first=True,
                                                dropout=self.config.dropout,
                                                dim_feedforward=self.config.decoder_ffn_embed_dim,
                                                activation=self.config.decoder_activate_fn)

        self.decoder = TransformerDecoder(decoder_layer=self.layer, num_layers=self.config.decoder_layers)

        if self.config.share_decoder_input_output_embed:
            self.output_layer = nn.Linear(self.token_emb.weight.size(1), self.token_emb.weight.size(0), bias=False)
            self.output_layer.weight = self.token_emb.weight
        else:
            self.output_layer = nn.Linear(self.token_emb.weight.size(1), self.token_emb.weight.size(0), bias=False)

        if self.config.pred_length:
            self.length_predictor = nn.Linear(self.config.decoder_embed_dim, self.config.max_length)

    def forward(self, prev_tgt_tokens: Tensor, encoder_outputs: dict) -> Dict:
        # src_emb = encoder_outputs["src_emb"]
        src_masks = encoder_outputs["src_masks"]
        encoder_features = encoder_outputs["encoder_features"]

        # prev_tgt_tokens, prev_tgt_key_padding_masks may have length parallel, so the first dim is
        # batch_size * length_beam_size
        src_batch_size, src_len, src_feat_num = encoder_features.size()
        tgt_batch_length_size, max_tgt_len = prev_tgt_tokens.size()
        if src_batch_size != tgt_batch_length_size:
            beam_size = int(tgt_batch_length_size / src_batch_size)
            # src_emb = src_emb.unsqueeze(1).expand(src_batch_size, beam_size, src_len, src_feat_num).view(
            #     src_batch_size * beam_size, src_len,
            #     src_feat_num)
            src_masks = src_masks.unsqueeze(1).expand(src_batch_size, beam_size, src_len).view(
                src_batch_size * beam_size, -1)
            encoder_features = encoder_features.unsqueeze(1).expand(src_batch_size, beam_size, src_len,
                                                                    src_feat_num).view(src_batch_size * beam_size,
                                                                                       src_len, src_feat_num)

        # TODO: construct mask
        tgt_token_embed = self.token_emb(prev_tgt_tokens)
        tgt_emb = self.embed_scale * tgt_token_embed + self.position_emb(prev_tgt_tokens)
        tgt_emb = self.dropout(tgt_emb)

        padding_mask = prev_tgt_tokens.eq(self.dictionary.padding_id)

        features = self.decoder(tgt=tgt_emb, memory=encoder_features, memory_key_padding_mask=src_masks,
                                tgt_key_padding_mask=padding_mask)

        logits = self.output_layer(features)

        decoder_outputs = {"logits": logits}

        if self.config.pred_length:
            length_logits = self.predict_length(encoder_features, src_masks)
            decoder_outputs['length_logits'] = length_logits

        return decoder_outputs

    def generate(self, decoder_input_tokens: Tensor, encoder_outputs: dict) -> Dict:
        src_masks = encoder_outputs["src_masks"]
        encoder_features = encoder_outputs["encoder_features"]

        tgt_token_embed = self.token_emb(decoder_input_tokens)

        tgt_emb = self.embed_scale * tgt_token_embed + self.position_emb(decoder_input_tokens)
        tgt_emb = self.dropout(tgt_emb)

        features = self.decoder(tgt=tgt_emb, memory=encoder_features, memory_key_padding_mask=src_masks)

        logits = self.output_layer(features)
        hypo_tokens = logits.argmax(dim=-1)
        decoder_outputs = {"hypo_tokens": hypo_tokens}
        return decoder_outputs

    def construct_mask(self, prev_tgt_tokens):
        tgt_max_len = prev_tgt_tokens.size(1)
        tri_lower_mask = prev_tgt_tokens.new_ones((tgt_max_len, tgt_max_len)).tril(0)
        funnel_mask = (tri_lower_mask + torch.flip(tri_lower_mask, dims=[-1])) == 0
        flip_funnel_mask = torch.flip(funnel_mask, dims=[0])
        tgt_masks = ~(funnel_mask + flip_funnel_mask)
        return tgt_masks

    def construct_train_batch(self, prev_tgt_tokens, ind, symmetry_ind):
        prev_tgt_tokens[:, ind] = self.dictionary.mask_id
        prev_tgt_tokens[:, symmetry_ind] = self.dictionary.mask_id
        if ind >= symmetry_ind:
            prev_tgt_tokens[:, :symmetry_ind] = self.dictionary.padding_id
            prev_tgt_tokens[:, ind:] = self.dictionary.padding_id
        else:
            prev_tgt_tokens[:, :ind] = self.dictionary.padding_id
            prev_tgt_tokens[:, symmetry_ind:] = self.dictionary.padding_id

        padding_mask = prev_tgt_tokens.eq(self.dictionary.padding_id)
        predict_mask = prev_tgt_tokens.ne(self.dictionary.mask_id)

        return prev_tgt_tokens, padding_mask, predict_mask

    def copy_src_embed(self, src_emb: Tensor, src_masks: Tensor, tgt_masks: Tensor) -> Tensor:
        src_unmasks = ~src_masks
        tgt_unmasks = ~tgt_masks
        length_sources = src_unmasks.sum(1)
        length_targets = tgt_unmasks.sum(1)
        mapped_inputs = uniform_assignment(length_sources, length_targets).masked_fill(tgt_masks, 0)
        # mapped_inputs = interplote(src_unmasks, tgt_unmasks)

        copied_embedding = torch.gather(
            src_emb,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_emb.size(-1)
            ),
        )
        copied_embedding[tgt_masks] = self.token_emb.weight[self.dictionary.padding_id]
        return copied_embedding

    def predict_length(self, encoder_features: Tensor, src_masks: Tensor) -> Tensor:
        """

        :param encoder_features: shape [B, T, C]
        :param src_masks: shape [B, T]
        :return: shape [B]
        """
        encoder_feats = encoder_features.masked_fill(src_masks[:, :, None], 0.)
        encoder_feats = encoder_feats.sum(1) / (~src_masks).sum(1)[:, None]

        length_logits = self.length_predictor(encoder_feats)
        return length_logits

    def interplote(self, src_emb: Tensor, src_masks: Tensor, tgt_masks: Tensor) -> Tensor:
        source_masks = ~src_masks
        decoder_masks = ~tgt_masks

        max_src_len = source_masks.size(1)
        max_trg_len = decoder_masks.size(1)
        src_lens = source_masks.sum(-1).float()  # batchsize
        trg_lens = decoder_masks.sum(-1).float()  # batchsize
        steps = src_lens / trg_lens  # batchsize
        index_t = torch.arange(0, max_trg_len)  # max_trg_len
        if decoder_masks.is_cuda:
            index_t = index_t.cuda(decoder_masks.get_device())

        index_t = steps[:, None] @ index_t[None, :].float()  # batch x max_trg_len
        index_s = torch.arange(0, max_src_len)  # max_src_len
        if decoder_masks.is_cuda:
            index_s = index_s.cuda(decoder_masks.get_device())

        indexxx_ = (index_s[None, None, :] - index_t[:, :, None]) ** 2  # batch x max_trg x max_src
        indexxx = softmax(torch.tensor(
            -indexxx_.float() / 0.3 - INF * (1 - source_masks[:, None, :].float())))  # batch x max_trg x max_src

        decoder_inputs = matmul(indexxx, src_emb)

        decoder_inputs[tgt_masks] = self.token_emb.weight[self.dictionary.padding_id]

        return decoder_inputs
