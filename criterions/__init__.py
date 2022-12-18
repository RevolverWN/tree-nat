import importlib
import os
from collections import defaultdict
from typing import Tuple, Dict, List
import torch.distributed as dist
from torch import Tensor

registry = defaultdict(dict)


def register_criterion(criterion_name):
    def register_criterion_cls(cls):
        registry[criterion_name]["cls"] = cls
        registry[criterion_name]["default_config_dict"] = cls.config
        return cls
    return register_criterion_cls


dir_name, base_name = os.path.split(__file__)
for filename in os.listdir(dir_name):
    if filename.endswith(".py") and filename != base_name:
        importlib.import_module("criterions." + filename[:filename.rfind(".py")])


class Criterion(object):
    def __init__(self, criterions: Dict, gpu_count: int):
        self.criterions = criterions
        self.gpu_count = gpu_count

    def train(self, model_outputs: Dict, target: Tensor, tgt_lengths: Tensor, padding_id, nsentences, ntokens) -> Dict:
        """
        a factory function for different criterions

        :param model_outputs:
        :param target:
        :param target_masks:
        :param padding_id:
        :return:
        """
        loss_dict = {}
        for cri_name, cri in self.criterions.items():
            if cri_name == "label_smoothed_cross_entropy":
                loss_dict[cri_name] = cri(model_outputs["logits"], target, padding_id)
            elif cri_name == "length_criterion":
                loss_dict[cri_name] = cri(model_outputs["length_logits"], tgt_lengths)

        for name, value in loss_dict.items():
            # if self.reduce_strategy[name] == "ntokens":
            #     loss_dict[name] = loss_dict[name] / ntokens
            if self.reduce_strategy[name] == "nsentences":
                loss_dict[name] = loss_dict[name] / nsentences

        loss = sum(loss_dict.values())

        return loss, {name: value.data for name, value in loss_dict.items()} # noqa

    def eval(self, model_outputs: Dict, target: Tensor, tgt_lengths: Tensor, padding_ids: List):
        loss_dict = {}
        for cri_name, cri in self.criterions.items():
            if cri_name == "label_smoothed_cross_entropy":
                loss_dict[cri_name] = cri(model_outputs["logits"], model_outputs["predict_mask"], target, padding_ids)
            elif cri_name == "length_criterion":
                loss_dict[cri_name] = cri(model_outputs["length_logits"], tgt_lengths)

        return loss_dict

    @property
    def reduce_strategy(self) -> Dict:
        return {cri_name: cri.samples_reduce for cri_name, cri in self.criterions.items()}



