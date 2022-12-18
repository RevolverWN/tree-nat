import itertools
import logging
import math
import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from fairseq.data.data_utils import batch_by_size


from dictionary import Dictionary

logger = logging.getLogger(__name__)


class TreeNode(object):
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def __eq__(self, other):
        if isinstance(other, TreeNode):
            return id(self) == id(other)


# TODO: all the CRUD(Create, Read, Update, Delete) operation must base on node, not node's data,
#  'cause nodes in the tree may have same data value!!! several methods in Tree class are based on data,
#  we leave the error correction in the future.
class Tree(object):
    def __init__(self):
        self.root = None
        self.flatten = []
        self.valid_data = None
        self.inorder = []
        self.single_child_nodes = []

    def add(self, item):
        node = TreeNode(item)
        if self.root is None:
            self.root = node
        else:
            q = [self.root]
            while True:
                pop_node = q.pop(0)
                if pop_node.left is None:
                    pop_node.left = node
                    return
                elif pop_node.right is None:
                    pop_node.right = node
                    return
                else:
                    q.append(pop_node.left)
                    q.append(pop_node.right)

    def get_parent(self, node):
        if self.root == node:
            return None
        tmp = [self.root]
        while tmp:
            pop_node = tmp.pop(0)
            if pop_node.left and pop_node.left == node:
                return pop_node
            if pop_node.right and pop_node.right == node:
                return pop_node
            if pop_node.left is not None:
                tmp.append(pop_node.left)
            if pop_node.right is not None:
                tmp.append(pop_node.right)
        return None

    def get_child(self, node):
        if self.root == node:
            return None
        tmp = [self.root]
        while tmp:
            pop_node = tmp.pop(0)
            if pop_node.left and pop_node.left.data == node.data:
                current_node = pop_node.left
                if current_node.left is not None:
                    left_child = current_node.left.data
                else:
                    left_child = None

                if current_node.right is not None:
                    right_child = current_node.right.data
                else:
                    right_child = None
                return left_child, right_child

            if pop_node.right and pop_node.right.data == node.data:
                current_node = pop_node.right
                if current_node.left is not None:
                    left_child = current_node.left.data
                else:
                    left_child = None

                if current_node.right is not None:
                    right_child = current_node.right.data
                else:
                    right_child = None
                return left_child, right_child

            if pop_node.left is not None:
                tmp.append(pop_node.left)
            if pop_node.right is not None:
                tmp.append(pop_node.right)
        return None

    def delete(self, item):
        if self.root is None:
            return False

        parent = self.get_parent(item)
        if parent:
            del_node = parent.left if parent.left.item == item else parent.right
            if del_node.left is None:
                if parent.left.item == item:
                    parent.left = del_node.right
                else:
                    parent.right = del_node.right
                del del_node
                return True
            elif del_node.right is None:
                if parent.left.item == item:
                    parent.left = del_node.left
                else:
                    parent.right = del_node.left
                del del_node
                return True
            else:
                tmp_pre = del_node
                tmp_next = del_node.right
                if tmp_next.left is None:

                    tmp_pre.right = tmp_next.right
                    tmp_next.left = del_node.left
                    tmp_next.right = del_node.right

                else:
                    while tmp_next.left:  # 让tmp指向右子树的最后一个叶子
                        tmp_pre = tmp_next
                        tmp_next = tmp_next.left

                    tmp_pre.left = tmp_next.right
                    tmp_next.left = del_node.left
                    tmp_next.right = del_node.right
                if parent.left.item == item:
                    parent.left = tmp_next
                else:
                    parent.right = tmp_next
                del del_node
                return True
        else:
            return False

    def sorted_array_to_BST(self, sequence):
        """
          :type sequence: List[int]
          :rtype: TreeNode
        """
        if len(sequence) == 0:
            return None
        mid = sequence[len(sequence) // 2]
        root = TreeNode(mid)
        root.left = self.sorted_array_to_BST(sequence[:len(sequence) // 2])
        root.right = self.sorted_array_to_BST(sequence[len(sequence) // 2 + 1:])
        return root

    def height(self, root):
        if root is None:
            return 0
        else:
            # Compute the height of left and right subtree
            l_height = self.height(root.left)
            r_height = self.height(root.right)
            # Find the greater one, and return it
            if l_height > r_height:
                return l_height + 1
            else:
                return r_height + 1

    def fill_single_node(self, item):
        node = TreeNode(item)
        if self.root is None:
            self.root = node
        else:
            q = [self.root]
            while True:
                pop_node = q.pop(0)
                if pop_node.left is None and pop_node.right is not None:
                    pop_node.left = node
                    return
                elif pop_node.right is None and pop_node.left is not None:
                    pop_node.right = node
                    return
                else:
                    q.append(pop_node.left)
                    q.append(pop_node.right)

    def level_order(self, root):
        self.flatten.clear()
        h = self.height(root)
        for i in range(1, h + 1):
            res = self.print_given_level(root, i)
            if res is not None:
                self.flatten.append(res)

    def print_given_level(self, root, level):
        if root is None:
            return
        if level == 1:
            return root
        elif level > 1:
            left_res = self.print_given_level(root.left, level - 1)
            if left_res is not None:
                self.flatten.append(left_res)

            right_res = self.print_given_level(root.right, level - 1)
            if right_res is not None:
                self.flatten.append(right_res)

    def set_root(self, root):
        self.root = root

    def get_root(self):
        return self.root

    def get_leaf_nodes(self):
        leafs = []
        self._collect_leaf_nodes(self.root, leafs)
        return leafs

    def _collect_leaf_nodes(self, node, leafs):
        if node is not None:
            if node.left is None and node.right is None:
                leafs.append(node)
            for n in [node.left, node.right]:
                self._collect_leaf_nodes(n, leafs)

    def set_valid_data(self):
        valid_data = []
        for data in self.flatten:
            if data is not None:
                valid_data.append(data)

        self.valid_data = valid_data

    def get_node(self, data):
        if self.root.data == data:
            return self.root
        tmp = [self.root]
        while tmp:
            pop_node = tmp.pop(0)
            if pop_node.data == data:
                return pop_node

            if pop_node.left is not None:
                tmp.append(pop_node.left)
            if pop_node.right is not None:
                tmp.append(pop_node.right)
        return None

    def set_node(self, node: TreeNode, value):
        if self.root == node:
            return self.root
        tmp = [self.root]
        while tmp:
            pop_node = tmp.pop(0)
            if pop_node == node:
                pop_node.data = value

            if pop_node.left is not None:
                tmp.append(pop_node.left)
            if pop_node.right is not None:
                tmp.append(pop_node.right)
        return None

    def get_all_parents(self, root, node, ancestors_list):
        # base case
        if root is None:
            return False

        # return true if a given node is found
        if root == node:
            return True

        # search node in the left subtree
        left = self.get_all_parents(root.left, node, ancestors_list)

        # search node in the right subtree
        right = False
        if not left:
            right = self.get_all_parents(root.right, node, ancestors_list)

        # if the given node is found in either left or right subtree,
        # the current node is an ancestor of a given node
        if left or right:
            ancestors_list.append(root.data)
            return ancestors_list

    def get_all_children(self, root):
        res = []
        tmp = [root]
        while tmp:
            pop_node = tmp.pop(0)
            if pop_node.left is not None:
                tmp.append(pop_node.left)
                res.append(pop_node.left)
            if pop_node.right is not None:
                tmp.append(pop_node.right)
                res.append(pop_node.right)
        return res

    def get_subtree(self, root):
        res = [root]
        tmp = [root]
        while tmp:
            pop_node = tmp.pop(0)
            if pop_node.left is not None:
                tmp.append(pop_node.left)
                res.append(pop_node.left)
            if pop_node.right is not None:
                tmp.append(pop_node.right)
                res.append(pop_node.right)
        return res

    def get_sibling(self, node):
        parent = self.get_parent(node)
        if parent.left == node:
            return parent.right

        return parent.left

    def travers_inorder(self, root):
        if root:
            # First recur on left child
            self.travers_inorder(root.left)

            # then print the data of node
            self.inorder.append(root.data)

            # now recur on right child
            self.travers_inorder(root.right)

    def prune_tree_preorder(self, root, value):
        if not root:
            return

        if root.data == value:
            root.left = None
            root.right = None
        else:
            self.prune_tree_preorder(root.left, value)
            self.prune_tree_preorder(root.right, value)

    def build_tree_from_level(self, arr: List, root, i, n):
        """

        :param arr: token list
        :param root: default None
        :param i: position
        :param n: arr length
        :return:
        """
        # Base case for recursion
        if i < n:
            temp = TreeNode(arr[i])
            root = temp

            # insert left child
            root.left = self.build_tree_from_level(arr, root.left,
                                                   2 * i + 1, n)

            # insert right child
            root.right = self.build_tree_from_level(arr, root.right,
                                                    2 * i + 2, n)
        return root

    def get_nodes_one_child(self, root):
        # nodes having single child

        # Base Case
        if not root:
            return

        # Condition to check if the node
        # is having only one child
        if not root.left and root.right:
            self.single_child_nodes.append(root)
        elif root.left and not root.right:
            self.single_child_nodes.append(root)

        # Traversing the left child
        self.get_nodes_one_child(root.left)

        # Traversing the right child
        self.get_nodes_one_child(root.right)
        return

    def add_padding(self, item, require_level):
        for i in range(1, require_level):
            node_list = self.get_level_nodes(self.root, i, i)
            for node in node_list:
                if node.left is None:
                    node.left = TreeNode(item)

                if node.right is None:
                    node.right = TreeNode(item)

    def get_level_nodes(self, root, start, end):
        nodes = []
        if root is None:
            return

        # create an empty queue and enqueue the root node
        queue = deque()
        queue.append(root)

        # maintains the level of the current node
        level = 0

        # loop till queue is empty
        while queue:

            # increment level by 1
            level = level + 1

            # calculate the total number of nodes at the current level
            size = len(queue)

            # process every node of the current level and enqueue their
            # non-empty left and right child
            while size > 0:
                size = size - 1
                curr = queue.popleft()

                # print the node if its level is between given levels
                if start <= level <= end:
                    nodes.append(curr)

                if curr.left:
                    queue.append(curr.left)

                if curr.right:
                    queue.append(curr.right)

        return nodes

    @staticmethod
    def sorted_array_to_list(sequence, max_len: int, padding_id: int):
        """

        :param sequence:
        :param max_len:
        :param padding_id:
        :return:
        """
        q = []
        sample_tensor = torch.zeros(size=(max_len,), dtype=torch.int64).fill_(padding_id)
        q.append(sequence)
        for i in range(len(sample_tensor)):
            pop_seq = q.pop(0)
            if len(pop_seq) != 0:
                sample_tensor[i] = pop_seq[len(pop_seq) // 2]

            q.append(pop_seq[:len(pop_seq) // 2])
            q.append(pop_seq[len(pop_seq) // 2 + 1:])

        return sample_tensor


def single_dataset_batch(sent_lengths, max_tokens, max_sentences, num_tokens_fn):
    if bool(max_sentences) == bool(max_tokens):
        raise ValueError("only one of max_tokens and max_sentences can be assigned")

    ordered_indices = np.argsort(sent_lengths)
    batches = batch_by_size(indices=ordered_indices,
                            num_tokens_fn=num_tokens_fn,
                            max_sentences=max_sentences,
                            max_tokens=max_tokens,
                            required_batch_size_multiple=8)
    return batches


class Seq2seqDataset(Dataset):
    def __init__(self, data_path: str, dictionary: Dictionary):
        self.data_path = data_path
        self.dictionary = dictionary

    def __getitem__(self, index: int) -> np.array:
        pass

    def num_tokens(self, index):
        pass

    def collate_fn(self):
        pass

    def batch_sampler(self):
        pass

    def prefetch(self, indices):
        pass

    @property
    def support_prefetch(self):
        return False


class IndexDataset(Seq2seqDataset):
    def __init__(self, data_path: str, dictionary: Dictionary):
        super(IndexDataset, self).__init__(data_path, dictionary)
        self._do_init(data_path)

    def __getitem__(self, index):
        pos = self.pointer[index]
        self.f_bin.seek(pos * 8)
        return np.frombuffer(self.f_bin.read(self.sent_lengths[index] * 8), dtype=np.int64)

    def _do_init(self, data_path):
        with open(data_path + '.ptr', 'rb') as f:
            self.sent_lengths = np.frombuffer(f.read(), dtype=np.int64)
            # block pointer should start from zero
            self.pointer = np.concatenate((np.array([0], dtype=self.sent_lengths.dtype), self.sent_lengths.cumsum()))
        self.f_bin = open(data_path + '.bin', 'rb')

    def num_tokens(self, index):
        return self.sent_lengths[index]

    def batch_sampler(self, max_sentences=None, max_tokens=4096):
        batches = single_dataset_batch(self.sent_lengths, max_tokens, max_sentences, self.num_tokens)
        return batches

    @staticmethod
    def create_index_files(index_path: str, corpus_path: str, dictionary: Dictionary):
        with open(index_path + '.bin', 'wb') as f_bin:
            with open(index_path + '.ptr', 'wb') as f_ptr:
                with open(corpus_path, 'r', encoding='utf-8') as f_data:
                    length = []
                    for line in f_data:
                        line = line.strip().split()
                        line = dictionary.encode_sentence(line)
                        length.append(len(line))
                        f_bin.write(np.array(line, dtype=np.int64).tobytes())
                    # the length of cum_length is dataset samples + 1 due to the starting position 0
                    f_ptr.write(np.array(length, dtype=np.int64).tobytes())

                    logger.info("create {} index samples".format(len(length)))

    def __len__(self):
        return len(self.sent_lengths)

    def __del__(self):
        self.f_bin.close()

    def __getstate__(self):
        state_dict = {"path": self.data_path, "dictionary": self.dictionary}
        return state_dict

    def __setstate__(self, state_dict):
        self._do_init(state_dict["path"])
        self.dictionary = state_dict["dictionary"]


class CacheIndexDataset(IndexDataset):

    def __init__(self, data_path, dictionary):
        super(CacheIndexDataset, self).__init__(data_path, dictionary)
        self.batches = self.batch_sampler()
        self.cache = {}

    def prefetch(self, indices):
        for index in indices:
            self.cache[index] = self.get_item(index)

    def get_item(self, index):
        pos = self.pointer[index]
        self.f_bin.seek(pos * 8)
        return np.frombuffer(self.f_bin.read(self.sent_lengths[index] * 8), dtype=np.int64)

    def __getitem__(self, index):
        if self.cache.get(index, False):
            data = self.cache[index]
            del data[index]
            return data
        else:
            return self.get_item(index)

    def support_prefetch(self):
        return True


class TextDataset(Seq2seqDataset):
    def __init__(self, data_path: str, dictionary: Dictionary):
        super(TextDataset, self).__init__(data_path, dictionary)

        self.dataset = []
        self.sent_lengths = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                self.dataset.append(line)
                self.sent_lengths.append(len(line))

        self.sent_lengths = np.array(self.sent_lengths)

    def __getitem__(self, index):
        return self.dictionary.encode_sentence(self.dataset[index])

    def batch_sampler(self, max_sentences=None, max_tokens=None):
        batches = single_dataset_batch(self.sent_lengths, max_tokens, max_sentences, self.num_tokens)
        return batches

    def __len__(self):
        return len(self.dataset)


def find_bin_tree_depth(seq_len):
    index = 1
    while True:
        lower_bound = int(math.pow(2, index - 1) - 1)
        high_bound = int(math.pow(2, index) - 1)
        if lower_bound < seq_len <= high_bound:
            # we add eos token at every non eos leaf node, so high_bound add 1 and index add 2
            index += 1
            high_bound = int(math.pow(2, index) - 1)
            return high_bound, index
        index += 1


class PairDataset(Dataset):
    """

    """

    def __init__(self, src_dataset, tgt_dataset):
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.shuffle = True

        self.mask_batch_data = False

    def __getitem__(self, index):
        src_data = self.src_dataset[index]
        tgt_data = self.tgt_dataset[index]
        return {
            "index": torch.tensor(index, dtype=torch.int64),
            "src": torch.tensor(src_data, dtype=torch.int64),
            "tgt": torch.tensor(tgt_data, dtype=torch.int64)
        }

    def __len__(self):
        return len(self.src_dataset)

    def num_tokens(self, index):
        return max(self.src_dataset.sent_lengths[index], self.tgt_dataset.sent_lengths[index])

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = np.maximum(self.tgt_dataset.sent_lengths[indices], self.src_dataset.sent_lengths[indices])
        return sizes

    def batch_sampler(self, max_sentences=-1, max_tokens=None) -> List:

        # indices = self.ordered_indices()
        # num_tokens_vec = self.num_tokens_vec(indices)
        # batches = batch_by_size_vec(indices, num_tokens_vec, max_tokens, max_sentences, 8)

        # --- code by wang --- #
        # sorted by target length, then source length

        ordered_indices = self.ordered_indices()

        # length_upper_limit = np.max(self.src_dataset.length[ordered_indices], self.tgt_dataset.length[ordered_indices])

        batches = batch_by_size(indices=ordered_indices,
                                num_tokens_fn=self.num_tokens,
                                max_sentences=max_sentences,
                                max_tokens=max_tokens,
                                required_batch_size_multiple=8)
        # --- code by wang --- #

        return batches

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)

        # sort by target length, then source length
        indices = indices[np.argsort(self.tgt_dataset.sent_lengths[indices], kind="mergesort")]  # noqa
        ordered_indices = indices[np.argsort(self.src_dataset.sent_lengths[indices], kind="mergesort")]  # noqa
        return ordered_indices

    def collate_fn(self, samples: List[Tuple[List, List]]) -> Dict:
        bsz = len(samples)
        id, src_len_list, tgt_len_list = [], [], []

        # targets = list(map(self.tgt_dataset.dictionary.decode_sentence, [sample["tgt"].numpy().tolist() for sample in samples]))

        src_max_len = max(sample["src"].size(0) for sample in samples)  # noqa
        tgt_max_len = max(sample["tgt"].size(0) for sample in samples) + 1  # noqa add bos token in the middle

        tree_max_len, level_num = find_bin_tree_depth(tgt_max_len)
        level_cum = [int(math.pow(2, index) - 1) for index in range(1, level_num + 1)]

        # src_tokens, tgt_tokens, previous_tgt_tokens initialize
        src_tokens = torch.zeros(size=(bsz, src_max_len), dtype=torch.int64).fill_(
            self.src_dataset.dictionary.padding_id)
        tgt_tokens = torch.zeros(size=(bsz, tree_max_len), dtype=torch.int64).fill_(
            self.tgt_dataset.dictionary.padding_id)
        previous_tgt_tokens = torch.zeros(size=(bsz, tree_max_len), dtype=torch.int64).fill_(
            self.tgt_dataset.dictionary.padding_id)
        eval_tgt_tokens = torch.zeros(size=(bsz, tgt_max_len), dtype=torch.int64).fill_(
            self.tgt_dataset.dictionary.padding_id) # ground truth

        # TODO: check if the assignment operation in-place or not

        for i, sample in enumerate(samples):
            index, src, tgt = sample["index"], sample["src"], sample["tgt"]  # noqa
            src_tokens[i][: len(src)] = src

            # 1 insert bos token
            insert_pos = int(len(tgt) / 2 + 1) if len(tgt) % 2 == 1 else int(len(tgt) / 2)

            bos_token = torch.zeros(size=(1,), dtype=torch.int64).fill_(self.tgt_dataset.dictionary.bos_id)
            tgt = torch.cat((tgt[:insert_pos], bos_token, tgt[insert_pos:]), dim=0)

            # 2 insert eos token
            # sample_tensor, sample_tensor_cp, tgt_len = self.data_process_tree(tgt, tree_max_len, level_num)
            sample_tensor, sample_tensor_cp, tgt_len = self.data_process_array(tgt, tree_max_len, level_cum)
            tgt_len_list.append(tgt_len)

            previous_tgt_tokens[i] = sample_tensor
            tgt_tokens[i] = sample_tensor_cp

            # insert bos token in tgt, but in BLEU postprocess stage removes all invalid tokens
            eval_tgt_tokens[i][: len(tgt)] = tgt

            id.append(index)
            src_len_list.append(len(src))

        src_len_list, ordered_index = torch.LongTensor(src_len_list).sort(descending=True)

        id = torch.LongTensor(id).index_select(dim=0, index=ordered_index)
        src_tokens = src_tokens.index_select(dim=0, index=ordered_index)
        tgt_tokens = tgt_tokens.index_select(dim=0, index=ordered_index)
        previous_tgt_tokens = previous_tgt_tokens.index_select(dim=0, index=ordered_index)
        eval_tgt_tokens = eval_tgt_tokens.index_select(dim=0, index=ordered_index)

        if self.mask_batch_data:
            previous_tgt_tokens = self.mask_batch(previous_tgt_tokens, level_num)

        # delete some unnecessary paddings
        cut_off = max(tgt_len_list)
        tgt_tokens = tgt_tokens[:, :cut_off]
        previous_tgt_tokens = previous_tgt_tokens[:, :cut_off]

        tgt_lengths = torch.tensor(tgt_len_list, dtype=torch.int64)

        ntokens = previous_tgt_tokens.eq(self.tgt_dataset.dictionary.mask_id).sum()

        batch = {
            "id": id,
            "nsentences": torch.tensor(len(samples), dtype=torch.int64),
            "ntokens": ntokens,
            "net_input": {"src_tokens": src_tokens,
                          "src_lengths": src_len_list,
                          "prev_tgt_tokens": previous_tgt_tokens},
            "target": tgt_tokens,
            "eval_target": eval_tgt_tokens,
            "tgt_lengths": tgt_lengths

        }
        return batch

    def mask_sibling(self, tree: Tree):
        tgt_tree = tree
        left_subtree_node = tgt_tree.get_subtree(tgt_tree.root.left)
        left_subtree_node = [node for node in left_subtree_node if node.data != self.tgt_dataset.dictionary.padding_id]

        right_subtree_node = tgt_tree.get_subtree(tgt_tree.root.right)
        right_subtree_node = [node for node in right_subtree_node if
                              node.data != self.tgt_dataset.dictionary.padding_id]

        mask_sample_list = []
        mask_sample_list.append(random.choice(left_subtree_node))
        mask_sample_list.append(random.choice(right_subtree_node))

        tgt_tree.level_order(tgt_tree.root)
        flatten = tgt_tree.flatten  # List[TreeNode]
        flatten_clone = deepcopy(flatten)  # List[TreeNode]

        # for i in range(6):
        #     print(flatten[int(math.pow(2, i)) - 1: int(math.pow(2, i + 1)) - 1])
        #     print("*" * 50)
        #     print("*" * 50)

        all_mask = []
        for mask_node in mask_sample_list:
            all_mask.append(mask_node)
            all_mask.append(tgt_tree.get_sibling(mask_node))

        # all_mask_clone = deepcopy(all_mask)
        # print(all_mask)

        for node in all_mask:
            tgt_tree.set_node(node, torch.tensor(self.tgt_dataset.dictionary.mask_id, dtype=torch.int64))

        for node in all_mask:
            children = tgt_tree.get_all_children(node)
            for child in children:
                tgt_tree.set_node(child, torch.tensor(self.tgt_dataset.dictionary.padding_id, dtype=torch.int64))

        return flatten, flatten_clone

    def mask_level(self, tree: Tree, level_num: int):
        tree.level_order(tree.root)
        flatten = tree.flatten  # List[TreeNode]
        flatten_clone = deepcopy(flatten)  # List[TreeNode]

        level_cum = [int(math.pow(2, index) - 1) for index in range(1, level_num + 1)]

        start = random.choice(level_cum[:-1])
        end = level_cum[level_cum.index(start) + 1]

        for node in flatten[start: end]:
            tree.set_node(node, torch.tensor(self.tgt_dataset.dictionary.mask_id, dtype=torch.int64))

        for node in flatten[start: end]:
            children = tree.get_all_children(node)
            for child in children:
                tree.set_node(child, torch.tensor(self.tgt_dataset.dictionary.padding_id, dtype=torch.int64))

        return flatten, flatten_clone

    def mask_level_bak(self, sequence: Tensor, level_cum: List):
        length = len(sequence)

        start = random.choice(level_cum[:-1])
        end = level_cum[level_cum.index(start) + 1]

        sequence_cp = deepcopy(sequence)
        tgt_sequence = torch.zeros(size=(length,), dtype=torch.int64).fill_(
            self.tgt_dataset.dictionary.padding_id)

        sequence[start: end] = self.tgt_dataset.dictionary.mask_id
        tgt_sequence[start: end] = sequence_cp[start: end]

        padding_num = 0
        for i in range(start, length):
            if (2 * i + 1) < length:
                sequence[2 * i + 1] = self.tgt_dataset.dictionary.padding_id
                padding_num += 1

            if (2 * i + 2) < length:
                sequence[2 * i + 2] = self.tgt_dataset.dictionary.padding_id
                padding_num += 1

        tgt_len = length - padding_num

        return sequence, tgt_sequence, tgt_len

    def mask_batch(self, tgt_tokens, level_num: int):

        level_cum = [int(math.pow(2, index) - 1) for index in range(1, level_num + 1)]

        start = random.choice(level_cum[:-1])
        end = level_cum[level_cum.index(start) + 1]

        tgt_tokens[:, start: end] = self.tgt_dataset.dictionary.mask_id
        tgt_tokens[:, end:] = self.tgt_dataset.dictionary.padding_id

        return tgt_tokens

    def add_eos_token(self, tree: Tree):
        # tree.level_order(tree.root)
        # eos_count = 0
        # for node in tree.flatten:
        #     if (node.left is None and node.right is not None) or (node.right is None and node.left is not None):
        #         eos_count += 1
        #
        # for i in range(eos_count):
        #     tree.fill_single_node(torch.tensor(self.tgt_dataset.dictionary.eos_id, dtype=torch.int64))
        tree.get_nodes_one_child(tree.root)  # O(logn)

        for n in tree.single_child_nodes:
            node_token = TreeNode(torch.tensor(self.tgt_dataset.dictionary.eos_id, dtype=torch.int64))
            if n.left is None:
                n.left = node_token
            elif n.right is None:
                n.right = node_token

        non_eos_leaf_node = [node for node in tree.get_leaf_nodes()
                             if node.data != torch.tensor(self.tgt_dataset.dictionary.eos_id, dtype=torch.int64)]

        for node in non_eos_leaf_node:
            node.left = TreeNode(torch.tensor(self.tgt_dataset.dictionary.eos_id, dtype=torch.int64))
            node.right = TreeNode(torch.tensor(self.tgt_dataset.dictionary.eos_id, dtype=torch.int64))

        return tree

    def data_process_tree(self, tgt, tree_max_len, level_num):
        tgt_tree = Tree()
        root = tgt_tree.sorted_array_to_BST(tgt)  # noqa O(logn)
        tgt_tree.set_root(root)

        tgt_tree = self.add_eos_token(tgt_tree)

        # 3 padding
        tgt_tree.level_order(tgt_tree.root)
        tgt_len = len(tgt_tree.flatten)

        tgt_tree.add_padding(torch.tensor(self.tgt_dataset.dictionary.padding_id, dtype=torch.int64), level_num)

        if not self.mask_batch_data:
            flatten, flatten_clone = self.mask_level(tgt_tree, level_num)
        else:
            tgt_tree.level_order(tgt_tree.root)
            flatten = tgt_tree.flatten  # List[TreeNode]
            flatten_clone = deepcopy(flatten)  # List[TreeNode]

        sample_tensor = torch.tensor([node.data for node in flatten], dtype=torch.int64)
        sample_tensor_cp = torch.tensor([node.data for node in flatten_clone], dtype=torch.int64)

        return sample_tensor, sample_tensor_cp, tgt_len

    def data_process_array(self, tgt, tree_max_len, level_cum: List):
        sample_tensor = Tree.sorted_array_to_list(tgt, tree_max_len, self.tgt_dataset.dictionary.padding_id)

        # add eos token
        length = len(sample_tensor)
        for j, data in enumerate(sample_tensor):
            if data != self.tgt_dataset.dictionary.padding_id and data != self.tgt_dataset.dictionary.eos_id:
                if (2 * j + 1) < length and sample_tensor[2 * j + 1] == self.tgt_dataset.dictionary.padding_id:
                    sample_tensor[2 * j + 1] = self.tgt_dataset.dictionary.eos_id

                if (2 * j + 2) < length and sample_tensor[2 * j + 2] == self.tgt_dataset.dictionary.padding_id:
                    sample_tensor[2 * j + 2] = self.tgt_dataset.dictionary.eos_id

        sample_tensor, sample_tensor_cp, tgt_len = self.mask_level_bak(sample_tensor, level_cum)

        return sample_tensor, sample_tensor_cp, tgt_len

    def __getstate__(self):
        state = self.__dict__
        return state

    def __setstate__(self, state):
        self.__dict__ = state


if __name__ == '__main__':
    d_en = Dictionary.load_dictionary_from_file('../test/dict.en')
    d_de = Dictionary.load_dictionary_from_file('../test/dict.de')

    en_dataset = IndexDataset('../test/test.en', max_tokens=4096)
    de_dataset = IndexDataset('../test/test.de', max_tokens=4096)

    dataset = PairDataset(en_dataset, de_dataset)
