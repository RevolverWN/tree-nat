import inspect
import json
import math
import operator
import os
import pickle
import sys
import argparse
import itertools
from dataclasses import dataclass, field
from pprint import pprint
from typing import TypeVar, Callable, Any

from hydra.experimental import initialize, compose

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


# 训练阶段：
# 先找出最大长度，保证每一个样本能放到正中央
# 假如最大长度是10，补充到11，对每个样本如果是偶数，中间插入bos，如果是奇数，补充到偶数，再插入bos


# 1 生成速度太慢，随机mask一层，全部生成



