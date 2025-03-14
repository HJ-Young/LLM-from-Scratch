from torch import Tensor
import torch
import torch.nn as nn

import pandas as pd

from collections import UserDict

import os
import json
from tqdm import tqdm
import numpy as np


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

from zhconv import convert
import sentencepiece as spm
import torch.nn.functional as F


class LabelSmoothLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, pad_idx: int = 0) -> None:
        super(LabelSmoothLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing)

    def forward(self, logits, labels):
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1).long()
        return self.loss(logits, labels)


class WarmUpScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        dim: int,
        factor: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch, verbose)
        self.factor = factor
        self.n_params = len(optimizer.param_groups)
        self.warmup_steps = warmup_steps
        self.dim = dim

    def get_lr(self) -> list[float]:
        lr = (
            self.factor
            * self.factor ** (-0.5)
            * min(self._step_count ** (-0.5), self._step_count * self.warmup_steps ** (-1.5))
        )
        return [lr] * self.n_params


def make_dirs(dirpath: str) -> None:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
