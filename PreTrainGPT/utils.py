from torch import Tensor
import torch.nn as nn

import pandas as pd

from datasets import Dataset

import os

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from tokenizers import Tokenizer


class LabelSmoothLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, pad_idx: int = 0) -> None:
        super(LabelSmoothLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing)

    def forward(self, logits: Tensor, labels:Tensor):
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
    ) -> None:
        self.factor = factor
        self.n_params = len(optimizer.param_groups)
        self.warmup_steps = warmup_steps
        self.dim = dim
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        lr = (
            self.factor * (self.factor ** (-0.5)) * min(self._step_count ** (-0.5), self._step_count * self.warmup_steps ** (-1.5))
        )
        return [lr] * self.n_params


def make_dirs(dirpath: str) -> None:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def build_ds(ds: Dataset, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer, batch_size: int) -> Dataset:

    def tokenize_function(example):
        source = example['translation']["en"]
        target = example['translation']["zh"]

        source_encoding = src_tokenizer.encode(source).ids
        target_encoding = tgt_tokenizer.encode(target).ids
        return {
            'source': source,
            'target': target,
            'source_indices': source_encoding,
            'target_indices': target_encoding
        }

    ds = ds.map(tokenize_function, remove_columns=ds.column_names)
    return ds