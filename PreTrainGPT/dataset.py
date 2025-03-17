from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import Tensor, LongTensor
from typing import Turple

from dataclasses import dataclass


@dataclass
class Batch:
    source: Tensor
    target: Tensor
    labels: Tensor
    num_tokens: int
    src_text: str = None
    tgt_text: str = None

class TranslationDataset:
    def __init__(self, ds: Dataset, pad_idx: int=0) -> None:
        self.ds = ds
        self.pad_idx = pad_idx

    def __getitem__(self, idx:int) -> Turple[list[int], list[int], list[str], list[str]]:
        row = self.ds[idx]
        return (row.source_indices, row.target_indices, row.source, row.target)

    def __len__(self) -> int:
        return len(self.ds)
    
    def collect_fn(self, batch: list[Turple[list[int], list[int], list[str], list[str]]]) -> Turple[LongTensor, LongTensor, LongTensor]:
        source_indices = [x[0] for x in batch]
        target_indices = [x[1] for x in batch]
        source_text = [x[2] for x in batch]
        target_text = [x[3] for x in batch]

        source_indices = [torch.LongTensor(indices) for indices in source_indices]
        target_indices = [torch.LongTensor(indices) for indices in target_indices]