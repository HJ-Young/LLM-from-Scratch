from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import Tensor, LongTensor
from typing import Tuple
from torch.utils.data import DataLoader
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

    def __getitem__(self, index:int) -> Tuple[list[int], list[int], list[str], list[str]]:
        row = self.ds[index]
        return (row["source_indices"], row["target_indices"], row["source"], row["target"])

    def __len__(self) -> int:
        return len(self.ds)
    
    def collate_fn(self, batch: list[Tuple[list[int], list[int], list[str], list[str]]]) -> Tuple[LongTensor, LongTensor, LongTensor]:
        source_indices = [x[0] for x in batch]
        target_indices = [x[1] for x in batch]
        source_text = [x[2] for x in batch]
        target_text = [x[3] for x in batch]

        source_indices = [torch.LongTensor(indices) for indices in source_indices]
        target_indices = [torch.LongTensor(indices) for indices in target_indices]
        source = pad_sequence(source_indices, padding_value=self.pad_idx, batch_first=True)
        target = pad_sequence(target_indices, padding_value=self.pad_idx, batch_first=True)

        labels = target[:, 1:].contiguous()
        target = target[:, :-1].contiguous()
        
        num_tokens = (labels != self.pad_idx).data.sum()

        return Batch(source, target, labels, num_tokens, source_text, target_text)