from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os

from tqdm import tqdm

from config import train_args, model_args
from transformer import TranslationHead
from dataset import TranslationDataset

from tokenizers import Tokenizer
from datasets import load_dataset

from dataclasses import asdict
import sacrebleu

import time


train_dataset = load_dataset("wmt/wmt17", split="train")
valid_dataset = load_dataset("wmt/wmt17", split="validation")
test_dataset = load_dataset("wmt/wmt17", split="test")

def train(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip: float,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> float: 
    


if __name__ == "__main__":
    assert os.path.exists(train_args.src_tokenizer_file), "should first run train_tokenizer.py to train the tokenizer"
    assert os.path.exists(train_args.tgt_tokenizer_path), "should first run train_tokenizer.py to train the tokenizer"
    source_tokenizer = Tokenizer.from_file(train_args.src_tokenizer_file)
    target_tokenizer = load_dataset(model_file=train_args.tgt_tokenizer_path)
