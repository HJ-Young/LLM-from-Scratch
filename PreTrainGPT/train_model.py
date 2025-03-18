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
    model.train()
    total_loss = 0.0
    tqdm_iter = tqdm(data_loader)

    for source, target, labels, _ in tqdm_iter:
        source = source.to(device)
        target = target.to(device)
        labels = labels.to(device)

        logits = model(source, target)

        loss = criterion(logits, labels)
        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        description = f" TRAIN  loss={loss.item():.6f}, learning rate={scheduler.get_last_lr()[0]:.7f}"

        del loss

        tqdm_iter.set_description(description)

    avg_loss = total_loss / len(data_loader)

    return avg_loss

@torch.no_grad()
def evaluation(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    tqdm_iter = tqdm(data_loader)

    for source, target, labels, _ in tqdm_iter:
        source = source.to(device)
        target = target.to(device)
        labels = labels.to(device)

        logits = model(source, target)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        
        del loss

    avg_loss = total_loss / len(data_loader)

    return avg_loss
    


if __name__ == "__main__":
    assert os.path.exists(train_args.src_tokenizer_file), "should first run train_tokenizer.py to train the tokenizer"
    assert os.path.exists(train_args.tgt_tokenizer_path), "should first run train_tokenizer.py to train the tokenizer"
    source_tokenizer = Tokenizer.from_file(train_args.src_tokenizer_file)
    target_tokenizer = Tokenizer.from_file(train_args.tgt_tokenizer_path)
