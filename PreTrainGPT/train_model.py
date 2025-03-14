from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os

from tqdm import tqdm

from config import train_args, model_args
from transformer import TranslationHead
from dataset import NMTDataset, inspect_dataset, load_dataset_builder

import sentencepiece as spm

from dataclasses import asdict
import sacrebleu

import time

import pandas as pd
from datasets import load_dataset

train_dataset = load_dataset("wmt/wmt17", split="train")
valid_dataset = load_dataset("wmt/wmt17", split="validation")
test_dataset = load_dataset("wmt/wmt17", split="test")


if __name__ == "__main__":
    assert os.path.exists(train_args.src_tokenizer_file), "should first run train_tokenizer.py to train the tokenizer"
    assert os.path.exists(train_args.tgt_tokenizer_path), "should first run train_tokenizer.py to train the tokenizer"
    source_tokenizer = spm.SentencePieceProcessor(model_file=train_args.src_tokenizer_file)
    target_tokenizer = spm.SentencePieceProcessor(model_file=train_args.tgt_tokenizer_path)
