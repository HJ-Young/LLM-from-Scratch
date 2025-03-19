from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os

from tqdm import tqdm

from config import train_args, model_args
from transformer import TranslationHead
from dataset import TranslationDataset
from tokenizers import Tokenizer
from datasets import load_dataset, load_from_disk
from tokenizers.processors import TemplateProcessing

import sacrebleu
from utils import build_ds, LabelSmoothLoss, WarmUpScheduler



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


    for batch in tqdm_iter:
        source = batch.source.to(device)
        target = batch.target.to(device)
        labels = batch.labels.to(device)

        print(source.shape)
        print(target.shape)

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
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    tqdm_iter = tqdm(data_loader)

    for batch in tqdm_iter:
        source = batch.source.to(device)
        target = batch.target.to(device)
        labels = batch.labels.to(device)

        logits = model(source, target)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        del loss

    avg_loss = total_loss / len(data_loader)

    return avg_loss
    
@torch.no_grad()
def calculate_bleu(
    model: TranslationHead,
    tgt_tokenizer: Tokenizer,
    data_loader: DataLoader,
    max_len: int,
    device: torch.device,
    save_result: bool = False,
    save_path: str = "result.txt",
    generation_mode: str = "greedy_search",
) -> float:
    candidates = []
    references = []

    model.eval()

    for batch in tqdm(data_loader):
        source = batch.source.to(device)

        token_indices = model.translate(
            source,
            max_gen_len=max_len,
            generation_mode=generation_mode,
        )
        token_indices = token_indices.cpu().tolist()
        candidates.extend(tgt_tokenizer.decode(token_indices))

        references.extend(batch.tgt_text)

    if save_result:
        with open(save_path, "w", encoding="utf-8") as f:
            for i in range(len(references)):
                f.write(
                    f"idx: {i:5} | reference: {references[i]} | candidate: {candidates[i]} \n"
                )

    bleu = sacrebleu.corpus_bleu(candidates, [references], tokenize="zh")

    return float(bleu.score)


if __name__ == "__main__":
    assert os.path.exists(train_args.src_tokenizer_file), "should first run train_tokenizer.py to train the tokenizer"
    assert os.path.exists(train_args.tgt_tokenizer_path), "should first run train_tokenizer.py to train the tokenizer"
    source_tokenizer = Tokenizer.from_file(train_args.src_tokenizer_file)
    target_tokenizer = Tokenizer.from_file(train_args.tgt_tokenizer_path)

    source_tokenizer.enable_padding(pad_id=0)
    target_tokenizer.enable_padding(pad_id=0)

    source_tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", source_tokenizer.token_to_id("[BOS]")),
            ("[EOS]", source_tokenizer.token_to_id("[EOS]")),
        ],
    )
    source_tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')

    target_tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", target_tokenizer.token_to_id("[BOS]")),
            ("[EOS]", target_tokenizer.token_to_id("[EOS]")),
        ],
    )
    target_tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')

    if train_args.only_test:
        train_args.use_wandb = False

    if train_args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"source tokenizer size: {source_tokenizer.get_vocab_size()}")
    print(f"target tokenizer size: {target_tokenizer.get_vocab_size()}")

    pad_idx = 0

    train_dataframe_path = os.path.join(
        train_args.dataset_path, train_args.dataframe_file.format("train")
    )
    test_dataframe_path = os.path.join(
        train_args.dataset_path, train_args.dataframe_file.format("test")
    )
    valid_dataframe_path = os.path.join(
        train_args.dataset_path, train_args.dataframe_file.format("dev")
    )

    if os.path.exists(train_dataframe_path) and train_args.use_dataframe_cache:
        train_df, test_df, valid_df = (
            load_from_disk(train_dataframe_path),
            load_from_disk(test_dataframe_path),
            load_from_disk(valid_dataframe_path),
        )
        print("Loads cached dataframes.")
    
    else:
        print("Create new dataframes...")
        train_df = build_ds(load_dataset("./data/zh-en", split="train"), source_tokenizer, target_tokenizer)
        train_df.save_to_disk(train_dataframe_path)
        print("Create valid dataframe")

        test_df = build_ds(load_dataset("./data/zh-en", split="test"), source_tokenizer, target_tokenizer)
        test_df.save_to_disk(test_dataframe_path)
        print("Create train dataframe")

        valid_df = build_ds(load_dataset("./data/zh-en", split="validation"), source_tokenizer, target_tokenizer)
        valid_df.save_to_disk(valid_dataframe_path)
        print("Create test dataframe")

    train_dataset, test_dataset, valid_dataset = (
        TranslationDataset(train_df, pad_idx=pad_idx),
        TranslationDataset(test_df, pad_idx=pad_idx),
        TranslationDataset(valid_df, pad_idx=pad_idx),
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=train_args.train_batch_size,
        collate_fn=train_dataset.collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=train_args.train_batch_size,
        collate_fn=valid_dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=train_args.train_batch_size,
        collate_fn=test_dataset.collate_fn,
    )

    model_args.source_vocab_size = source_tokenizer.get_vocab_size()
    model_args.target_vocab_size = target_tokenizer.get_vocab_size()

    model = TranslationHead(
        model_args,
        pad_idx=0,
        bos_idx=2,
        eos_idx=3,
    ).to(device)

    print(model)

    if train_args.use_wandb:
        import wandb

        wandb.init(
            project="transformer",
            config={
                "architecture": "Transformer",
                "dataset": "en-zh",
                "epochs": train_args.num_epochs
            }
        )

    train_criterion = LabelSmoothLoss(train_args.label_smoothing, model_args.pad_idx)
    valid_criterion = LabelSmoothLoss(pad_idx=model_args.pad_idx)

    optimizer = torch.optim.Adam(
        model.parameters(), betas=train_args.betas, eps=train_args.eps
    )
    scheduler = WarmUpScheduler(
        optimizer,
        warmup_steps=train_args.warmup_steps,
        dim=model_args.d_model,
        factor=train_args.warmup_factor,
    )


    best_score = float("inf")

    print(f"begin train with arguments: {train_args}")

    print(f"total train steps: {len(train_dataloader) * train_args.num_epochs}")

    if not train_args.only_test:
        for epoch in range(train_args.num_epochs):
            train_loss = train(
                model,
                train_dataloader,
                train_criterion,
                optimizer,
                device,
                train_args.grad_clipping,
                scheduler,
            )

            valid_loss = evaluate(model, valid_dataloader, valid_criterion, device)


            print(
                f"end of epoch {epoch+1:3d} | train loss: {train_loss:.4f} | valid loss: {valid_loss:.4f}"
            )
            if train_args.use_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                    }
                )

            if valid_loss < best_score:
                best_score = valid_loss
                print(f"Save model with best valid_loss :{valid_loss:.2f}")

                torch.save(model.state_dict(), train_args.model_save_path)
            else:
                print(f"Now score is {best_score}")


    model.load_state_dict(torch.load(train_args.model_save_path, weights_only=True))
    test_loss = evaluate(model, test_dataloader, valid_criterion, device)
    test_bleu_score = calculate_bleu(
        model,
        target_tokenizer,
        test_dataloader,
        train_args.max_gen_len,
        device,
        save_result=True,
        generation_mode=train_args.generation_mode,
    )
    print(f"Test LOSS: {test_loss: .4f} Test bleu score: {test_bleu_score:.2f}")


    
