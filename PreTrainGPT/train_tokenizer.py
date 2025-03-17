from concurrent.futures import ProcessPoolExecutor
import os
from config import train_args, model_args
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


train_dataset = load_dataset("/home/work/LLM-from-Scratch/data", "zh-en", split="train")["translation"]
valid_dataset = load_dataset("/home/work/LLM-from-Scratch/data", "zh-en", split="validation")["translation"]
test_dataset = load_dataset("/home/work/LLM-from-Scratch/data", "zh-en", split="test")["translation"]


def get_mt_pairs(datasets):
    zh_sentence = []
    en_sentence = []
    for ds in datasets:
        for pair in ds:
            zh_sentence.append(pair["zh"] + "\n")
            en_sentence.append(pair["en"] + "\n")
    assert len(zh_sentence) == len(en_sentence)
    print(f"the total number of sentences: {len(zh_sentence)}")

    return zh_sentence, en_sentence


def train_bpe(
    input_files: list[str],
    save_path: str,
    vocab_size: int,
    pad_token: str = "[PAD]",
    unk_token: str = "[UNK]",
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
    mask_token: str = "[MASK]",
):

    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    trainer = BpeTrainer(vocab_size=vocab_size,special_tokens=[unk_token, cls_token, sep_token, pad_token, mask_token])

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(input_files, trainer)
    tokenizer.save(save_path)


def train_tokenizer(
    src_corpus_path: str,
    tgt_corpus_path: str,
    src_vocab_size: int,
    tgt_vocab_size: int,
) -> None:
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(train_bpe, [src_corpus_path], train_args.src_tokenizer_file, src_vocab_size),
            executor.submit(train_bpe, [tgt_corpus_path], train_args.tgt_tokenizer_path, tgt_vocab_size),
        ]

        for future in futures:
            future.result()

    source_tokenizer = Tokenizer.from_file(train_args.src_tokenizer_file)

    source_text = """
    Tesla is recalling nearly all 2 million of its cars on US roads to limit the use of its 
    Autopilot feature following a two-year probe by US safety regulators of roughly 1,000 crashes 
    in which the feature was engaged. The limitations on Autopilot serve as a blow to Tesla’s efforts 
    to market its vehicles to buyers willing to pay extra to have their cars do the driving for them.
    """

    source_token = source_tokenizer.encode(source_text)
    print(source_token.tokens)
    ids = source_token.ids
    print(ids)
    print(source_tokenizer.decode(ids))

    target_text = """
        新华社北京1月2日电（记者丁雅雯、李唐宁）2024年元旦假期，旅游消费十分火爆。旅游平台数据显示，旅游相关产品订单量大幅增长，“异地跨年”“南北互跨”成关键词。
        业内人士认为，元旦假期旅游“开门红”彰显消费潜力，预计2024年旅游消费有望保持上升势头。
    """

    target_tokenizer = Tokenizer.from_file(train_args.tgt_tokenizer_path)
    target_token = target_tokenizer.encode(target_text)
    print(target_token.tokens)
    ids = target_token.ids
    print(ids)
    print(target_tokenizer.decode(ids))


if __name__ == "__main__":
    os.makedirs(train_args.dataset_path, exist_ok=True)
    os.makedirs(train_args.tokenizer_save_dir, exist_ok=True)

    train_tokenizer(
        f"{train_args.dataset_path}/corpus.en",
        f"{train_args.dataset_path}/corpus.zh",
        src_vocab_size=model_args.source_vocab_size,
        tgt_vocab_size=model_args.target_vocab_size,
    )
