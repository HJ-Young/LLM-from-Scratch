from collections import defaultdict
import jieba
from typing import Tuple


class BPETokenizer:
    def __init__(self, special_tokens=None):
        self.token_freqs = defaultdict(int)
        self.token_to_idx: dict[str, int] = {}
        self.idx_to_token: dict[int, str] = {}
        self.merges = {}

        if special_tokens is None:
            special_tokens = []
        special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
            "Ġ",  # stands for </w>
        ] + special_tokens

        for special_token in special_tokens:
            self._add_token(special_token)

    def _add_token(self, token: str) -> None:
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    @property
    def vocab_size(self):
        return len(self.token_to_idx)

    def _learn_vocab(self, corpus: list[str]) -> None:
        for sentence in corpus:
            sentence = sentence.lower()
            words = [word + "Ġ" for word in jieba.cut(sentence) if word != " "]
            for word in words:
                self.token_freqs[word] += 1

    def _compute_pair_freqs(self, splits) -> dict[Tuple, int]:
        pair_freqs = defaultdict(int)

        for word, freq in self.token_freqs.item():
            split = splits[word]
            if len(split) == 1:
                continue

            for i in range(len(split) - 1):
                pair = split[i] + split[i + 1]
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(self, splits, a: str, b: str):
        for word in self.token_freqs:
            split = splits[word]
            if len(split == 1):
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def _merge_vocab(self, vocab_size, splits):
        merges = {}

        while self.vocab_size < vocab_size:
            pair_freqs = self._compute_pair_freqs(splits)
            best_pair = None
            max_freq = 0

            for pair, freq in pair_freqs.item():
                if freq > max_freq:
                    best_pair = pair
                    max_freq = freq

            if best_pair is None:
                print(f"No token, vocab size is {vocab_size}")
                break

            self._merge_pair(splits, best_pair[0], best_pair[1])
            merges[best_pair] = best_pair[0] + best_pair[1]
            self._add_token(best_pair[0] + best_pair[1])
        return merges

    def train(self, corpus, vocab_size):
        self._learn_vocab(corpus)

        splits = {word: [c for c in word] for word in self.token_freqs.keys()}

        for split in splits.values():
            for c in split:
                self._add_token(c)

        self.merges = self._merge_vocab(vocab_size, splits)

    def tokenize(self, text):
        text = text.lower()

        words = [w + "Ġ" for w in jieba.cut(text) if w != " "]
        splits = [[c for c in word] for word in words]

        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split
        return sum(splits, [])

    def _convert_idx_to_token(self, idx: int) -> str:
        return self.idx_to_token[idx]

    def _convert_token_to_idx(self, token: str) -> int:
        return self.token_to_idx[token]

    def _convert_idxs_to_tokens(self, idxs: list[int]) -> list[str]:
        return [self.idx_to_token[idx] for idx in idxs]

    def _convert_tokens_to_idxs(self, tokens: list[str]) -> list[int]:
        return [self.token_to_idx[token] for token in tokens]

    def encode(self, text):
        tokens = self.tokenize(text)
        return self._convert_tokens_to_idxs(tokens)

    def clean_up_tokenization(self, out_string: str) -> str:
        out_string = (
            out_string.replace("Ġ", " ")
            .replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    def decode(self, idxs):
        tokens = self._convert_idxs_to_tokens(idxs)
        return self.clean_up_tokenization("".join(tokens))
