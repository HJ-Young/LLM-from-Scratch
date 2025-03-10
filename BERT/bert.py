import torch
from torch import nn
import math


class AddNorm(nn.Module):
    def __init__(self, dropout, norm_shape):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_output, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_output)
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, valid_lens):
        attn_score = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if valid_lens is not None:
            mask = torch.arange(attn_score.size(-1))[None, :] < valid_lens[:, None]
            attn_score = attn_score.masked_fill(~mask, float("-inf"))
        attn_score = nn.functional.softmax(attn_score, dim=-1)
        return torch.bmm(self.dropout(attn_score), v)


class MultiHeadAttention(nn.Module):

    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.w_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.w_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.w_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.w_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)

    def transpose_qkv(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1).permute(0, 2, 1, 4)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def forward(self, q, k, v, valid_lens):
        q = self.transpose_qkv(self.w_q(q))
        k = self.transpose_qkv(self.w_k(k))
        v = self.transpose_qkv(self.w_v(v))

        attn = []

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        attn = self.attention(q, k, v, valid_lens)
        attn = attn.reshape(-1, self.num_heads, attn[0].shape[0], attn[0].shape[1]).permute(0, 2, 1, 3)
        attn.reshape(attn.shape[0], attn.shape[1], -1)
        return self.w_o(attn)


class EncoderBlock(nn.Module):

    def __init__(
        self,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        dropout,
        query_size=768,
        key_size=768,
        value_size=768,
        **kwargs,
    ):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attn_norm = AddNorm(dropout, norm_shape)
        self.attn = MultiHeadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.ffn_norm = AddNorm(dropout, norm_shape)

    def forward(self, X, valid_lens):
        Y = self.attn_norm(X, self.attn(X, X, X, X, valid_lens))
        return self.ffn_norm(Y, self.ffn(Y))


class BERTEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        max_len=1000,
        query_size=768,
        key_size=768,
        value_size=768,
        **kwargs,
    ):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for layer in range(num_layers):
            self.blks.add_module(
                f"block{layer}",
                EncoderBlock(
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    query_size,
                    key_size,
                    value_size,
                ),
            )

        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X += self.pos_encoding[:, : X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
