import torch
from torch import nn
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int = 512, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = 10000 ** (-torch.arange(0, dim, 2) / dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, X):
        X = X + self.pe[:, : X.size(1)]
        return self.dropout(X)


class MHA(nn.Module):
    def __init__(self, n_embed, n_hiddens, n_heads, dropout, **kwargs):
        super(MHA, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.w_q = nn.Linear(n_embed, n_hiddens)
        self.w_k = nn.Linear(n_embed, n_hiddens)
        self.w_v = nn.Linear(n_embed, n_hiddens)
        self.w_o = nn.Linear(n_hiddens, n_hiddens)
        self.dropout = nn.Dropout(p=dropout)

    def _attention(self, Q, K, V, mask=None, keep_attention=False):
        score = torch.matmul(Q, K) / math.sqrt(Q.shape[-1])
        if mask is not None:
            score.masked_fill(mask == 0, float("-inf"))

        weights = self.dropout(score, dim=-1)
        output = torch.matmul(weights, V)

        if keep_attention:
            self._weights = weights
        else:
            del weights
        return output

    def forward(self, query, key, value, mask, keep_attention=False):
        Q, K, V = self.w_q(query), self.w_k(key), self.w_v(value)
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        Q = Q.reshape(Q.shape[0], Q.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        K = K.reshape(K.shape[0], K.shape[1], self.n_heads, -1).permute(0, 2, 3, 1)
        V = V.reshape(V.shape[0], V.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)

        attn = self._attention(Q, K, V, mask, keep_attention)
        attn = attn.transpose(1, 2)
        attn = attn.reshape(batch_size, seq_len, -1)

        del Q
        del K
        del V
        return self.w_o(attn)


class LayerNorm(nn.Module):
    def __init__(self, gamma, beta, eps=1e-9):
        super(LayerNorm, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

    def forward(self, X):
        X = X - X.mean(dim=-1, keepdims=True)
        X = X / (torch.sqrt(X.pow(2).mean(dim=-1, keepdims=True)) + self.eps)
        return self.gamma * X + self.beta
    

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_n_input, ffn_n_hiddens, ffn_n_output, )
