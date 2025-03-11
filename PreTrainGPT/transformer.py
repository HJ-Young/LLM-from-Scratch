import torch
from torch import nn, Tensor
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
    def __init__(self, n_hiddens, n_heads, dropout, **kwargs):
        super(MHA, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.w_q = nn.Linear(n_hiddens, n_hiddens)
        self.w_k = nn.Linear(n_hiddens, n_hiddens)
        self.w_v = nn.Linear(n_hiddens, n_hiddens)
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
    def __init__(self, features, eps=1e-9):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, X):
        X = X - X.mean(dim=-1, keepdims=True)
        X = X / (torch.sqrt(X.pow(2).mean(dim=-1, keepdims=True)) + self.eps)
        return self.gamma * X + self.beta


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_n_input, ffn_n_hiddens, dropout):
        super(PositionWiseFFN, self).__init__()
        self.ffn1 = nn.Linear(ffn_n_input, ffn_n_hiddens)
        self.ffn2 = nn.Linear(ffn_n_hiddens, ffn_n_input)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X):
        return self.ffn2(self.dropout(nn.functional.relu(self.ffn1(X))))


class EncoderBlock(nn.Module):
    def __init__(self, n_hiddens, n_heads, ffn_n_hiddens, dropout, pre_ln=False):
        self.attn = MHA(n_hiddens, n_heads, dropout)
        self.attn_ln = LayerNorm(n_hiddens)

        self.ffn = PositionWiseFFN(n_hiddens, ffn_n_hiddens, dropout)
        self.ffn_ln = LayerNorm(n_hiddens)

        self.pre_ln = pre_ln
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, X, mask=None, keep_attention=False):
        if self.pre_ln:
            X = X + self.dropout1(self.attn(self.attn_ln(X), self.attn_ln(X), self.attn_ln(X), mask, keep_attention))
            X = X + self.dropout2(self.ffn(self.ffn_ln(X)))
        else:
            X = self.attn_ln(X + self.dropout2(self.attn(X, X, X, mask, keep_attention)))
            X = self.ffn_ln(X + self.dropout2(self.ffn(X)))
        return X


class Encoder(nn.Module):
    def __init__(self, n_hiddens, n_heads, ffn_n_hiddens, n_layers, dropout, pre_ln=False):
        super(Encoder, self).__init__()
        self.blks = nn.Sequential()
        for layer in range(n_layers):
            self.blks.add_module(
                f"Encoder Block {layer}", EncoderBlock(n_hiddens, n_heads, ffn_n_hiddens, dropout, pre_ln)
            )

        self.ln = LayerNorm(n_hiddens)

    def forward(self, X, mask, keep_attention=False):
        for blk in self.blks:
            X = blk(X, mask, keep_attention)

        return self.ln(X)


def make_mask(src, pad_idx: int = 0):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask


class DecoderBlock(nn.Module):
    def __init__(self, n_hiddens, n_heads, ffn_n_hiddens, n_layers, dropout, pre_ln=False):
        self.masked_attn = MHA(n_hiddens, n_heads, dropout)
        self.masked_attn_ln = LayerNorm(n_hiddens)

        self.attn = MHA(n_hiddens, n_heads, dropout)
        self.attn_ln = LayerNorm(n_hiddens)

        self.ffn = PositionWiseFFN(n_hiddens, ffn_n_hiddens, dropout)
        self.ffn_ln = LayerNorm(n_hiddens)

        self.pre_ln = pre_ln
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, X, enc_output, mask, enc_mask, keep_attention):
        if self.pre_ln:
            X = X + self.masked_attn_ln(self.dropout1(self.masked_attn(X, X, X, mask, keep_attention)))
            X = X + self.attn_ln(self.dropout2(self.attn(X, enc_output, enc_output, mask, keep_attention)))
            X = X + self.ffn_ln(self.dropout3(self.ffn(X)))
        else:
            X = self.masked_attn_ln(X + self.dropout1(self.masked_attn(X, X, X, mask, keep_attention)))
            X = self.attn_ln(X + self.dropout2(self.attn(X, enc_output, enc_output, mask, keep_attention)))
            X = self.ffn_ln(X + self.dropout3(self.ffn(X)))
        return X


class Decoder(nn.Module):
    def __init__(self, n_hiddens, n_heads, ffn_n_hiddens, n_layers, dropout, pre_ln=False):
        super(Decoder, self).__init__()
        self.blks = nn.Sequential()
        for layer in range(n_layers):
            self.blks.add_module(
                f"Dncoder Block {layer}", DecoderBlock(n_hiddens, n_heads, ffn_n_hiddens, dropout, pre_ln)
            )

        self.ln = LayerNorm(n_hiddens)

    def forward(self, X, enc_output, mask, enc_mask, keep_attention=False):
        for blk in self.blks:
            X = blk(X, enc_output, mask, enc_mask, keep_attention)

        return self.ln(X)


def make_tgt_mask(tgt, pad_idx: int = 0):
    seq_len = tgt.size()[-1]
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    subseq_mask = torch.tril(torch.ones((seq_len, seq_len))).bool()

    tgt_mask = tgt_mask & subseq_mask

    return tgt_mask


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_positions: int = 5000,
        pad_idx: int = 0,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.src_embedding = nn.Embedding(source_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(target_vocab_size, d_model)

        self.enc_pos = PositionalEncoding(d_model, dropout, max_positions)
        self.dec_pos = PositionalEncoding(d_model, dropout, max_positions)

        self.encoder = Encoder(d_model, num_encoder_layers, n_heads, d_ff, dropout, norm_first)
        self.decoder = Decoder(d_model, num_decoder_layers, n_heads, d_ff, dropout, norm_first)

        self.pad_idx = pad_idx

    def encode(self, src: Tensor, src_mask: Tensor = None, keep_attentions: bool = False) -> Tensor:
        src_embed = self.enc_pos(self.src_embedding(src))
        return self.encoder(src_embed, src_mask, keep_attentions)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor = None,
        memory_mask: Tensor = None,
        keep_attentions: bool = False,
    ):

        # tgt_embed (batch_size, tgt_seq_length, d_model)
        tgt_embed = self.dec_pos(self.tgt_embedding(tgt))
        # logits (batch_size, tgt_seq_length, d_model)
        logits = self.decoder(tgt_embed, memory, tgt_mask, memory_mask, keep_attentions)

        return logits

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        keep_attentions: bool = False,
    ):
        memory = self.encode(src, src_mask, keep_attentions)
        return self.decode(tgt, memory, tgt_mask, src_mask, keep_attentions)


class TranslationHead(nn.Module):
    def __init__(self, config: ModelArugment, pad_idx: int, bos_idx: int, eos_idx: int) -> None:
        super().__init__()
        self.config = config

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        self.transformer = Transformer(**asdict(config))
        self.lm_head = nn.Linear(config.d_model, config.target_vocab_size, bias=False)
        self.reset_parameters()

    def forward(
        self, src: Tensor, tgt: Tensor, src_mask: Tensor = None, tgt_mask: Tensor = None, keep_attentions: bool = False
    ) -> Tensor:
        if src_mask is None and tgt_mask is None:
            src_mask, tgt_mask = self.create_masks(src, tgt, self.pad_idx)
        output = self.transformer(src, tgt, src_mask, tgt_mask, keep_attentions)

        return self.lm_head(output)

    @torch.no_grad()
    def translate(
        self,
        src: Tensor,
        src_mask: Tensor = None,
        max_gen_len: int = 60,
        num_beams: int = 3,
        keep_attentions: bool = False,
        generation_mode: str = "greedy_search",
    ):
        if src_mask is None:
            src_mask = self.create_masks(src, pad_idx=self.pad_idx)[0]
        generation_mode = generation_mode.lower()
        if generation_mode == "greedy_search":
            return self._greedy_search(src, src_mask, max_gen_len, keep_attentions)
        else:
            return self._beam_search(src, src_mask, max_gen_len, num_beams, keep_attentions)
