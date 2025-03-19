import torch
from torch import nn, Tensor
import math
import numpy as np
from dataclasses import asdict
from config import ModelArugment
from typing import Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int = 512, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(- (torch.arange(0, dim, 2) * (math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MHA(nn.Module):
    def __init__(
            self,
            n_hiddens: int = 512,
            n_heads: int = 8,
            dropout: float = 0.1,
            bias: bool = True,
        ):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = n_hiddens // n_heads
        self.w_q = nn.Linear(n_hiddens, n_hiddens, bias=bias)
        self.w_k = nn.Linear(n_hiddens, n_hiddens, bias=bias)
        self.w_v = nn.Linear(n_hiddens, n_hiddens, bias=bias)
        self.w_o = nn.Linear(n_hiddens, n_hiddens, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def _attention(self, Q, K, V, mask=None, keep_attention=False):
        score = torch.matmul(Q, K) / math.sqrt(Q.shape[-1])
        if mask is not None:
            score.masked_fill(mask == 0, float("-inf"))

        score = nn.functional.softmax(score, dim=-1)

        weights = self.dropout(score)
        output = torch.matmul(weights, V)

        if keep_attention:
            self.weights = weights
        else:
            del weights
        return output

    def forward(self, query, key, value, mask, keep_attention=False):
        Q, K, V = self.w_q(query), self.w_k(key), self.w_v(value)
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        Q = Q.reshape(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        V = V.reshape(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)

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

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdims=True)
        x = x / (torch.sqrt(x.pow(2).mean(dim=-1, keepdims=True)) + self.eps)
        return self.gamma * x + self.beta


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_n_input, ffn_n_hiddens, dropout):
        super(PositionWiseFFN, self).__init__()
        self.ffn1 = nn.Linear(ffn_n_input, ffn_n_hiddens)
        self.ffn2 = nn.Linear(ffn_n_hiddens, ffn_n_input)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.ffn2(self.dropout(nn.functional.relu(self.ffn1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, n_hiddens: int, n_heads: int, ffn_n_hiddens: int, dropout: float, bias:bool=False, pre_ln=False):
        super().__init__()
        self.attn = MHA(n_hiddens, n_heads, dropout, bias)
        self.attn_ln = LayerNorm(n_hiddens)

        self.ffn = PositionWiseFFN(n_hiddens, ffn_n_hiddens, dropout)
        self.ffn_ln = LayerNorm(n_hiddens)

        self.pre_ln = pre_ln
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, mask=None, keep_attention=False):
        if self.pre_ln:
            x = x + self.dropout1(self.attn(self.attn_ln(x), self.attn_ln(x), self.attn_ln(x), mask, keep_attention))
            x = x + self.dropout2(self.ffn(self.ffn_ln(x)))
        else:
            x = self.attn_ln(x + self.dropout2(self.attn(x, x, x, mask, keep_attention)))
            x = self.ffn_ln(x + self.dropout2(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, n_hiddens: int, n_heads: int, ffn_n_hiddens: int, n_layers:int, dropout: float, bias=True, pre_ln=False):
        super(Encoder, self).__init__()
        self.blks = nn.Sequential()
        for layer in range(n_layers):
            self.blks.add_module(
                f"Encoder Block {layer}", EncoderBlock(n_hiddens, n_heads, ffn_n_hiddens, dropout, bias, pre_ln)
            )

        self.ln = LayerNorm(n_hiddens)

    def forward(self, x: Tensor, mask: Tensor, keep_attention: bool=False):
        for i, blk in enumerate(self.blks):
            x = blk(x, mask, keep_attention)

        return self.ln(x)


def make_mask(src: Tensor, pad_idx: int = 0):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask


class DecoderBlock(nn.Module):
    def __init__(self, n_hiddens: int, n_heads: int, ffn_n_hiddens: int, dropout: float, bias: bool=False, pre_ln=False):
        super().__init__()
        self.masked_attn = MHA(n_hiddens, n_heads, dropout, bias)
        self.masked_attn_ln = LayerNorm(n_hiddens)

        self.attn = MHA(n_hiddens, n_heads, dropout, bias)
        self.attn_ln = LayerNorm(n_hiddens)

        self.ffn = PositionWiseFFN(n_hiddens, ffn_n_hiddens, dropout)
        self.ffn_ln = LayerNorm(n_hiddens)

        self.pre_ln = pre_ln
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, enc_output: Tensor, mask:Tensor, enc_mask:Tensor, keep_attention:bool):
        if self.pre_ln:
            x = x + self.masked_attn_ln(self.dropout1(self.masked_attn(x, x, x, mask, keep_attention)))
            x = x + self.attn_ln(self.dropout2(self.attn(x, enc_output, enc_output, enc_mask, keep_attention)))
            x = x + self.ffn_ln(self.dropout3(self.ffn(x)))
        else:
            x = self.masked_attn_ln(x + self.dropout1(self.masked_attn(x, x, x, mask, keep_attention)))
            x = self.attn_ln(x + self.dropout2(self.attn(x, enc_output, enc_output, enc_mask, keep_attention)))
            x = self.ffn_ln(x + self.dropout3(self.ffn(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, n_hiddens: int, n_heads: int, ffn_n_hiddens: int, n_layers: int, dropout: float, bias: bool=False, pre_ln: bool=False):
        super(Decoder, self).__init__()
        self.blks = nn.Sequential()
        for layer in range(n_layers):
            self.blks.add_module(
                f"Dncoder Block {layer}", DecoderBlock(n_hiddens, n_heads, ffn_n_hiddens, dropout, bias, pre_ln)
            )

        self.ln = LayerNorm(n_hiddens)

    def forward(self, x, enc_output, mask, enc_mask, keep_attention=False):
        for i, blk in enumerate(self.blks):
            x = blk(x, enc_output, mask, enc_mask, keep_attention)

        return self.ln(x)


def make_tgt_mask(tgt: Tensor, pad_idx: int = 0):
    # tgt (batch_size * n_heads, seq_len, n_hiddens)
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
        attention_bias: bool=True,
    ) -> None:
        super().__init__()
        self.src_embedding = nn.Embedding(source_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(target_vocab_size, d_model)

        self.enc_pos = PositionalEncoding(d_model, dropout, max_positions)
        self.dec_pos = PositionalEncoding(d_model, dropout, max_positions)

        self.encoder = Encoder(d_model, n_heads, d_ff, num_encoder_layers, dropout, attention_bias, norm_first)
        self.decoder = Decoder(d_model, n_heads, d_ff, num_decoder_layers, dropout, attention_bias, norm_first)

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
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor=None,
        tgt_mask: Tensor=None,
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

    
    def reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_subsequent_mask(seq_len: int, device: torch.device) -> Tensor:
        subseq_mask = torch.tril(
            torch.ones((1, 1, seq_len, seq_len), device=device)
        ).int()
        return subseq_mask
    
    @staticmethod
    def create_masks(
        src: Tensor, tgt: Tensor = None, pad_idx: int = 0
    ) -> Tuple[Tensor, Tensor]:
        src_mask = src.ne(pad_idx).long().unsqueeze(1).unsqueeze(2)

        tgt_mask = None

        if tgt is not None:
            tgt_seq_len = tgt.size()[-1]
            tgt_mask = tgt.ne(pad_idx).long().unsqueeze(1).unsqueeze(2)

            subseq_mask = TranslationHead.generate_subsequent_mask(
                tgt_seq_len, src.device
            )

            tgt_mask = tgt_mask & subseq_mask

        return src_mask, tgt_mask

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
        keep_attentions: bool = False,
        generation_mode: str = "greedy_search",
    ):
        if src_mask is None:
            src_mask = self.create_masks(src, pad_idx=self.pad_idx)[0]
        generation_mode = generation_mode.lower()
        if generation_mode == "greedy_search":
            return self._greedy_search(src, src_mask, max_gen_len, keep_attentions)
        else:
            return None
                

    def _greedy_search(self, src: Tensor, src_mask: Tensor, max_gen_len: int, keep_attention: bool):
        memory = self.transformer.encode(src, src_mask)
        batch_size = src.shape[0]
        device = src.device

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        decoder_inputs = torch.LongTensor(batch_size, 1).fill_(self.bos_idx).to(device)

        eos_idx_tensor = torch.tensor([self.eos_idx]).to(device)

        finished = False

        while not finished:
            tgt_mask = self.generate_subsequent_mask(decoder_inputs.size(1), device)

            logits = self.lm_head(
                self.transformer.decode(
                    decoder_inputs, 
                    memory, 
                    tgt_mask=tgt_mask, 
                    memory_mask=src_mask, 
                    keep_attentions=keep_attention
                )
            )

            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)

            next_tokens = next_tokens * unfinished_sequences + self.pad_idx * (1 - unfinished_sequences)

            decoder_inputs = torch.cat(decoder_inputs, next_tokens[:, None], dim=-1)

            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_idx_tensor.shape[0], 1)
                .ne(eos_idx_tensor.unsqueeze(1))
                .prod(dim=0)
            )

            if unfinished_sequences.max() == 0:
                finished = True

            if decoder_inputs.shape[-1] >= max_gen_len:
                finished = True

        return decoder_inputs