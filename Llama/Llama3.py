from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
from torch import nn
import json
import matplotlib.pyplot as plt

tokenizer_path = "Meta-Llama-3-8B/tokenizer.model"
special_tokens = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|reserved_special_token_2|>",
    "<|reserved_special_token_3|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_4|>",
    "<|eot_id|>",  # end of turn
] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)

model = torch.load("Meta-Llama-3-8B/consolidated.00.pth")
with open("Meta-Llama-3-8B/params.json", "r") as f:
    config = json.load(f)

dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])

prompt = "the answer to the ultimate question of life, the universe, and everything is "
tokens = [128000] + tokenizer.encode(prompt)

embedding_layer = nn.Embedding(vocab_size, dim)
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)


def rms_norm(x, norm_w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm_eps) * norm_w


token_embeddings = rms_norm(token_embeddings_unnormalized, model["layers.0.attention_norm.weight"])


# RoPE + MHA

## q layer0
q_layer0 = model["layers.0.attention.wq.weight"]

w_q = q_layer0.reshape(n_heads, q_layer0.shape[0] // n_heads, dim)  # (32, 128, 4096)
w_q_head0 = w_q[0]

q_head0 = torch.matmul(token_embeddings, w_q_head0.T)  # (17 * 128)
q_head0_pairs = q_head0.float().view(q_head0.shape[0], -1, 2)

half_dim = q_head0_pairs.shape[1]
zero_to_one_split_into_parts = torch.tensor(range(half_dim)) / half_dim
freqs = 1.0 / (rope_theta**zero_to_one_split_into_parts)

position_freqs = torch.outer(torch.arange(q_head0_pairs.shape[0]), freqs)
freq_cis = torch.polar(torch.ones_like(position_freqs), position_freqs)

q_head0_pairs_as_complex = torch.view_as_complex(q_head0_pairs)
q_head0_pairs_as_complex_rotated = q_head0_pairs_as_complex * freq_cis

q_head0_pairs_rotated = torch.view_as_real(q_head0_pairs_as_complex_rotated)
q_head0_pairs_rotated = q_head0_pairs_rotated.view(q_head0.shape)

## k layer0
k_layer0 = model["layers.0.attention.wk.weight"]

k_layer0 = k_layer0.reshape(n_kv_heads, k_layer0.shape[0] // n_kv_heads, dim)  # (8, 128, 4096)
w_k_head0 = k_layer0[0]

k_head0 = torch.matmul(token_embeddings, w_k_head0.T)
k_head0_pairs = k_head0.reshape(k_head0.shape[0], -1, 2)

k_head0_pairs_as_complex = torch.view_as_complex(k_head0_pairs)
k_head0_pairs_as_complex_rotated = k_head0_pairs_as_complex * freq_cis

k_head0_pairs_rotated = torch.view_as_real(k_head0_pairs_as_complex_rotated).view(w_k_head0.shape)

qk = torch.matmul(q_head0_pairs_rotated, k_head0_pairs_rotated.T) / (q_head0_pairs_rotated.shape[-1] ** 0.5)

mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
mask = torch.triu(mask, diagonal=1)

qk_masked = qk + mask
qk_masked_softmax = torch.nn.functional.softmax(qk_masked, dim=1).to(torch.bfloat16)

v_layer0 = model["layers.0.attention.wv.weight"]
v_layer0.reshape(n_kv_heads, v_layer0.shape[0] // n_kv_heads, dim)
v_layer0_head0 = v_layer0[0]
w_v = torch.matmul(token_embeddings, v_layer0_head0.T)

qkv_attn = torch.matmul(qk, w_v)

qkv = []

for head in range(n_heads):
    w_q = q_layer0[head]
    w_k = k_layer0[head // 4]
    w_v = v_layer0[head // 4]

    w_q = torch.matmul(token_embeddings, w_q).reshape(w_q.shape[0], -1, 2)
    w_k = torch.matmul(token_embeddings, w_k).reshape(w_k.shape[0], -1, 2)
    w_v = torch.matmul(token_embeddings, w_v)

    w_q_rotated = torch.view_as_complex(w_q) * freq_cis
    w_k_rotated = torch.view_as_complex(w_k) * freq_cis

    w_q = torch.view_as_real(w_q_rotated).reshape(w_q.shape[0], -1)
    w_k = torch.view_as_real(w_k_rotated).reshape(w_k.shape[0], -1)

    mask = torch.full((w_q.shape[0], w_q.shape[0]), float("-inf"), device=tokens.device)
    mask = torch.triu(mask, diagonal=1)

    qk = torch.nn.functional.softmax(torch.matmul(w_q, w_k.T) / (w_q.shape[-1] ** 0.5) + mask)

    qkv.append(torch.matmul(qk, w_v))  # (17 * 128)

qkv_attn = torch.cat(qkv, dim=-1)

w_o = model["layers.0.attention.wo.weight"]
attn = torch.matmul(qkv_attn, w_o.T)

# FFN
ffn_input = attn + token_embeddings_unnormalized
ffn_input = rms_norm(ffn_input, model["layers.0.ffn_norm.weight"])


def silu(x, delta=1):
    return x / (1 + torch.exp(-delta * x))


w1 = model["layers.0.feed_forward.w1.weight"]
w2 = model["layers.0.feed_forward.w2.weight"]
w3 = model["layers.0.feed_forward.w3.weight"]

ffn_output = torch.matmul(torch.functional.F.silu(torch.matmul(ffn_input, w1.T)) * torch.matmul(ffn_input, w3.T), w2.T)
