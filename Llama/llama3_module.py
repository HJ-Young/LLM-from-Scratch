from torch import torch
from torch import nn
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import json

# Llama3 = embedding + WPE + 32 * Block


def RMS_Norm(x, norm_w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_w


def SwiGLU(x, beta=1.0):
    return x / (1 + torch.exp(-beta * x))


# x.shape = (sequence_len, dim)
def RoPE(x, theta):
    x_len, dim = x.shape[0], x.shape[1]
    x = x.reshape(x_len, dim // 2, 2)
    freqs = torch.arange(dim // 2, dtype=x.dtype, device=x.device) / (dim // 2)
    freqs = 1.0 / (theta**freqs)
    freqs = torch.outer(torch.arange(x_len, dtype=x.dtype, device=x.device), freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    pos_freq = torch.view_as_complex(x) * freqs
    return pos_freq.view_as_real(pos_freq).reshape(x_len, dim)


def GQA(x, w_q, w_k, w_v, w_o, n_heads, n_kv_heads, theta):
    w_q = w_q.reshape(n_heads, w_q.shape[0] // n_heads, w_q.shape[1])
    w_k = w_k.reshape(n_kv_heads, w_k.shape[0] // n_kv_heads, w_k.shape[1])
    w_v = w_v.reshape(n_kv_heads, w_v.shape[0] // n_kv_heads, w_v.shape[1])

    mask = (1 - torch.tri(x.shape[0])) * float("-inf")

    qkv = []

    for head in range(n_heads):
        w_q_head = RoPE(torch.matmul(x, w_q[head].T), theta)
        w_k_head = RoPE(torch.matmul(x, w_k[head // 4]), theta)  # (128, 4096)
        w_v_head = torch.matmul(x, w_v[head // 4])
        qk = nn.functional.softmax(torch.matmul(w_q_head, w_k_head.T) / torch.sqrt(w_q_head.shape[1]) + mask)
        qkv.append(torch.matmul(qk, w_v_head))
    return torch.matmul(torch.cat(qkv, dim=-1), w_o.T)


def FFN(x, w_1, w_2, w_3):
    x = SwiGLU(torch.matmul(x, w_1.T)) * torch.matmul(x, w_3.T)
    return torch.matmul(x, w_2.T)


def get_tokenizer(path):
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

    bpe_ranks = load_tiktoken_bpe(path)
    tokenizer = tiktoken.Encoding(
        name=Path(path).name,
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks=bpe_ranks,
        special_tokens={token: len(bpe_ranks) + i for i, token in enumerate(bpe_ranks)},
    )
    return tokenizer


def generate(prompt, model_path="Meta-Llama-3-8B/"):
    model = torch.load(model_path + "consolidated.00.pth")
    tokenizer = get_tokenizer("tokenizer.model")
    tokens = [12800] + tokenizer.encode(prompt)

    with open(model_path + "parmas.json", "r") as f:
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

    embedding_layer = nn.Embedding(vocab_size, dim)
    embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
    X = embedding_layer(tokens).to(torch.bfloat16)

    for layer in range(n_layers):
        attn_norm = model[f"layers.{layer}.attention_norm.weight"]
        ffn_norm = model[f"layers.{layer}.ffn_norm.weight"]
        w_k = model[f"layers.{layer}.attention.wk.weight"]
        w_q = model[f"layers.{layer}.attention.wq.weight"]
        w_v = model[f"layers.{layer}.attention.wv.weight"]
        w_o = model[f"layers.{layer}.attention.wo.weight"]
        ffn_w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
        ffn_w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
        ffn_w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
        attn_input = RMS_Norm(X, attn_norm)
        attn_output = GQA(attn_input, w_q, w_k, w_v, w_o, n_heads, n_kv_heads, rope_theta)

        X = X + attn_input
        ffn_input = RMS_Norm(X, ffn_norm)
        ffn_output = FFN(ffn_input, ffn_w1, ffn_w2, ffn_w3)
        X = ffn_output + X

    norm_w = model["norm.weight"]
    output = RMS_Norm(X, norm_w)
    output_w = model["output_weight"]
    logits = torch.matmal(output[-1], output_w)
    next_token = torch.argmax(logits)
    return tokenizer.decode(next_token)
