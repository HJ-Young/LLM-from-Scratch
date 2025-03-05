import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def linear(x, w, b):
    return x @ w + b


def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)

    qkv = np.split(x, 3, axis=-1)
    qkv = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))

    casual_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    output_head = [attention(q, k, v, casual_mask) for q, k, v in zip(*qkv)]

    output = np.hstack(output_head)
    return linear(output, **c_proj)


def ffn(x, c_fc, c_proj):
    x = linear(x, **c_fc)
    return linear(gelu(x), **c_proj)


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def layer_norm(x, g, b, eps=1e-5):
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x_var = np.var(x, axis=-1, keepdims=True) + eps
    x = (x - x_mean) / np.sqrt(x_var)

    return x * g + b


def gpt2(input, wte, wpe, blocks, ln_f, n_head):
    x = wte[input] + wpe[np.arange(len(input))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)

    x = layer_norm(x, **ln_f)
    return x @ wte.T


def generate(input, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(input, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])
        input.append(int(next_id))

    return input[len(input) - n_tokens_to_generate :]


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    input_idx = encoder.encode(prompt)
    assert len(input_idx) + n_tokens_to_generate < hparams["n_ctx"], "Prompt is toooooooo long"

    output_idx = generate(input_idx, params, hparams["n_head"], n_tokens_to_generate)

    output = encoder.decode(output_idx)
    return output


if __name__ == "__main__":
    import fire

    fire.Fire(main)
