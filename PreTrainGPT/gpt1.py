import torch
import math
from torch import nn


class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, X):
        return 0.5 * X * (1 + torch.tanh((torch.sqrt(2) / math.pi) * (x + 0.044714 * x**3)))


wte = nn.Embedding(vocab_size, num_hiddens)
wpe = nn.Embedding(max_len, num_hiddens)
