import torch
import math
from torch import nn


class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, X):
        return 0.5 * X * (1 + torch.tanh((torch.sqrt(2) / math.pi) * (X + 0.044714 * X**3)))
