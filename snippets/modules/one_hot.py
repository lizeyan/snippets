import torch
import torch.nn as nn


class OneHot(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, size: int=None):
        return one_hot(x, size)


def one_hot(x, size):
    y = torch.zeros(x.size() + (size, ), dtype=x.dtype, device=x.device)
    y.scatter_(-1, x.type(torch.long).unsqueeze(-1), 1)
    return y.type_as(x)
