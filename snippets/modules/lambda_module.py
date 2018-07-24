import torch.nn as nn
import typing


class Lambda(nn.ModuleDict):
    def __init__(self, func: typing.Callable):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)
