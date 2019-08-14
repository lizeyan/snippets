import typing

import torch.nn as nn


class Lambda(nn.Module):
    """
    Wrapper a callable without any parameters to a Module
    """
    def __init__(self, func: typing.Callable):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)
