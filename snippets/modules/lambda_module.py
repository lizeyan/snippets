import typing

import torch.nn as nn


class Lambda(nn.Module):
    """
    Not that only function without any tensor parameters are suitable,
    """
    def __init__(self, func: typing.Callable):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def cuda(self, device=None):
        if isinstance(self.func, nn.Module):
            # noinspection PyUnresolvedReferences
            self.func.cuda(device)
        return self

    def cpu(self):
        if isinstance(self.func, nn.Module):
            # noinspection PyUnresolvedReferences
            self.func.cpu()
        return self
