import torch.nn as nn
import typing


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
            self.func.cuda(device)
        return self

    def cpu(self):
        if isinstance(self.func, nn.Module):
            self.func.cpu()
        return self
