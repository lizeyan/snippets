import torch.nn as nn
import typing


class MultiLayerPerceptron(nn.Module):
    _activation_dict = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "leakyrelu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "prelu": nn.PReLU,
    }

    def __init__(self, input_size: typing.SupportsInt, net_sizes: typing.List[typing.SupportsInt],
                 activation: typing.Union[typing.Callable[..., typing.Callable], typing.AnyStr]="leakyrelu",
                 bias=True):
        super().__init__()
        self._net_sizes = net_sizes
        if isinstance(activation, str):
            try:
                self._activation_cls = self._activation_dict[activation.lower()]
            except KeyError:
                raise ValueError(f"Unknown activation name {activation}")
        else:
            self._activation_cls = activation
        self._bias = bias

        self._layers = []
        for in_features, out_features in zip([input_size] + net_sizes[:-1], net_sizes[:]):
            self._layers.append(nn.Linear(in_features=int(in_features),
                                          out_features=int(out_features),
                                          bias=self._bias))
            self._layers.append(self._activation_cls())
        self._layers = nn.ModuleList(self._layers)

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


MLP = MultiLayerPerceptron

