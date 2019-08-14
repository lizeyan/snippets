import torch.nn as nn
import typing


class MultiLayerPerceptron(nn.Sequential):
    def __init__(self, features: typing.Sequence[int],
                 activation_class=nn.LeakyReLU,
                 bias=True, last_activation=False):
        super().__init__()
        layers = []
        for in_features, out_features in zip(features[0:-1], features[1:]):
            layers.append(
                nn.Linear(in_features, out_features, bias)
            )
            layers.append(
                activation_class()
            )
        if not last_activation:
            layers.pop()
        super().__init__(*layers)


MLP = MultiLayerPerceptron
