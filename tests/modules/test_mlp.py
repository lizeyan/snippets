import unittest

import torch
import torch.distributions as dist
import torch.nn as nn

from snippets.modules import MLP


class TestMLP(unittest.TestCase):
    def test_mlp_shape(self):
        features = (37, 31, 23, 11, 5, 2)
        x = torch.randn(11, 5, 37)
        mlp = MLP(features, activation_class=nn.LeakyReLU)
        y = mlp(x)
        self.assertEqual(y.size(), (11, 5, features[-1]))

