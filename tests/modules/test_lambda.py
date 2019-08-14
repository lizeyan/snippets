import unittest

import torch
import torch.distributions as dist
import torch.nn as nn

from snippets.modules import Lambda


class TestLambda(unittest.TestCase):
    def test_lambda_unary(self):
        x = torch.randn(3, 5, 7, 11)
        module = Lambda(lambda _x: _x ** 2)
        self.assertTrue(torch.all(x ** 2 == module(x)).item())

    def test_lambda_binary(self):
        x = torch.randn(3, 5, 7, 11)
        y = torch.randn(3, 5, 7, 11)
        module = Lambda(lambda _x, _y: _x * _y)
        self.assertTrue(torch.all(x * y == module(x, y)).item())


