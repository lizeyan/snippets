import unittest
from snippets.modules import one_hot, OneHot
import numpy as np
import torch


def form(x, dtype=np.int64):
    return torch.from_numpy(np.asarray(x, dtype=dtype))


class TestOneHot(unittest.TestCase):
    def assertTensorEqual(self, a, b):
        self.assertTrue(a.dtype == b.dtype, f"a({a.dtype}) and b({b.dtype}) have different dtype")
        self.assertTrue((a - b).sum() == torch.zeros(1, dtype=a.dtype), msg=f"a and b is not equal")

    def test(self):
        a = form([1, 3])
        b = form([[0, 1], [2, 0]])
        c = form([[[0.]], [[1.]]], dtype=np.float32)

        self.assertTensorEqual(one_hot(a, size=5),
                               form([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]]))
        self.assertTensorEqual(OneHot()(a, size=5),
                               form([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]]))
        self.assertTensorEqual(one_hot(a, size=4),
                               form([[0, 1, 0, 0], [0, 0, 0, 1]]))

        self.assertTensorEqual(one_hot(b, size=3),
                               form([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]]))
        self.assertTensorEqual(one_hot(c, size=3),
                               form([[[[1, 0, 0]]], [[[0, 1, 0]]]], dtype=np.float32))
