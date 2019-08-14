import unittest

from snippets.data import ParallelDataset
import torch


class TestParallelDataset(unittest.TestCase):
    def test_parallel_dataset(self):
        dataset_a = torch.randn(100, 100)
        dataset_b = torch.randn(100, 100)
        dataset_c = torch.randn(100, 100)
        dataset = ParallelDataset(dataset_a, dataset_b, dataset_c)
        self.assertEqual(len(dataset), 100)
        for item in (1, 11, 13):
            self.assertIsInstance(dataset[item], tuple)
            self.assertTrue(torch.all(dataset[item][0] == dataset_a[item]).item())
            self.assertTrue(torch.all(dataset[item][1] == dataset_b[item]).item())
            self.assertTrue(torch.all(dataset[item][2] == dataset_c[item]).item())

