import unittest
import os
from snippets.scaffold import get_gpu_metrics, sort_gpu_index


class TestChooseDevices(unittest.TestCase):
    def setUp(self):
        self.path = os.environ["PATH"]

    def tearDown(self):
        os.environ["PATH"] = self.path

    def test(self):
        self.assertEqual(len(sort_gpu_index()), len(get_gpu_metrics()))
