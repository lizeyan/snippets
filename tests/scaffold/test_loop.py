import torch
import unittest
from snippets.scaffold import TrainLoop, TestLoop
import time
import re


class TestTrainLoop(unittest.TestCase):
    def setUp(self):
        self.line = ""

    def print(self, s: str):
        self.line = s

    def test_counts(self):
        epoch_counts = 0
        step_counts = 0
        with TrainLoop(max_steps=10, max_epochs=3, print_fn=None).with_context() as train_loop:
            for epoch in train_loop.iter_epochs():
                epoch_counts += 1
                for _ in train_loop.iter_steps([torch.Tensor(5) for i in range(6)]):
                    step_counts += 1
                    train_loop.submit_metric("value1", 1)
                    if epoch % 2 == 0:
                        train_loop.submit_metric("value2", [1, 2])
                    train_loop.submit_data("value1", 1)
                    if epoch % 2 == 0:
                        train_loop.submit_data("value2", [1, 2])
                train_loop.submit_metric("value3", [[1]])
                train_loop.submit_data("value3", [[1]])
            self.assertEqual(epoch_counts, 2)
            self.assertEqual(step_counts, 10)
            self.assertEqual(train_loop.get_metric("value1")[0], [1, 2])
            self.assertEqual(train_loop.get_metric("value1")[1], [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1]])
            self.assertEqual(train_loop.get_metric("value2")[0], [2])
            self.assertEqual(train_loop.get_metric("value2")[1], [[[1, 2], [1, 2], [1, 2], [1, 2]]])
            self.assertEqual(train_loop.get_metric("value3")[0], [1, 2])
            self.assertEqual(train_loop.get_metric("value3")[1], [[[[1]]], [[[1]]]])
            self.assertEqual(train_loop.get_data("value1")[0], [1, 2])
            self.assertEqual(train_loop.get_data("value1")[1], [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1]])
            self.assertEqual(train_loop.get_data("value2")[0], [2])
            self.assertEqual(train_loop.get_data("value2")[1], [[[1, 2], [1, 2], [1, 2], [1, 2]]])
            self.assertEqual(train_loop.get_data("value3")[0], [1, 2])
            self.assertEqual(train_loop.get_data("value3")[1], [[[[1]]], [[[1]]]])
            self.assertEqual(train_loop.get_metric(1)["value1"], [1, 1, 1, 1, 1, 1])
            self.assertEqual(train_loop.get_metric(1)["value3"], [[[1]]])
            self.assertEqual(train_loop.get_data(1)["value1"], [1, 1, 1, 1, 1, 1])
            self.assertEqual(train_loop.get_data(1)["value3"], [[[1]]])

        step_counts = 0
        with TestLoop(print_fn=None).with_context() as test_loop:
            for _ in test_loop.iter_steps([torch.Tensor(5) for i in range(6)]):
                step_counts += 1
                test_loop.submit_metric("value1", 1)
                test_loop.submit_data("value1", 1)
            self.assertEqual(step_counts, 6)
            self.assertEqual(test_loop.get_metric("value1"), [1, 1, 1, 1, 1, 1])
            self.assertEqual(test_loop.get_data("value1"), [1, 1, 1, 1, 1, 1])

    def test_eta(self):
        with TrainLoop(max_epochs=3, print_fn=self.print).with_context() as train_loop:
            for epoch in train_loop.iter_epochs():
                time.sleep(0.01)
                if epoch > 1:
                    eta = float(re.search("ETA:(?P<eta>[\d.]+)s", self.line).group("eta"))
                    self.assertAlmostEqual(eta, 0.03 - (epoch - 1) * 0.01, places=2)

        with TrainLoop(max_epochs=3, max_steps=10, print_fn=self.print).with_context() as train_loop:
            for epoch in train_loop.iter_epochs():
                for step, _ in train_loop.iter_steps([torch.Tensor() for _ in range(6)]):
                    time.sleep(0.01)
                if epoch > 1:
                    eta = float(re.search("ETA:(?P<eta>[\d.]+)s", self.line).group("eta"))
                    self.assertAlmostEqual(eta, 0.1 - (epoch - 1) * 6 * 0.01, places=2)

    def test_print(self):
        with TrainLoop(max_epochs=14, print_fn=self.print, disp_epoch_freq=5).with_context() as train_loop:
            for epoch in train_loop.iter_epochs():
                if epoch == 6 or epoch == 11:
                    self.assertTrue(f"epoch:{epoch - 1}" in self.line.lower())
        self.assertTrue(f"epoch:{14}" in self.line.lower())

        with TrainLoop(max_steps=23, print_fn=self.print).with_context() as train_loop:
            for epoch in train_loop.iter_epochs():
                for step, _ in train_loop.iter_steps([torch.Tensor(1) for _ in range(6)]):
                    pass
                if epoch > 1:
                    self.assertEqual(re.search("epoch:[\d+]/[\d+]]", self.line), None)


