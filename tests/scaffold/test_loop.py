import unittest
from snippets.scaffold import TrainLoop, TestLoop


class TestTrainLoop(unittest.TestCase):
    def test_counts(self):
        epoch_counts = 0
        step_counts = 0
        with TrainLoop(max_steps=10, max_epochs=3).with_context() as train_loop:
            for _ in train_loop.iter_epochs():
                epoch_counts += 1
                for _ in train_loop.iter_steps([1, 2, 3, 4, 5, 6]):
                    step_counts += 1
        self.assertEqual(epoch_counts, 2)
        self.assertEqual(step_counts, 10)

