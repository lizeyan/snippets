from snippets.utilities import *
import unittest


class TestSnippets(unittest.TestCase):
    def test_same_length(self):
        a = [1, 2, 3]
        b = []
        c = [1, 2, 3]
        self.assertEqual(in_same_length(), True)
        self.assertEqual(in_same_length(a, b, c), False)
        self.assertEqual(in_same_length(a, c), True)

    def test_split(self):
        arr = list(range(10))
        a, b, c, d = split(arr, [0.1, 0.2, 0.3, 0.4])
        self.assertListEqual(a, [0])
        self.assertListEqual(b, [1, 2])
        self.assertListEqual(c, [3, 4, 5])
        self.assertListEqual(d, [6, 7, 8, 9])
        a, b, = split(arr, [0.4, 0.2])
        self.assertListEqual(a, [0, 1, 2, 3])
        self.assertListEqual(b, [4, 5])
