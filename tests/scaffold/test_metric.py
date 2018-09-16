from snippets.scaffold import Metric
import unittest


class TestMetric(unittest.TestCase):
    def test(self):
        metric = Metric("test")
        self.assertEqual(metric.name, "test")
        metric.collect(1, 1)
        metric.collect(2, 2)
        metric.collect(4, 4)
        metric.collect(3, 3)
        metric.collect(5, [1, 2, 3, 4, 5])
        metric.collect(3, 3)
        metric.collect(3, 3)
        self.assertEqual(metric.data, {1: 1, 2: 2, 4: 4, 3: [3, 3, 3], 5: [1, 2, 3, 4, 5]})
        self.assertEqual(metric["all"], list({1: 1, 2: 2, 4: 4, 3: [3, 3, 3], 5: [1, 2, 3, 4, 5]}.values()))
        self.assertEqual(metric["last"], [1, 2, 3, 4, 5])
        self.assertEqual(metric[3], [3, 3, 3])
        self.assertEqual(metric[[3, 4, 5]], [[3, 3, 3], 4, [1, 2, 3, 4, 5]])
        self.assertEqual(metric.format(1), "test:1.000")
        self.assertEqual(metric.format(3), "test:3.000(±0.000)")
        self.assertEqual(metric.format([2, 4]), "test:3.000(±1.000)")
        self.assertEqual(metric.format([2, 4, 6, 7, 8]), "test:3.000(±1.000)")
        ok = True
        try:
            with Metric.raise_key_error():
                _ = metric[128]
        except KeyError:
            ok = False
        finally:
            Metric._IGNORE_KEY_ERROR = True
            if ok:
                raise ValueError("Key Error is ignored")
        ok = True
        try:
            with Metric.raise_key_error():
                _ = metric[0, 4, 128]
        except KeyError:
            ok = False
        finally:
            Metric._IGNORE_KEY_ERROR = True
            if ok:
                raise ValueError("Key Error is ignored")


