from snippets.utilities import assert_positive_integer
import unittest


class TestAssertion(unittest.TestCase):
    def test_positive_integer(self):
        def __try_failed(key, value):
            ok = False
            try:
                assert_positive_integer(key=value)
            except ValueError:
                ok = True
            finally:
                self.assertEqual(ok, True)

        __try_failed("a", 0.1)
        __try_failed("b", "132-12356")
        assert_positive_integer(a=1)
        assert_positive_integer(a="1")
        assert_positive_integer(a=23.0)




