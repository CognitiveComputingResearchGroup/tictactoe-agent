from unittest import TestCase

from common import Workspace


class TestWorkspace(TestCase):
    def test_init(self):
        try:
            w = Workspace()
        except Exception as e:
            self.fail(e)

    def test_call(self):
        try:
            w = Workspace()

            # invoke __call__ several times (testing for exceptions)
            for v in range(100):
                w('content')
                
        except Exception as e:
            self.fail(e)

    def test_next(self):
        try:
            w = Workspace()

            # next should return the last value that was updated into Workspace (using __call__)
            for expected in range(10):
                w(expected)  # Update Workspace
                actual = next(w)
                self.assertEqual(expected, actual)
        except Exception as e:
            self.fail(e)

    def test_iter(self):
        try:
            w = Workspace()

            # adding expected values to workspace
            values = range(10)
            for v in values:
                w(v)

            # should be able to iterate over expected values in order of insertion
            for expected, actual in zip(values, w):
                self.assertEqual(expected, actual)
        except Exception as e:
            self.fail(e)
