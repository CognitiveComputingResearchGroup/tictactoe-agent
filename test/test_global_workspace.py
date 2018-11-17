from unittest import TestCase

from common import GlobalWorkspace


class TestGlobalGlobalWorkspace(TestCase):
    def test_init(self):
        try:
            w = GlobalWorkspace()
        except Exception as e:
            self.fail(e)

    def test_call(self):
        try:
            w = GlobalWorkspace()

            # invoke __call__ several times (testing for exceptions)
            for v in range(100):
                w([map(str, range(10))])

        except Exception as e:
            self.fail(e)

    def test_next(self):
        try:
            w = GlobalWorkspace()

            # next should return the last coalition that was added to the GlobalWorkspace (using __call__)
            for expected in range(10):
                w(expected)  # Update GlobalWorkspace
                actual = next(w)
                self.assertEqual(expected, actual)
        except Exception as e:
            self.fail(e)
