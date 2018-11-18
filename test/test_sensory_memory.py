from unittest import TestCase

from common import SensoryMemory


class TestSensoryMemory(TestCase):
    def test_init(self):
        try:
            sm = SensoryMemory()
        except Exception as e:
            self.fail(e)

    def test_call(self):
        try:
            sm = SensoryMemory()

            # invoke __call__ several times (testing for exceptions)
            for v in range(100):
                sm([map(str, range(10))])

        except Exception as e:
            self.fail(e)

    def test_next(self):
        try:
            sm = SensoryMemory()

            # next should return the last sensory content that was added to the SensoryMemory (using __call__)
            for expected in range(10):
                sm(expected)  # Update SensoryMemory
                actual = next(sm)
                self.assertEqual(expected, actual)
        except Exception as e:
            self.fail(e)
