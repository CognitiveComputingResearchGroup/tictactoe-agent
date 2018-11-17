from unittest import TestCase

from common import CueingProcess, Module


class TestCueingProcess(TestCase):
    def test_init(self):
        try:
            c = CueingProcess()
        except Exception as e:
            self.fail(e)

    def test_call(self):
        try:
            c = CueingProcess()
            m = DummyModule()

            # invoke __call__ several times (testing for exceptions)
            for v in range(100):
                c(v, m)

        except Exception as e:
            self.fail(e)

    def test_next(self):
        try:
            c = CueingProcess()
            m = DummyModule()

            for expected in range(10):
                c(expected, m)  # Perform cue operation
                actual = next(c)
                self.assertListEqual([expected], actual)
        except Exception as e:
            self.fail(e)


class DummyModule(Module):
    def __init__(self):
        super().__init__()
        self.content = []

    def __call__(self, content):
        self.content.append(content)

    def __next__(self):
        return self.content[-1]
