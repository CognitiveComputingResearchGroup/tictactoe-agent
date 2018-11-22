from unittest import TestCase

from agent import running, run


class TestAgent(TestCase):
    def test_run(self):
        try:
            # Verify termination condition on bounded run
            count = run(1)
            self.assertEqual(count, 1)

        except Exception as e:
            self.fail(e)

    def test_running(self):
        try:

            self.assertTrue(running(step=0, last=1))
            self.assertFalse(running(step=1, last=1))
            self.assertFalse(running(step=2, last=1))
            self.assertTrue(running(step=100, last=None))  # Run forever if last not specified
            self.assertTrue(running(step=100))  # Should run forever (tests default value of last == None)

        except Exception as e:
            self.fail(e)
