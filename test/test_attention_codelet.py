from unittest import TestCase

from common import AttentionCodelet, Workspace


class TestAttentionCodelet(TestCase):
    def test_init(self):
        try:
            _ = AttentionCodelet()
            _ = AttentionCodelet(is_match=lambda x: False)
            _ = AttentionCodelet(match_content=2)
        except Exception as e:
            self.fail(e)

    def test_call(self):

        # With default is_match
        try:
            c = AttentionCodelet()
            w = Workspace()

            # Add content to the Workspace
            for i in range(10):
                w(i)

            # invoke __call__ several times (testing for exceptions)
            for v in range(100):
                c(w)

        except Exception as e:
            self.fail(e)

        # With custom is_match
        try:
            c = AttentionCodelet(is_match=lambda x: x < 3)
            w = Workspace()

            # Add content to the Workspace
            for i in range(10):
                w(i)

            # invoke __call__ several times (testing for exceptions)
            for v in range(100):
                c(w)

        except Exception as e:
            self.fail(e)

    def test_next(self):
        try:
            w = Workspace()

            # Add content to the Workspace
            for i in range(10):
                w(i)

            for v in range(10):
                c = AttentionCodelet(match_content=5)
                c(w)
                coalition= next(c)
                self.assertEqual(5, coalition[0])
        except Exception as e:
            self.fail(e)
