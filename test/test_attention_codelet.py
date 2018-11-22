from unittest import TestCase
from hypothesis import given
from hypothesis import strategies as st

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

    @given(st.integers(), st.lists(st.integers()))
    def test_next_with_integer_lists(self, a, b):
        try:
            w = Workspace()

            # Add content to the Workspace
            for i in b:
                w(i)

            c = AttentionCodelet(match_content=a)
            c(w)
            coalition= next(c)
            if a in b:
                self.assertIn(a, coalition)
            else:
                self.assertNotIn(a, coalition)
        except Exception as e:
            self.fail(e)
