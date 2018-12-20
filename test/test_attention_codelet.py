from unittest import TestCase

from agent import codelet_happy_match

class TestAttentionCodelet(TestCase):
    def test_happy_match(self):
        # this case is invalid because the attn codelet will treat the string 
        # as a list of strings and test on "h" "e" "l" ...
        # codelet_happy_match("happy")
        # self.assertEqual(next(codelet_happy_match), ["happy"])

        codelet_happy_match([8])
        self.assertEqual(next(codelet_happy_match), [])

        codelet_happy_match(["x"])
        self.assertEqual(next(codelet_happy_match), [])

        codelet_happy_match([["x", "happy"], 897])
        self.assertEqual(next(codelet_happy_match), [["x", "happy"]])
