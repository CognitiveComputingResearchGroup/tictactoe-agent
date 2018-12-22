from collections import Counter
from unittest import TestCase

from common import ProceduralMemory, Scheme, exact_match_context_by_move
from env.environment import Move


class TestProceduralMemory(TestCase):
    def test_init(self):
        try:
            pm = ProceduralMemory()
            pm = ProceduralMemory(initial_schemes=[Scheme() for i in range(10)])
        except Exception as e:
            self.fail(e)

    def test_call(self):
        try:
            pm = ProceduralMemory()

            # invoke __call__ several times (testing for exceptions)
            for v in range(100):
                pm([map(str, range(10))])

        except Exception as e:
            self.fail(e)

    def test_next(self):

        try:
            # Test Case 1
            #############
            pm = ProceduralMemory()

            # No initial _schemes, so should return None
            self.assertEqual(next(pm), [])

            # Test Case 2
            #############
            scheme = Scheme()
            pm = ProceduralMemory(initial_schemes=[scheme])

            # Only a single scheme, so all calls to next should return that scheme
            for i in range(10):
                self.assertEqual(next(pm), scheme)

            # Test Case 2
            #############

            scheme_1 = Scheme(action='X')
            scheme_2 = Scheme(action='Y')

            pm = ProceduralMemory(initial_schemes=[scheme_1, scheme_2])

            # Two _schemes -- each should be randomly selected (uniform distribution)
            selected = Counter([next(pm) for i in range(10000)])

            # Checking for approximately equal counts
            self.assertAlmostEqual(selected[scheme_1], selected[scheme_2], delta=2000)

        except Exception as e:
            self.fail(e)

    def test_exact_match_context_by_move(self):
        scheme_1 = Scheme(context=Move(1, 'X'))
        scheme_2 = Scheme(context=Move(2, 'X'))

        # Test with content = None
        content = None
        self.assertFalse(exact_match_context_by_move(content, scheme_1))
        self.assertFalse(exact_match_context_by_move(content, scheme_2))

        # Test with type(content) == Move
        content = Move(1, 'X')
        self.assertTrue(exact_match_context_by_move(content, scheme_1))
        self.assertFalse(exact_match_context_by_move(content, scheme_2))

        # Test with non-iterable content where type(content) == Move
        content = 123
        self.assertFalse(exact_match_context_by_move(content, scheme_1))
        self.assertFalse(exact_match_context_by_move(content, scheme_2))

        # Test with iterable content containing content(type) == Move
        content = [Move(1, 'X'), 'Other Stuff', ['More', 'Stuff']]
        self.assertTrue(exact_match_context_by_move(content, scheme_1))
        self.assertFalse(exact_match_context_by_move(content, scheme_2))
