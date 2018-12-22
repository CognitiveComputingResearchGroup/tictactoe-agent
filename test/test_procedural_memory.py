from collections import Counter
from unittest import TestCase

from common import ProceduralMemory, Scheme, exact_match_context_by_move, board_position_after_move
from env.environment import Move, Board


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

    def test_board_position_after_move(self):

        BLANK_BOARD = Board.blank_board()

        # Scenario 1: Add single mark to blank board
        for pos in range(9):
            move = Move(position=pos, mark='X')
            board = BLANK_BOARD

            expected_board = board.copy()
            expected_board[move.position] = move.mark

            self.assertEqual(board_position_after_move(board, move), expected_board)

            # Verify no side effects
            self.assertEqual(board, BLANK_BOARD)

        # Scenario 2: Add single mark to non-blank board / no conflict in mark
        CENTER_MARK_BOARD = BLANK_BOARD.copy()
        CENTER_MARK_BOARD[4] = 'X'

        for pos in range(9):
            move = Move(position=pos, mark='X')
            board = CENTER_MARK_BOARD

            # Note that in the case where we are adding a mark in the center square, the expected
            # board will be equal to the original board
            expected_board = board.copy()
            expected_board[move.position] = move.mark

            self.assertEqual(board_position_after_move(board, move), expected_board)

            # Verify no side effects
            self.assertEqual(board, CENTER_MARK_BOARD)
