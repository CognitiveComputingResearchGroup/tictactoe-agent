from random import random
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

    def test_candidate_behaviors(self):

        try:
            # Test Case 1
            #############
            pm = ProceduralMemory()

            # No initial _schemes, so should return None
            self.assertEqual(pm.candidate_behaviors, [])

            # Test Case 2
            #############
            scheme = Scheme(current_activation=1.0)
            pm = ProceduralMemory(initial_schemes=[scheme])

            # Only a single scheme
            self.assertListEqual(pm.candidate_behaviors, [scheme])

            # Test Case 2
            #############

            scheme_1 = Scheme(action='X', current_activation=1.0)
            scheme_2 = Scheme(action='Y', current_activation=1.0)

            pm = ProceduralMemory(initial_schemes=[scheme_1, scheme_2])

            # Two _schemes
            self.assertListEqual(pm.candidate_behaviors, [scheme_1, scheme_2])

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

    def test_receive_broadcast(self):
        initial_schemes = [Scheme(current_activation=random()) for i in range(1000)]
        pm = ProceduralMemory(initial_schemes=initial_schemes)

        # Scenario 1: broadcast = None
        pm.receive_broadcast(None)

        self.assertListEqual(initial_schemes, pm._schemes)

        # Scenario 2: Multiple schemes, Single activated candidate behavior
        initial_schemes = [Scheme(context=v) for v in [1.0, 1000.0, 2.0]]
        pm = ProceduralMemory(initial_schemes, context_match=lambda s, b: 1 - abs(b - s.context),
                              activation_threshold=0.7)
        pm.receive_broadcast(1000.0)
        self.assertListEqual(pm.candidate_behaviors, [initial_schemes[1]])

        # Scenario 3: Multiple schemes, Multiple activated candidate behavior
        initial_schemes = [Scheme(context=v) for v in [1.0, 1000.0, 1000.0]]
        pm = ProceduralMemory(initial_schemes, context_match=lambda s, b: 1 - abs(b - s.context),
                              activation_threshold=0.7)
        pm.receive_broadcast(1000.0)
        self.assertListEqual(pm.candidate_behaviors, [initial_schemes[1], initial_schemes[2]])

        # Scenario 4: Multiple schemes, Random activated behavior
        initial_schemes = [Scheme(context=v) for v in [1.0, 2.0, 3.0]]
        pm = ProceduralMemory(initial_schemes, context_match=lambda s, b: 1 - abs(b - s.context),
                              activation_threshold=0.7)

        pm.receive_broadcast(1000.0)
        candidate_behaviors = pm.candidate_behaviors
        self.assertTrue(len(candidate_behaviors), 1)
