from unittest import TestCase

from common import StructureBuildingCodelet, Workspace, is_board, create_move
from env.environment import Board, BLANK


class TestStructureBuildingCodelet(TestCase):
    def test_init(self):
        try:
            c = StructureBuildingCodelet()
            c = StructureBuildingCodelet(select=lambda x: False)
            c = StructureBuildingCodelet(transform=lambda x: x + 1)
            c = StructureBuildingCodelet(select=lambda x: False, transform=lambda x: x + 1)
        except Exception as e:
            self.fail(e)

    def test_call(self):

        # With default select and action
        try:
            c = StructureBuildingCodelet()
            w = Workspace()

            # Add content to the Workspace
            for i in range(10):
                w(i)

            # invoke __call__ several times (testing for exceptions)
            for v in range(100):
                c(w)

        except Exception as e:
            self.fail(e)

        # With custom select and action
        try:
            c = StructureBuildingCodelet(select=lambda x: x < 3, transform=lambda x: x + 1)
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
                c = StructureBuildingCodelet(select=lambda x: x == v, transform=lambda x: x + 1)
                c(w)
                actual = next(c)
                self.assertEqual([v + 1], actual)
        except Exception as e:
            self.fail(e)

    def test_is_board(self):
        try:
            self.assertTrue(is_board(Board.blank_board()))
            self.assertFalse(is_board('NOT A BOARD'))
            self.assertFalse(is_board(None))

        except Exception as e:
            self.fail(e)

    def test_create_move(self):
        try:
            BLANK_BOARD = Board.blank_board()

            move, new_board = create_move(BLANK_BOARD)
            self.assertIsNotNone(move)
            self.assertIsNotNone(new_board)

            expected_board = BLANK_BOARD.copy()
            expected_board[move[0]] = move[1]

            self.assertEqual(new_board, expected_board)

        except Exception as e:
            self.fail(e)

    def test_create_move_sbc(self):

        EMPTY_BOARD = Board.blank_board()
        SINGLE_BLANK_BOARD = Board(['O', 'X', 'O', 'X', 'O', 'X', 'O', BLANK, 'O'])
        FULL_BOARD = Board(['X'] * 9)

        try:
            # Scenario #1 - Empty Board
            ###########################
            sbc = StructureBuildingCodelet(select=lambda s: is_board(s) and not s.is_full(), transform=create_move)

            # Update Workspace
            workspace = Workspace()
            workspace(EMPTY_BOARD)

            # Update SBC and Retrieve New Structures
            sbc(workspace)

            # Should only be one new structure
            new_structures = next(sbc)
            self.assertEqual(len(new_structures), 1)

            # Both move and new board should be defined
            move, new_board = new_structures[0]
            self.assertIsNotNone(move)
            self.assertIsNotNone(new_board)

            # Should be a legal move
            self.assertTrue(0 <= move.position < len(EMPTY_BOARD))

            # Move and returned board should be consistent
            expected_board = EMPTY_BOARD.copy()
            expected_board[move.position] = move.mark
            self.assertEqual(new_board, expected_board)

            # Scenario #2 - Full Board
            ##########################
            sbc = StructureBuildingCodelet(select=lambda s: is_board(s) and not s.is_full(), transform=create_move)

            # Update Workspace
            workspace = Workspace()
            workspace(FULL_BOARD)

            # Update SBC and Retrieve New Structures
            sbc(workspace)
            new_structures = next(sbc)
            self.assertEqual(len(new_structures), 0)

            # Scenario #3 - Single Open Position
            ####################################
            sbc = StructureBuildingCodelet(select=lambda s: is_board(s) and not s.is_full(), transform=create_move)

            # Update Workspace
            workspace = Workspace()
            workspace(SINGLE_BLANK_BOARD)

            # Update SBC and Retrieve New Structures
            sbc(workspace)
            new_structures = next(sbc)
            self.assertEqual(len(new_structures), 1)

            move, new_board = new_structures[0]
            blank_pos = SINGLE_BLANK_BOARD.first_blank
            self.assertEqual(move, (blank_pos, 'X'))

            expected_board = SINGLE_BLANK_BOARD.copy()
            expected_board[move.position] = move.mark

            self.assertEqual(new_board, expected_board)
            self.assertTrue(expected_board.is_full())

            # Scenario #4 - Multiple Boards in Workspace
            ############################################
            sbc = StructureBuildingCodelet(select=lambda s: is_board(s) and not s.is_full(), transform=create_move)

            # Update Workspace
            workspace = Workspace()

            workspace(SINGLE_BLANK_BOARD)
            workspace(FULL_BOARD)
            workspace(EMPTY_BOARD)

            # Update SBC and Retrieve New Structures
            sbc(workspace)
            new_structures = next(sbc)

            # Full board filtered out by select criteria
            self.assertEqual(len(new_structures), 2)

        except Exception as e:
            self.fail(e)
