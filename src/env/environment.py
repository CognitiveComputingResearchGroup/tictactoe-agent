#!/usr/bin/python
#  coding=utf-8

BLANK = -1
PLAYER1 = 0
PLAYER2 = 1
INVALID_MOVE = 9
PAIN = 1


class Board(object):

    def __init__(self, board):
        self._board_marks = {PLAYER1: 'X', PLAYER2: 'O', BLANK: ' '}
        self._board = board
        self._win_zones = [(0, 1, 2),
                           (3, 4, 5),
                           (6, 7, 8),
                           (0, 3, 6),
                           (1, 4, 7),
                           (2, 5, 8),
                           (0, 4, 8),
                           (2, 4, 6)]
        self._winner = None

    @classmethod
    def blank_board(cls):
        return cls([BLANK]*9)

    @property
    def blanks(self):
        temp1 = list(enumerate(self))
        temp = [i for i, x in temp1 if x == BLANK]
        return temp

    @property
    def first_blank(self):
        return self.blanks[0]

    @property
    def winner(self):
        return self._winner

    @property
    def looser(self):
        if self.winner is None:
            return None
        return PLAYER1 if self.winner == PLAYER2 else PLAYER2

    def is_blank(self, position):
        return True if self._board[position] == BLANK else False

    def reset(self):
        self._board = [BLANK]*9
        self._winner = None

    def haswon(self):

        def are_marks_same(marks, mark):
            return all([x == mark for x in marks])

        for indices in self._win_zones:
            marks_at_indices = map(lambda x: self._board[x], indices)
            if are_marks_same(marks_at_indices, PLAYER1):
                self._winner = PLAYER1
                return True
            if are_marks_same(marks_at_indices, PLAYER2):
                self._winner = PLAYER2
                return True
        return False

    def __getitem__(self, pos):
        return self._board[pos]

    def __setitem__(self, pos, mark):
        if mark not in ['X', 'O']:
            raise ValueError

        self._board[pos] = next(key for key, value in self._board_marks.items() if value == mark)

    def __iter__(self):
        return iter(self._board)

    def __repr__(self):
        return str(self._board)

    def __str__(self):
        return self.board_string()

    def board_string(self):
        board_template = '\n' \
                         '{}│{}│{}\n' \
                         '─┼─┼─\n' \
                         '{}│{}│{}\n' \
                         '─┼─┼─\n' \
                         '{}│{}│{}\n'
        ox_board = [self._board_marks[value] for value in self._board]
        return board_template.format(*ox_board)


def first_blank(board):
    return Board(board).first_blank

