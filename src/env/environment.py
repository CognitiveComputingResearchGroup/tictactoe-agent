#!/usr/bin/python
#  coding=utf-8
from functools import partial

import lidapy
from lidapy import Config
from lidapy import Task
from lidapy.modules import Environment
import numpy as np

# Topic definitions
board_state_topic = lidapy.Topic('oxplayer/env/board')
action_topic = lidapy.Topic('oxplayer/player2/action')
turn_topic = lidapy.Topic('oxplayer/env/turn')
pain_topics = [lidapy.Topic('oxplayer/player1/pain'),
               lidapy.Topic('oxplayer/player2/pain')]
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

    def iswinning(self):

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

    def __str__(self):
        return str(self._board)

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


class OXPlayerEnvironment(Environment):

    def __init__(self, tasks=None):
        super(OXPlayerEnvironment, self).__init__(tasks=tasks)
        self._board = Board.blank_board()
        self._turn = PLAYER1
        self._player1 = partial(np.random.choice, a=self._board.blanks)
        self._player2 = OXPlayerEnvironment.receive_move

        self.builtin_tasks = [Task(name='update_env', callback=self.update),
                              Task(name='publish_board', callback=self.publish_board),
                              Task(name='publish_turn', callback=self.publish_turn)]

    def publish_board(self):
        board_state_topic.send(str(self._board))
        lidapy.loginfo('Published from Environment to '+str(board_state_topic)+':'+str(self._board))

    def publish_turn(self):
        turn_topic.send(self._turn)
        lidapy.loginfo('Published from Environment to '+str(turn_topic)+':'+str(self._turn))

    def _is_end(self):
        if self._board.iswinning():
            return True
        if BLANK not in self._board:
            return True
        return False

    @staticmethod
    def receive_move():
        msg = action_topic.receive(timeout=1)
        move = INVALID_MOVE
        if msg is not None:
            lidapy.loginfo('Environment received:'+str(msg))
            move = int(msg)
        return move

    def _make_move(self, pos):
        if -1 < pos < 9:
            if self._board[pos] == -1:
                self._board[pos] = 'X' if self._turn == 0 else 'O'
                return True
        return False

    def board_string(self):
        return self._board.board_string()

    def move_possible(self, move):
        return 0 <= move < 9 and self._board.is_blank(move)

    def update(self):
        if self._is_end():
            pain_topics[self._board.looser].send(PAIN)
            self._turn = self._board.winner
            self._board.reset()

        if self._turn == PLAYER1:
            move = self._player1(), PLAYER1
        else:
            move = self._player2(), PLAYER2  # receive_action should be bound to an instance

        if self.move_possible(move[0]) and self._turn == move[1]:
            if self._make_move(move[0]):
                self._turn = PLAYER2 if self._turn == PLAYER1 else PLAYER1

        lidapy.loginfo(self.board_string())
        print('running')


if __name__ == '__main__':

    agent_config = Config()
    agent_config.set_param('rate_in_hz', 1)
    lidapy.init(config=agent_config)

    ox_env = OXPlayerEnvironment()
    ox_env.start()

