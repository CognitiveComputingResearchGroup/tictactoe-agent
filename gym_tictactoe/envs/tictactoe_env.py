import logging
import random

import gym
import numpy as np
from gym import spaces

logger = logging.getLogger(__name__)


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=3):
        self._size = size
        self._board = None
        self._done = False
        self._n_moves = 0

        # reward values
        self.win_reward = 1
        self.lose_reward = -1
        self.draw_reward = 0
        self.illegal_move = -0.1

        # actions
        self.action_space = spaces.Discrete(self._size ** 2)

        # observations
        self.observation_space = spaces.Box(low=np.array([-2] * (self._size ** 2)),
                                            high=np.array([1] * (self._size ** 2)),
                                            dtype=np.int64)

    def step(self, action=None):

        reward = 0

        info = {
            'board': str(self._board),
            'done': str(self._done),
            'comment:': 'none'
        }

        # Check if current state is terminal, if so then return warning
        if self._done:
            logger.warning(
                """
                You are calling 'step()' even though this environment has already returned done = True. 
                You should always call 'reset()' once you receive 'done = True' -- any further steps are 
                undefined behavior.
                """
            )

            info['comment'] = 'post-game action'

            return self._board.asarray(), reward, self._done, info

        # Verify legal action -- action compatible with current board
        elif action not in self._board.blanks:
            logger.warning('Illegal action: ({})'.format(action))

            reward = self.illegal_move
            info['comment'] = 'illegal action'

        elif action is not None:
            # update board with player's action
            self._board[action] = X

            # check if player won
            if self._board.has_winner():

                reward = self.win_reward
                self._done = True

                info['comment'] = 'player wins'

            elif self._board.is_full():

                reward = self.draw_reward
                self._done = True

                info['comment'] = 'draw'

            # add opponent's action
            else:

                opponent_action = random.choice(self._board.blanks)
                self._board[opponent_action] = O

                # check if opponent won
                if self._board.has_winner():
                    reward = self.lose_reward
                    self._done = True

                    info['comment'] = 'opponent wins'

        info['board'] = str(self._board)
        info['done'] = self._done

        return self._board.asarray(), reward, self._done, info

    def reset(self):
        self._board = Board()
        self._done = False

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(self._board)
        else:
            return self._board.asarray()


BLANK = 0
X = 1
O = -1


class Board(object):

    def __init__(self, size=3):
        self.size = size
        self._board = np.array([BLANK] * (self.size ** 2))

        self._mark_dict = {BLANK: ' ', X: 'X', O: 'O'}

    @property
    def blanks(self):
        temp1 = list(enumerate(self))
        temp = [i for i, x in temp1 if x == BLANK]
        return temp

    def __getitem__(self, pos):
        return self._board[pos]

    def __setitem__(self, pos, mark):
        if mark not in [BLANK, X, O]:
            raise ValueError

        self._board[pos] = mark

    def __repr__(self):
        return str(self._board)

    def __str__(self):
        return self.board_string()

    def __eq__(self, other):
        if not isinstance(other, Board):
            return False

        return self._board == other._board

    def board_string(self):
        board_template = '\n' \
                         '{}│{}│{}\n' \
                         '─┼─┼─\n' \
                         '{}│{}│{}\n' \
                         '─┼─┼─\n' \
                         '{}│{}│{}\n'
        ox_board = [self._mark_dict[mark] for mark in self._board]
        return board_template.format(*ox_board)

    def is_full(self):
        return len(self.blanks) == 0

    def is_empty(self):
        return len(self.blanks) == self.size

    def asarray(self):
        return self._board

    def has_winner(self):
        board_2d = np.reshape(np.array(self._board), (self.size, self.size))

        # sum of board elements along rows
        row_sums = list(np.sum(board_2d, 0))

        # sum of board elements along columns
        col_sums = list(np.sum(board_2d, 1))

        # sum of elements along diagonals
        diag_sums = [np.sum(np.diag(board_2d)), np.sum(np.diag(np.flip(board_2d, axis=1)))]

        return True in (self.size == np.abs(row_sums + col_sums + diag_sums))

    def is_draw(self):
        return self.is_full() and not self.has_winner()
