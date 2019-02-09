import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='TicTacToe-v0',
    entry_point='gym_tictactoe.envs:TicTacToeEnv',
    timestep_limit=10,
    reward_threshold=1.0,
    nondeterministic=True,
)
