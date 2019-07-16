from common import *
import tqdm
import numpy as np
from agent import run

import gym
import sys
sys.path.append("..")
import gym_tictactoe  # Needed to add 'TicTacToe-v0' into gym registry

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Number of cognitive cycles to execute (None -> forever)
# Number of cognitive cycles to execute (None -> forever)
N_STEPS = 1000
experimenting = True


if __name__ == '__main__':
    rewards = []
    for i in tqdm.tqdm(range(100)):
        environment = gym.make('TicTacToe-v0')
        environment.reset()
        _, reward = run(environment, n=N_STEPS, render=not experimenting)
        rewards.append(reward)

    print(np.mean(rewards))

