import gym
from gym.envs.toy_text import frozen_lake, discrete
from gym.envs.registration import register

import numpy as np

import simulation
import base_algorithm

register(
    id='Deterministic-4x4-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False})

register(
    id='Deterministic-8x8-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '8x8',
            'is_slippery': False})

register(
    id='Stochastic-4x4-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': True})


def main():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    blackjack = gym.make('Blackjack-v0')
    algorithm = base_algorithm.BaseAlgorithm()
    #score = simulation.simulate(env, algorithm)
    # Score is a 2D array. Each row corresponds to an environment
    # And each column correspond to the cumulative reward collected for each trial
    score = simulation.simulate_multiple_environment([env, blackjack], algorithm)
    print('Your score is', score)
    print('Your mean score is ', np.mean(score, axis=1))


if __name__ == '__main__':
    main()
