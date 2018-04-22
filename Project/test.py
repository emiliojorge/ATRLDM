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

register(
    id='Stochastic-8x8-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '8x8',
            'is_slippery': True})


def main():
    env_names = ['Deterministic-8x8-FrozenLake-v0',
                  'Stochastic-8x8-FrozenLake-v0',
                  'Blackjack-v0',
                  'Roulette-v0',
                  'NChain-v0']
    envs = [ gym.make(name) for name in env_names ]

    algorithm = base_algorithm.BaseAlgorithm()

    horizon = 1000
    num_trials = 10

    print(f'Running {len(env_names)} environments for {horizon} timesteps over {num_trials} trials...')
    scores = simulation.simulate_multiple_environment(envs, algorithm, T=horizon, num_trials=num_trials, discount=1)
    print('Your score is', scores)

    mean_scores = np.mean(scores, axis=1)
    for i, score in enumerate(mean_scores):
        print(f'Environment: "{env_names[i]}"')
        print(f'-- Mean reward: {score}')

if __name__ == '__main__':
    main()
