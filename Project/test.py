import time

import gym
import numpy as np
import simulation
from base_algorithm import BaseAlgorithm
from gym.envs.registration import register
from bayesian_qlearning import Bayesian_Qlearning
from random_agent import RandomAgent
from util import EpsilonGreedy

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
    env_names = ['Deterministic-4x4-FrozenLake-v0',
                 'Deterministic-8x8-FrozenLake-v0',
                 'Stochastic-8x8-FrozenLake-v0',
                 'Stochastic-4x4-FrozenLake-v0',
                 'Blackjack-v0',
                 'Roulette-v0',
                 'NChain-v0']

    rng = np.random.RandomState(25)

    repeat_environments = False
    compare_random = True

    save_results = True

    if repeat_environments == True:
        env_names = rng.choice(env_names, size=12, replace=True)

    envs = [gym.make(name) for name in env_names]

    algorithm = BaseAlgorithm(exploration=True, explorer=EpsilonGreedy(start=0.5, end=0.05, steps=1000),
                              use_database=True)

    # algorithm = ZapQAgent()
    # algorithm = QAgent(exploration=True, explorer=EpsilonGreedy(start=0.95, end=0.05, steps=1000))#eps_start=0.95, eps_end=0.05, eps_num=1000, learning_rate=lambda n: 1/n)
    # algorithm = DynaQAgent()#planning_steps=50, eps_start=0.95, eps_end=0.05, eps_num=1000, learning_rate=lambda n: 1/n)
    # algorithm = Bayesian_Qlearning()#action_selection="q-sampling", update_method="mom")
    # algorithm = Speedy_Qlearning()
    # algorithm = MeanAgent()

    random_algorithm = RandomAgent()

    horizon = 100000
    num_trials = 10

    print(f'Running {len(env_names)} environments for {horizon} timesteps over {num_trials} trials...')
    start = time.time()
    scores = simulation.simulate_multiple_environment(envs, algorithm, T=horizon, num_trials=num_trials, discount=1)
    print(f'{algorithm.algorithm} took {time.time()-start} seconds')
    # print('Your score is', scores)
    mean_scores = np.mean(scores, axis=1)
    std_scores = np.std(scores, axis=1)

    if compare_random == True:
        random_scores = simulation.simulate_multiple_environment(envs, random_algorithm, T=horizon,
                                                                 num_trials=num_trials,
                                                                 discount=1)
        mean_random = np.mean(random_scores, axis=1)
        std_random = np.std(random_scores, axis=1)
        for i, (score, rand_score, score_std, rand_std) in enumerate(
                zip(mean_scores, mean_random, std_scores, std_random)):
            print(f'Environment: "{env_names[i]}"')
            print(f'-- Mean reward: {score} -- Std: {score_std}')
            print(f'-- Random reward: {rand_score} -- Std: {rand_std}')

    else:
        for i, (score, score_std) in enumerate(zip(mean_scores, std_scores)):
            print(f'Environment: "{env_names[i]}"')
            print(f'-- Mean reward: {score} -- Var: {score_std}')

    if save_results:
        with open('results.csv', 'w') as file:
            file.write('environment, runs, trials, mean_score, std_deviation\n')
            for env, mean, sigma in zip(env_names, mean_scores, std_scores):
                file.write('{}, {}, {}, {}, {}\n'.format(env, horizon, num_trials, mean, sigma))
    else:
        return env_names, scores

if __name__ == '__main__':
    main()
