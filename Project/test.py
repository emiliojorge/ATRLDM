import base_algorithm
import gym
import numpy as np
import simulation
from gym.envs.registration import register

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
				 'Blackjack-v0',
				 'Roulette-v0',
				 'NChain-v0']

	rng = np.random.RandomState(25)
	env_names = rng.choice(env_names, size=12, replace=True)
	envs = [ gym.make(name) for name in env_names ]

	algorithm = base_algorithm.BaseAlgorithm()
	#algorithm = QAgent(eps_start=0.95, eps_end=0.05, eps_num=1000, learning_rate=lambda n: 1/n)
	# algorithm = DynaQAgent(planning_steps=50, eps_start=0.95, eps_end=0.05, eps_num=1000, learning_rate=lambda n: 1/n)
	#algorithm = Bayesian_Qlearning(action_selection="random", update_method="mom")
	#algorithm = RandomAgent()

	horizon = 20000
	num_trials = 3

	print(f'Running {len(env_names)} environments for {horizon} timesteps over {num_trials} trials...')
	scores = simulation.simulate_multiple_environment(envs, algorithm, T=horizon, num_trials=num_trials, discount=1)
	#print('Your score is', scores)

	mean_scores = np.mean(scores, axis=1)
	for i, score in enumerate(mean_scores):
		print(f'Environment: "{env_names[i]}"')
		print(f'-- Mean reward: {score}')

if __name__ == '__main__':
	main()
