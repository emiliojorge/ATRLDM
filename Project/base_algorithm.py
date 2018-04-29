import json
import math

import numpy as np
from dynaq_agent import DynaQAgent
from gym import spaces
from q_agent import QAgent
from bayesian_qlearning import Bayesian_Qlearning
from zapq_agent import ZapQAgent
from speedyQ import Speedy_Qlearning


AGENT_TYPES = {'q': QAgent,
			   'dynaq': DynaQAgent,
			   'zapq': ZapQAgent,
			   'bayesQ': Bayesian_Qlearning,
			   'speedyQ': Speedy_Qlearning}


class BaseAlgorithm(object):
	def __init__(self):
		self.action_space = None

		self.agents = []
		self.greediness = None

		self._set_up()

	def _set_up(self):
		"""
		Sets up the Ensemble RL agent by reading configs from config.json.
		:return: None
		"""
		with open('config.json') as config_file:
			config = json.load(config_file)

		for a in config['start_agents']:
			agent = AGENT_TYPES[a]()
			self.agents.append(agent)

		self.greediness = config['greediness']

	def reset(self):
		"""
		This should reset the algorithm so that it is ready for a new environment
		"""
		# for a in self.agents:
		#     a.reset()
		self.agents = []
		self._set_up()

	def initialize(self, num_states, num_action, discount):
		"""
		Initialize the algorithm just before its first interaction with a new environment
		Args:
			num_states: int the number of states in the environment
			num_action: int, the number of actions in the environment
			discount: double in [0, 1], the discount factor
		"""
		self.action_space = spaces.Discrete(num_action)

		for a in self.agents:
			a.initialize(num_states, num_action, discount)

	def observe_transition(self, state, action, next_state, reward):
		"""
		Observe a new transition: state,action,next_state,reward
		This means at state and upon playing action you transition to
		next_state and obtains reward

		"""
		if not all((state, action, next_state, reward)):
			return

		for a in self.agents:
			a.observe_transition(state, action, next_state, reward)

	def _majority_vote(self, agents_actions):
		"""
		Implements the majority voting algorithm as presented in
		"Ensemble Algorithms in Reinforcement Learning" (M.A.Wiering and H. von Hasselt, 2008)
		:param agents_actions: List of actions that the agents picked
		:return: action
		"""

		actions = [a for a in range(self.action_space.n)]

		policy = np.zeros(len(actions))
		for action in actions:
			preference_value = agents_actions.count(action)
			policy[action] = math.exp(preference_value / (1 / self.greediness))

		policy /= np.sum(policy, 0)

		return np.random.choice(actions, p=policy)

	def play(self, state):
		"""
		Returns the action to play at state

		Args:
			state the state
		Returns:
			The action to play
		"""
		actions = []

		for agent in self.agents:
			action = agent.play(state)
			actions.append(action)

		return self._majority_vote(actions)
