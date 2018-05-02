import numpy as np


class RandomAgent(object):
	"""An agent using random actions"""

	def __init__(self):
		self.algorithm = "Random"

	def reset(self):
		"""
		This should reset the algorithm so that it is ready for a new environment
		"""
		pass

	def initialize(self, num_states, num_action, discount):
		self.num_action = num_action

	def observe_transition(self, state, action, next_state, reward):
		pass

	def play(self, state):
		return np.random.randint(self.num_action)