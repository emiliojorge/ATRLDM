import numpy as np

class Speedy_Qlearning(object):
	""" Implements the Asynchronous version of Speedy Qlearning.

	For documentation on arguments see Qlearning function above
	"""

	def __init__(self, exploration=False):
		self.algorithm = "Speedy_Qlearning"


	def reset(self):
		"""
		This should reset the algorithm so that it is ready for a new environment
		"""
		pass

	def initialize(self, num_states, num_action, discount):
		"""
		Initialize the algorithm just before its first interaction with a new environment
		Args:
			num_states: int the number of states in the environment
			num_action: int, the number of actions in the environment
			discount: double in [0, 1], the discount factor
		"""
		self.discount = discount
		self.num_states = num_states
		self.num_action = num_action
		self.Q = np.zeros((self.num_states, self.num_action))
		self.Q_old = np.zeros((self.num_states, self.num_action))
		self.k = np.zeros((self.num_states, self.num_action))

	def observe_transition(self, state, action, next_state, reward):
		"""
		Observe a new transition: state,action,next_state,reward
		This means at state and upon playing action you transition to
		next_state and obtains reward

		"""
		alpha = 1/(1+self.k[state,action])
		self.Q_old[state,action] =  self.Q[state, action]
		self.Q[state,action] = (1-alpha)*self.Q[state,action] + alpha*(
				self.k[state,action]*self.TkQsa(self.Q, next_state, reward)
				- (self.k[state,action]-1)*self.TkQsa(self.Q_old, next_state, reward))
		self.k[state,action] += 1


	def play(self, state):
		"""
		Returns the action to play at state

		Args:
			state the state
		Returns:
			The action to play
		"""
		return self.Q[state,:].argmax()



	def TkQsa(self, Qk, new_state, reward):
		return reward + self.discount*Qk[new_state,:].max()

