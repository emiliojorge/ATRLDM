
import time
import numpy as np

class RandomPolicyObservation(object):
	"""Generate observation by running a random policy.

	How to use it?

	observations = RandomPolicyObservation(env) # to initialize it

	Then to get a new observation use

	state, action, reward, next_state = observations.observe()

	Here s_n is state, a_n is action, r_n is reward and s_{n+1} is next_state

	"""
	def __init__(self, env):
		"""

		Arguments:

			env: Environment gym.core.Environment

		"""
		self.env = env
		self.env.reset()

	def observe(self):
		"""
		Generate a new observation

		Returns a tuple (state, action, reward, next_state ) = (s_n, a_n, r_n, s_{n+1})
		"""

		state = self.env.s
		action = self.env.action_space.sample()
		next_state, reward, done, _ = self.env.step(action)
		if done:
			self.env.reset()

		return (state, action, reward, next_state)




class RandomStateObservation(object):
	"""Generate observations by sampling the (state, action) uniformly randomly.

	This is guaranteed to visit all possible state,action pairs as the
	number of observations tend to infinity (regardless of the properties of the MDP).

	How to use it?

	observations = RandomStateObservation(env) # to initialize it

	Then to get a new observation use

	state, action, reward, next_state = observations.observe()

	Here s_n is state, a_n is action, r_n is reward and s_{n+1} is next_state

	"""
	def __init__(self, env):
		"""

		Arguments:

			env: Environment gym.core.Environment

		"""
		self.env = env
		self.env.reset()

	def observe(self):
		"""
		Generate a new observation

		Returns a tuple (state, action, reward, next_state ) = (s_n, a_n, r_n, s_{n+1})
		"""
		state = self.env.observation_space.sample()
		action = self.env.action_space.sample()
		self.env.s = state
		next_state, reward, done, _ = self.env.step(action)
		if done:
			self.env.reset()

		return (state, action, reward, next_state)


def timing(f):
	""" Function wrapper to print the time it tooks to execute a given function """
	def wrap(*args, **kwargs):
		time1 = time.time()
		ret = f(*args, **kwargs)
		time2 = time.time()
		print('Function took %0.3f ms' % ((time2-time1)*1000.0))
		return ret
	return wrap



def compute_statistics(env, Q, num_interactions=100000):

	""" Function to compute and print the total rewards of the greedy policy with respect to Q

	env: gym.core.Environment
		The environment

	Q: An array of shape [env.nS x env.nA] representing state, action values

	num_interactions: the number of interactions with the environment

	Returns the total rewards obtained by the policy greedy(Q)

	"""



	state = env.reset()

	total_reward = 0.

	for i in range(num_interactions):

		action = np.argmax(Q[state])
		state, reward, done, _ = env.step(action)
		total_reward += reward

		if done:
			state = env.reset()

	print("Total reward= %f after %f interactions with the environment" % (total_reward,num_interactions))

	return total_reward







def render_single_Q(env, Q):
	"""Renders Q function once on environment. Watch your agent play!

	Note that if your have a really bad agent it might never finish the game

	Parameters
	----------
	env: gym.core.Environment
	  Environment to play Q function on. Must have nS, nA, and P as
	  attributes.
	Q: np.array of shape [env.nS x env.nA]
	  state-action values.
	"""

	episode_reward = 0
	state = env.reset()
	done = False
	while not done:
		env.render()
		time.sleep(0.5) # Seconds between frames. Modify as you wish.
		action = np.argmax(Q[state])
		state, reward, done, _ = env.step(action)
		episode_reward += reward

	print( "Episode reward: %f" % episode_reward)


def polynomial_learning_rate(n, w=1.):
	""" Implements a polynomial learning rate of the form (1/n**w)

	n: Integer
		The iteration number
	w: float between (0.5, 1]

	Returns 1./n**w as the rate
	"""
	assert n > 0, "Make sure the number of times a state action pair has been observed is always greater than 0 before calling polynomial_learning_rate"

	return 1./n**w


