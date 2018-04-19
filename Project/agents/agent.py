class Agent(object):
	"""Base class for agents -- all agents should inherit this."""

	def __init__(self, env):
		"""Initialize the agent."""

		self.env = env

		# maybe extract number of actions and states from env here

	def observe(self, state, action, reward, next_state):
		"""Give the agent an observation (a transition)."""
		raise NotImplementedError

	def act(self, state):
		"""Return the agent's action in given state."""
		raise NotImplementedError