from agent import Agent

class DynaQAgent(Agent):
	"""Implements a Dyna-Q agent."""

	def __init__(self, env):
		"""Initialize the agent."""
		super().__init__(env)

	def observe(self, state, action, reward, next_state):
		"""Give the agent an observation (a transition)."""
		raise NotImplementedError

	def act(self, state):
		"""Return the agent's action in given state."""
		raise NotImplementedError