import numpy as np


class QBase(object):
    """An agent using Q-learning and an epsilon-greedy policy."""

    def __init__(self, eps_start=1.0, eps_end=0.05, eps_num=1000, learning_rate=lambda n: 1 / n ** 0.5):

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_num = eps_num
        self.learning_rate = learning_rate

        self.discount = None
        self.num_states = None
        self.num_action = None
        self.nu = None  # state-action pair visit counts

        self.eps = lambda n: self.eps_start - (self.eps_start - self.eps_end) / self.eps_num * n

    def reset(self):
        """
        This should reset the algorithm so that it is ready for a new environment
        """
        raise NotImplementedError("Please implement this function in your agent")

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
        self.nu = np.zeros((num_states, num_action))

    def _get_exploration(self):
        """Get linearly decaying for epsilon-greedy policy."""
        num_episodes = np.sum(self.nu)

        if num_episodes >= self.eps_num:
            return self.eps_end
        else:
            return self.eps_start - (self.eps_start - self.eps_end) / self.eps_num * num_episodes

    def _get_step(self, state, action):
        """
        Return step size for the Q-learning.
        """
        return self.learning_rate(self.nu[state, action])

    def observe_transition(self, state, action, next_state, reward):
        """
        Observe a new transition: state,action,next_state,reward
        This means at state and upon playing action you transition to
        next_state and obtains reward

        """
        raise NotImplementedError("Please implement this function in your agent")

    def play(self, state):
        """
        Returns the action to play at state

        Args:
            state the state
        Returns:
            The action to play
        """
        raise NotImplementedError("Please implement this function in your agent")
