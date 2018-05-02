import numpy as np

from gym import spaces

class MeanAgent(object):
    """An agent acting greedily and myopically by taking the action maximizing the reward in the next
       step which is taken as the mean of previous rewards."""

    def __init__(self):
        pass

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
        self.num_states = num_states
        self.num_action = num_action

        self.means = np.zeros((num_states, num_action))
        self.visits = np.zeros((num_states, num_action)) # Visit counts

    def observe_transition(self, state, action, next_state, reward):
        """
        Observe a new transition: state,action,next_state,reward
        This means at state and upon playing action you transition to
        next_state and obtains reward

        """

        # Online updating of mean
        self.visits[state,action] += 1
        self.means[state,action] += (reward - self.means[state,action]) / self.visits[state,action]

    def play(self, state):
        """
        Returns the action to play at state

        Args:
            state the state
        Returns:
            The action to play
        """
        if state is None:
            return 0

        return np.argmax(self.means[state,:])