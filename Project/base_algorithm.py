from gym import spaces


class BaseAlgorithm(object):
    def __init__(self):
        self.action_space = None

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
        self.action_space = spaces.Discrete(num_action)

    def observe_transition(self, state, action, next_state, reward):
        """
        Observe a new transition: state,action,next_state,reward
        This means at state and upon playing action you transition to
        next_state and obtains reward

        """
        pass

    def play(self, state):
        """
        Returns the action to play at state

        Args:
            state the state
        Returns:
            The action to play
        """

        return self.action_space.sample()

