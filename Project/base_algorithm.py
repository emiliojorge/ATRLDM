from random import choice

from gym import spaces


class BaseAlgorithm(object):
    def __init__(self):
        self.action_space = None

        self.agents = []

    def reset(self):
        """
        This should reset the algorithm so that it is ready for a new environment
        """
        for a in self.agents:
            a.reset()


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

        for a in self.agents:
            a.observe_tranisition(state, action, next_state, reward)

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

        return choice(actions)
