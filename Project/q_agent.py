import numpy as np

from gym import spaces

class QAgent(object):
    """An agent using Q-learning and an epsilon-greedy policy."""

    def __init__(self, double=False, eps_start=1.0, eps_end=0.05, eps_num=1000, learning_rate=lambda n: 1/n**0.5):
        self.double = double # Enables double Q-learning
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_num = eps_num
        self.learning_rate = learning_rate

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

        self.eps = lambda n: self.eps_start - (self.eps_start-self.eps_end)/self.eps_num * n

        if self.double:
            self.Q1 = np.zeros((num_states, num_action))
            self.Q2 = np.zeros((num_states, num_action))
        else:
            self.Q = np.zeros((num_states, num_action))

        self.nu = np.zeros((num_states, num_action)) # state-action pair visit counts

    def get_exploration(self):
        """Get linearly decaying for epsilon-greedy policy."""
        num_episodes = np.sum(self.nu)

        if num_episodes >= self.eps_num:
            return self.eps_end
        else:
            return self.eps_start - (self.eps_start-self.eps_end)/self.eps_num * num_episodes


    def get_step(self, state, action):
        """
        Return step size for the Q-learning.
        """
        return self.learning_rate(self.nu[state,action])

    def observe_transition(self, state, action, next_state, reward):
        """
        Observe a new transition: state,action,next_state,reward
        This means at state and upon playing action you transition to
        next_state and obtains reward

        """
        self.nu[state,action] += 1

        if self.double:
            # Flip a coin to decide which Q to update
            if np.random.random() < 0.5:
                max_a = np.argmax(self.Q1[next_state,:])
                self.Q1[state,action] += self.get_step(state, action) * (reward + self.discount * self.Q2[next_state,max_a] - self.Q1[state,action])
            else:
                max_a = np.argmax(self.Q2[next_state,:])
                self.Q2[state,action] += self.get_step(state, action) * (reward + self.discount * self.Q1[next_state,max_a] - self.Q2[state,action])
        else:
            self.Q[state,action] += self.get_step(state, action) * (reward + self.discount * np.max(self.Q[next_state,:]) - self.Q[state,action])

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

        if np.random.random() < self.get_exploration():
            return np.random.randint(0, self.num_action)
        else:
            if self.double:
                a = np.argmax(self.Q1[state,:] + self.Q2[state,:])
            else:
                a = np.argmax(self.Q[state,:])
            return a
