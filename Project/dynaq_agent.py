import numpy as np

from gym import spaces

class DynaQAgent(object):
    """A Dyna-Q agent using an epsilon-greedy policy."""

    def __init__(self, planning_steps=50, eps_start=1.0, eps_end=0.05, eps_num=1000, learning_rate=lambda n: 1/n**0.5):
        self.planning_steps = planning_steps
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

        self.Q = np.zeros((num_states, num_action))
        self.nu = np.zeros((num_states, num_action)) # state-action pair visit counts

        self.model = {}

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

    def update_model(self, state, action, next_state, reward):
        """
        Updates the model with the given transition.
        """
        if (state, action) not in self.model.keys():
            self.model[(state, action)] = []

        self.model[(state, action)].append((next_state, reward))

    def sample_result(self, state, action):
        """
        Samples the model for the result of taking given action in given state.
        """
        assert (state, action) in self.model.keys()

        r = np.random.randint(len(self.model[(state, action)]))
        (next_state, reward) = self.model[(state, action)][r]
        return (next_state, reward)

    def sample_state_action(self):
        """
        Sample previously observed state and action.
        """

        # Not sure if state should be sampled first and then action rather than both simultaneously.
        # Doing this for now for simplicity.
        r = np.random.randint(len(self.model.keys()))
        (state, action) = list(self.model.keys())[r]
        return (state, action)

    def plan(self):
        """
        Do the planning step in the Dyna-Q algorithm.
        """
        for n in range(self.planning_steps):
            (state, action) = self.sample_state_action()
            (next_state, reward) = self.sample_result(state, action)
            self.Q[state,action] += self.get_step(state, action) * (reward + self.discount * np.max(self.Q[next_state,:]) - self.Q[state,action])

    def observe_transition(self, state, action, next_state, reward):
        """
        Observe a new transition: state,action,next_state,reward
        This means at state and upon playing action you transition to
        next_state and obtains reward

        """
        self.nu[state,action] += 1
        self.Q[state,action] += self.get_step(state, action) * (reward + self.discount * np.max(self.Q[next_state,:]) - self.Q[state,action])
        
        if state: # Don't add terminal state to model
            self.update_model(state, action, next_state, reward)
        
        if len(self.model.keys()) > 0:
            self.plan()

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
            return np.argmax(self.Q[state,:])
