import numpy as np


class QAgent(object):
    """An agent using Q-learning and an epsilon-greedy policy."""

    def __init__(self, double=False, lamda=0.75, exploration=False, explorer=None,
                 learning_rate=lambda n: 1 / n ** 0.5):
        self.double = double # Enables double Q-learning

        self.exploration = exploration
        self.explorer = explorer

        self.learning_rate = learning_rate
        self.algorithm = "Q-learning"

        self.eligibility_traces = None
        self.lamda = lamda
        self.learning_rates = None

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

        if self.explorer:
            self.explorer.reset()

        if self.double:
            self.Q1 = np.zeros((num_states, num_action))
            self.Q2 = np.zeros((num_states, num_action))
        else:
            self.Q = np.zeros((num_states, num_action))

        self.nu = np.zeros((num_states, num_action)) # state-action pair visit counts
        self.eligibility_traces = np.zeros((num_states, num_action))
        self.learning_rates = np.ones((num_states, num_action))

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
        self.learning_rates[state, action] = self.get_step(state, action)

        self.eligibility_traces = self.discount * self.lamda * self.eligibility_traces
        self.eligibility_traces[state, action] += 1

        if self.double:
            # Flip a coin to decide which Q to update
            if np.random.random() < 0.5:
                max_a = np.argmax(self.Q1[next_state,:])
                td_error = reward + self.discount * self.Q2[next_state, max_a] - self.Q1[state, action]
                self.Q1 += self.learning_rates * self.eligibility_traces * td_error
            else:
                max_a = np.argmax(self.Q2[next_state,:])
                td_error = reward + self.discount * self.Q1[next_state, max_a] - self.Q2[state, action]
                self.Q2 += self.learning_rates * self.eligibility_traces * td_error
        else:
            td_error = reward + self.discount * np.max(self.Q[next_state, :]) - self.Q[state, action]
            self.Q += self.learning_rates * self.eligibility_traces * td_error

        return td_error

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

        if self.exploration and np.random.random() < self.explorer.get_eps():
            return np.random.randint(0, self.num_action)
        else:
            if self.double:
                a = np.argmax(self.Q1[state,:] + self.Q2[state,:])
            else:
                a = np.argmax(self.Q[state,:])
            return a
