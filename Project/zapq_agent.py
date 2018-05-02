import numpy as np


class ZapQAgent(object):
    """An agent using Q-learning and an epsilon-greedy policy."""

    def __init__(self, exploration=False, explorer=None, learning_rate=lambda n: 1 / n):

        self.exploration = exploration
        self.explorer = explorer

        self.learning_rate = learning_rate
        self.learning_rate_gamma = lambda n: 1 / n ** 0.75

        self.algorithm = "ZapQAgent"

        self.discount = None
        self.num_states = None
        self.num_action = None

        self.lamda = 0.95

        self.Q = None
        self.A = None
        self.nu = None  # state-action pair visit counts
        self.n = 0
        self.eligibility_traces = None

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

        self.Q = np.zeros((num_states, num_action))
        d = num_states * num_action
        self.A = np.zeros((d, d))
        self.eligibility_traces = np.zeros((d, 1))

        self.nu = np.zeros((num_states, num_action))  # state-action pair visit counts

        if self.explorer:
            self.explorer.reset()

    def get_step(self, state, action):
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

        def psi(s, a):
            d = np.zeros((self.num_states, self.num_action))
            d[s, a] = 1
            return d.reshape((self.num_states * self.num_action, 1))

        self.nu[state, action] += 1

        next_action = np.argmax(self.Q[next_state, :])
        td_error = reward + self.discount * self.Q[next_state, next_action] - self.Q[state, action]

        next_A = self.eligibility_traces.dot(
            np.transpose((self.discount * psi(next_state, next_action) - psi(state, action))))
        self.A = self.A + self.learning_rate_gamma(self.n + 1) * (next_A - self.A)

        inv_A = np.linalg.pinv(self.A)
        next_Q = self.Q.reshape((self.num_states * self.num_action, 1)) - self.learning_rate(self.n + 1) * inv_A.dot(
            self.eligibility_traces) * td_error
        self.Q = next_Q.reshape((self.num_states, self.num_action))

        self.eligibility_traces = self.lamda * self.discount * self.eligibility_traces + psi(next_state, next_action)

        self.n += 1

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
            return np.argmax(self.Q[state, :])
