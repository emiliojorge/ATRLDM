import json
import math
from collections import defaultdict
from copy import deepcopy
import numpy as np
from bayesian_qlearning import Bayesian_Qlearning
from dynaq_agent import DynaQAgent
from gym import spaces
from mean_agent import MeanAgent
from q_agent import QAgent
from speedyQ import Speedy_Qlearning
from zapq_agent import ZapQAgent


AGENT_TYPES = {'q': QAgent,
               'dynaq': DynaQAgent,
               'bayesQ': Bayesian_Qlearning,
               'speedyQ': Speedy_Qlearning,
               'mean': MeanAgent}


class BaseAlgorithm(object):
    def __init__(self, exploration=False, explorer=None, use_database=True):
        self.action_space = None
        self.use_database = use_database
        self.exploration = exploration
        self.explorer = explorer
        self.agent_database = defaultdict(list)
        self.num_action = None
        self.num_states = None

        self.agents = []
        self.greediness = None
        self.algorithm = "BaseAlgorithm"
        self._set_up()

    def _set_up(self):
        """
        Sets up the Ensemble RL agent by reading configs from config.json.
        :return: None
        """
        with open('config.json') as config_file:
            config = json.load(config_file)

        for a in config['start_agents']:
            agent = AGENT_TYPES[a]()
            self.agents.append(agent)

        self.greediness = config['greediness']


    def reset(self):
        """
        This should reset the algorithm so that it is ready for a new environment
        """
        # for a in self.agents:
        #     a.reset()

        #Save old agents if applicable
        if self.use_database == True  and self.num_states != None:
            for a in self.agents:
                self.agent_database[(self.num_states, self.num_action)].append(deepcopy(a))

        self.agents = []
        self._set_up()


    def initialize(self, num_states, num_action, discount):
        """
        Initialize the algorithm just before its first interaction with a new environment
        Args:
            num_states: int the number of states in the environment
            num_action: int, the number of actions in the environment
            discount: double in [0, 1], the discount factor
        """

        self.action_space = spaces.Discrete(num_action)

        self.num_action = num_action
        self.num_states = num_states

        #Get stored agents
        if (self.use_database == True and self.num_states != None and
        self.agent_database[(self.num_states, self.num_action)]!=[]):
            for a in np.random.choice(self.agent_database[(self.num_states, self.num_action)], size=2):
                self.agents.append(a)

        for a in self.agents:
            a.initialize(num_states, num_action, discount)

        if self.explorer:
            self.explorer.reset()

    def observe_transition(self, state, action, next_state, reward):
        """
        Observe a new transition: state,action,next_state,reward
        This means at state and upon playing action you transition to
        next_state and obtains reward

        """
        if None in (state, action, next_state, reward):
            raise ValueError(state, action, next_state, reward)

        for a in self.agents:
            a.observe_transition(state, action, next_state, reward)

    def _majority_vote(self, agents_actions):
        """
        Implements the majority voting algorithm as presented in
        "Ensemble Algorithms in Reinforcement Learning" (M.A.Wiering and H. von Hasselt, 2008)
        :param agents_actions: List of actions that the agents picked
        :return: action
        """

        actions = [a for a in range(self.action_space.n)]

        policy = np.zeros(len(actions))
        for action in actions:
            preference_value = agents_actions.count(action)
            policy[action] = math.exp(preference_value / (1 / self.greediness))

        policy /= np.sum(policy, 0)

        return np.random.choice(actions, p=policy)

    def play(self, state):
        """
        Returns the action to play at state

        Args:
            state the state
        Returns:
            The action to play
        """

        if self.exploration and np.random.random() < self.explorer.get_eps():
            return np.random.randint(0, self.num_action)
        else:
            actions = []

            for agent in self.agents:
                action = agent.play(state)
                actions.append(action)

            return self._majority_vote(actions)
