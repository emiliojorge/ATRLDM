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
from util import EpsilonGreedy
from zapq_agent import ZapQAgent

AGENT_TYPES = {'q': QAgent,
               'dynaq': DynaQAgent,
               'bayesQ': Bayesian_Qlearning,
               'speedyQ': Speedy_Qlearning,
               'zapq': ZapQAgent,
               'mean': MeanAgent}


class BaseAlgorithm(object):
    def __init__(self, exploration=True, explorer=EpsilonGreedy(start=1.0, end=0.05, steps=2000), use_database=True, expert_steps=100, action_selection="moving average"):
        self.action_space = None
        self.num_action = None
        self.num_states = None
        self.expert_steps = expert_steps
        self.action_selection = action_selection

        self.exploration = exploration
        self.explorer = explorer

        self.use_database = use_database
        self.agent_database = defaultdict(list)

        self.expert_history = None
        self.expert_counter = None
        self.moving_average = None

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

        for name, params in config['start_agents'].items():
            if self.action_selection == "moving average":
                explorer = EpsilonGreedy(**config['agent_explorer'])
                agent = AGENT_TYPES[name](**params, explorer=deepcopy(explorer), exploration=True)
            else:
                agent = AGENT_TYPES[name](**params)
            self.agents.append(agent)

        np.random.seed(config['seed'])
        self.greediness = config['greediness']

    def meta_reset(self):
        """
        Hard reset of the algorithm including wiping the database.
        :return: None
        """
        self.__init__(self.exploration, self.explorer, self.use_database)
        self.explorer.reset()

    def reset(self):
        """
        Resets the algorithm so that it is ready for a new environment
        """
        # Save old agents if applicable
        if self.use_database and self.num_states is not None:
            if self.action_selection == "moving average":
                idx = self.moving_average.argsort()[-1:][::-1]
                for i in idx:
                    self.agent_database[(self.num_states, self.num_action)].append(deepcopy(self.agents[i]))
            else:
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

        # Get stored agents
        if self.use_database and self.num_states is not None:
            if len(self.agent_database[(self.num_states, self.num_action)]) <= 3:
                # We just pick the available three agents
                selected_agents = self.agent_database[(self.num_states, self.num_action)]
            else:
                # We randomly select three agents
                selected_agents = np.random.choice(self.agent_database[(self.num_states, self.num_action)], size=3, replace=False)

            for a in selected_agents:
                self.agents.insert(0, deepcopy(a))

        for a in self.agents:
            a.initialize(num_states, num_action, discount)

        if self.explorer:
            self.explorer.reset()

        self.expert_history = np.zeros(len(self.agents))
        self.expert_counter = None
        self.moving_average = np.zeros(len(self.agents))

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

        if self.action_selection == "moving average":
            self.expert_rewards.append(reward)

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

    def _epsilon_greedy_experts(self, state):
        """
        Let the expert agent play the state when following the moving average strategy.
        :param state: current state of the environment
        :return: action that the expert picks
        """
        if self.expert_counter is None or self.expert_counter > self.expert_steps:
            self._new_expert()

        self.expert_counter += 1

        # "Run down" epsilon in each agent even if they are not used.
        for i, a in enumerate(self.agents):
            if i != self.expert_id:
                try:
                    a.explorer.get_eps()
                except AttributeError:
                    pass

        return self.agents[self.expert_id].play(state)

    def _new_expert(self):
        """
        Selects a new expert agent based on an moving average strategy that plays for the next 100 steps.
        :return: None
        """
        if self.expert_counter is not None:
            self.moving_average[self.expert_id] = self.moving_average[self.expert_id]*0.9 + (1-0.9)*sum(self.expert_rewards)
        self.expert_rewards = []

        if np.random.random() < self.explorer.get_eps():
            self.expert_id = np.random.randint(len(self.agents))
        else:
            self.expert_id = np.argmax(self.moving_average+np.random.normal(0,0.00001, size = len(self.agents)))

        self.expert_history[self.expert_id] += 1
        self.expert_counter = 0

    def play(self, state):
        """
        Returns the action to play at state

        Args:
            state the state
        Returns:
            The action to play
        """

        if self.action_selection == "majority vote":
            if self.exploration and np.random.random() < self.explorer.get_eps():
                return np.random.randint(0, self.num_action)
            else:
                actions = []
                for agent in self.agents:
                    action = agent.play(state)
                    actions.append(action)
                return self._majority_vote(actions)

        elif self.action_selection == "moving average":
            return self._epsilon_greedy_experts(state)

        else:
            raise ValueError("Incorrect method for selecting actions")
