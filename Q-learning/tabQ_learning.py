import numpy as np
import gym
import time
from lake_envs import *
import value_iteration
import utility


@utility.timing # Will print the time it takes to run this function.
def Qlearning(env, num_observations, gamma=0.95, learning_rate=utility.polynomial_learning_rate):
  """Learn state-action values using the Asynchronous Q-learning algorithm.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_observations: int 
    Number of observations of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: a Function taking an integer as argument and returning a float.
    Learning rate.  the return value is a  Number in range [0, 1).
    Use it by calling learning_rate(n)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state, action values
  """

  Q = np.zeros((env.nS, env.nA))

  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################


  return Q



@utility.timing # Will print the time it takes to run this function.
def speedy_Qlearning(env, num_observations, gamma=0.95, learning_rate=utility.polynomial_learning_rate):
  """ Implements the Asynchronous version of Speedy Qlearning.

  For documentation on arguments see Qlearning function above
  """

  Q = np.zeros((env.nS, env.nA))

  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################

  

  return Q


@utility.timing # Will print the time it takes to run this function.
def zap_QZerolearning(env, num_observations, gamma=0.95, learning_rate=utility.polynomial_learning_rate):
  """ Implements the tabular version (that is \theta should be Q) of Zap Q(0) learning.

  For documentation on arguments see Qlearning function above
  """

  Q = np.zeros((env.nS, env.nA))

  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################

  

  return Q




# Feel free to run your own debug code in main!
# You are not required to submit any of your debug code.
def main():
  env = gym.make('Stochastic-4x4-FrozenLake-v0')
  """
  You can also try other environments like

  env = gym.make('Deterministic-8x8-FrozenLake-v0')

  or

  env = gym.make('Deterministic-4x4-FrozenLake-v0')
  """
  Q = Qlearning(env, num_observations=1)

  """
  You can also run your other implementation like this

  Q = speedy_Qlearning(env, num_observations=1)
  or

  Q = zap_QZerolearning(env, num_observations=1)

  You can also run Value iteration (already implemented)
  and compare against you solution like this

  Q, _ = value_iteration.value_iteration(env)

  """
  

  # You can check if your solution is epsilon close to the optimal
  value_iteration.isQvalueErrorEpsilonClose(env, Q, gamma=0.95, epsilon=0.05)

  # You can compute the statistics of your solution
  utility.compute_statistics(env, Q)

  # You can render your final Q and watch your agent play one game
  utility.render_single_Q(env, Q)

if __name__ == '__main__':
    main()
