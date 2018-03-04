import numpy as np


def evaluate_rewards_and_transitions(env):
    """ Compute the reward and transitions probabilities from a gym environment

    env: gym.core.Environment
    Environment. Must have nS, nA, and P as attributes.

    Returns the reward R and the probabilities T both matrix of size [nS, nA, nS]

    """

    # Intiailize T and R matrices
    R = np.zeros((env.nS, env.nA, env.nS))
    T = np.zeros((env.nS, env.nA, env.nS))

    # Iterate over states, actions, and transitions
    for state in range(env.nS):
        for action in range(env.nA):
            for transition in env.P[state][action]:
                probability, next_state, reward, done = transition
                R[state, action, next_state] = reward
                T[state, action, next_state] = probability

            # Normalize T across state + action axes
            T[state, action, :] /= np.sum(T[state, action, :])

    return R, T


def value_iteration(env, R=None, T=None, gamma=0.95, max_iterations=10**6, delta=10**-3):
    """ Runs Value Iteration on a gym environment 

    env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.

    R: matrix of size [nS, nA, nS] representing the rewards. If None, automatically
      computed from env.

    T: matrix of size [nS, nA, nS] representing the probabilities. 
        If None, it is automatically computed from env.

    gamma: float, the discount factor

    max_iterations: int, the maximum number of iterations to do for the value iteration

    delta: The precision desired. The value iteration stops, if the distance between
    two successive Value function is less than delta


    Returns: a tuple (Q, num_iteration)
    with Q An array of shape [env.nS x env.nA] representing state, action values
    and num_iteration the effective number of iteration run by the algorithm until
    convergence
    """
    V = np.zeros(env.nS)
    Q = np.zeros((env.nS, env.nA))
    if R is None or T is None:
        R, T = evaluate_rewards_and_transitions(env)

    for i in range(max_iterations):
        previous_V = V.copy()
        Q = np.einsum('ijk,ijk -> ij', T, R + gamma * V)
        V = np.max(Q, axis=1)

        if np.max(np.abs(V - previous_V)) < delta:
            break

    # Get and return optimal policy
 
    return Q, i + 1

def isQvalueErrorEpsilonClose(env, Q, gamma, epsilon):
    """
    Checks whether or not the provided Q value is epsilon close 
    to the optimal Q value

    Arguments:
    env: gym.core.Environment
    Environment. Must have nS, nA, and P as attributes.
    Q: The Q value to test
     An array of shape [env.nS x env.nA] representing state, action values

    gamma: float, The discount factor used to compute Q

    epsilon: float, check if ||Q-Q*||< epsilon

    Returns True if ||Q-Q*||< epsilon, False otherwise

    """

    # Computing optimal Q using value iteration and making sure it is
    # as close as possible to Q*
    delta=10**-5
    Qopt, _ = value_iteration(env, delta=delta, gamma=gamma)
    Vopt = np.dot(env.isd, np.max(Qopt, axis=1))

    # This is the error of the Q returned by value iteration 
    # according to [Puterman, 1994, Th. 6.3.1]
    VoptError = 2*delta*gamma/(1.-gamma)


    # The value of the provided Q
    V = np.dot(env.isd, np.max(Q, axis=1))

    isEpsilonClose = np.abs(V-Vopt) < VoptError + epsilon

    if isEpsilonClose:

        print("Great!! The value of your provided Q is {}, and this is {} close to the optimal value of {}".format(V, epsilon, Vopt))

    else:

        print("Not yet close!! The value of your provided Q is {}, and this is not {} close to the optimal value of {}".format(V, epsilon, Vopt))

    return isEpsilonClose


