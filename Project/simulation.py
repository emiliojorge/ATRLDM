import numpy as np
from gym import spaces


def get_size_space(space):
    """
    Get the total dimension of a space
    Args:
    space: Instance of gym.Spaces
    """

    if isinstance(space, spaces.Tuple):
        num = 1
        for sub_space in space.spaces:
            num *= sub_space.n
        return num

    if isinstance(space, spaces.Discrete):
        return space.n

    raise NotImplementedError


def convert_space_to_integer(data, space):
    """
        Convert a space into integer
        data: An observation from this space
        space: Instance of gym.Spaces
        Returns: Integer encoding the data
    """

    if isinstance(data, tuple):

        idx = data[0]
        tmp = 1

        for i in range(1, len(data)):
            tmp *= space.spaces[i-1].n
            idx += tmp*data[i]

        return idx

    if isinstance(data, int):
        return data

    #raise NotImplementedError




def simulate(env, algorithm, T=4194304, num_trials=20, discount=1):
    """
        This run a simulation over many trials and return the minimum score over the trials
    Args:
        env: An openai gym environment
        algorithm:  An algorithm deriving from BaseAlgorithm
        T: The horizon
        num_trials: The number of trials
        discount The discount factor
    """
    cumulative_rewards = np.zeros(num_trials)
    for trial in range(num_trials):
        # Reset environment and get initial state
        s = env.reset()

        # Reset algorithm
        algorithm.reset()
        algorithm.initialize(get_size_space(env.observation_space), get_size_space(env.action_space), discount)

        # Simulate with the environment for T steps
        episode_t = 0
        for t in range(T):

            # Get the action from the algorithm
            converted_s = convert_space_to_integer(s, env.observation_space)
            a = algorithm.play(converted_s)

            # Play the action
            next_state, reward, done, _ = env.step(a)
            converted_s_next = convert_space_to_integer(next_state, env.observation_space)

            # Let the algorithm observe the transition
            algorithm.observe_transition(converted_s, a, converted_s_next, reward)

            # Update the cumulative rewards and the state
            cumulative_rewards[trial] += reward*(discount**episode_t)
            s = next_state
            episode_t += 1
            if done:
                s = env.reset()
                episode_t = 0

    return cumulative_rewards


# Over many environments your final score will be the average of the score
# returned by single experiment
def simulate_multiple_environment(envs, algorithm, T=1000, num_trials=4, discount=1):

    rewards = [None]*len(envs)

    for i in range(len(envs)):
        env = envs[i]
        rewards[i] = simulate(env, algorithm, T, num_trials, discount)

    return np.asarray(rewards)
