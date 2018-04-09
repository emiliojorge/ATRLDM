import numpy as np
import gym
import time
from lake_envs import *
import value_iteration
import utility
from collections import defaultdict
from scipy.special import gamma as gamma_fun
from scipy.special import digamma as digamma
import scipy.stats  as stats


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
		Learning rate.	the return value is a	Number in range [0, 1).
		Use it by calling learning_rate(n)

	Returns
	-------
	np.array
		An array of shape [env.nS x env.nA] representing state, action values
	"""

	Q = np.zeros((env.nS, env.nA))

	count = np.zeros([env.nS, env.nA])

	observations = utility.RandomStateObservation(env)
	for n in range(num_observations):
		state, action, reward, next_state = observations.observe()
		count[state, action] += 1
		Q[state,action] += learning_rate(count[state,action])*(reward + gamma*Q[next_state,:].max()-Q[state,action])
	return Q



@utility.timing # Will print the time it takes to run this function.
def speedy_Qlearning(env, num_observations, gamma=0.95, learning_rate=utility.polynomial_learning_rate):
	""" Implements the Asynchronous version of Speedy Qlearning.

	For documentation on arguments see Qlearning function above
	"""
	Q = np.zeros((env.nS, env.nA))
	Q_old = np.zeros((env.nS, env.nA))
	count = np.zeros([env.nS, env.nA])
	Q_new = np.zeros((env.nS, env.nA))
	r_k_sum = np.zeros((env.nS, env.nA))
	s_k = defaultdict(lambda: defaultdict(list))
	np.all(count>0)
	k=0
	alpha = 1/(k+1)
	observations = utility.RandomStateObservation(env)
	def TkQ_sa(Qk, s_k_sa, r_k_sa_sum, count_sa):
		return 1/count_sa * (r_k_sa_sum + gamma*np.sum([Qk[s,:].max() for s in s_k_sa]))

	for i in range(num_observations):
		state, action, reward, next_state = observations.observe()
		r_k_sum[state,action] += reward #sum up old rewards
		s_k[state][action].append(next_state) #add to list of s_k
		count[state, action] += 1

		Q_new[state,action] = (1-alpha)*Q[state,action] + alpha*(
				k*TkQ_sa(Q, s_k[state][action], r_k_sum[state,action], count[state,action])
				- (k-1)*TkQ_sa(Q_old, s_k[state][action], r_k_sum[state,action], count[state,action]))

		if np.all(count>0):
			k=k+1
			count = np.zeros([env.nS, env.nA])
			r_k_sum = np.zeros((env.nS, env.nA))
			s_k = defaultdict(lambda: defaultdict(list))
			alpha = 1/(k+1)
			Q_old = Q
			Q = Q_new
			Q_new = np.zeros((env.nS, env.nA))

	return Q



@utility.timing # Will print the time it takes to run this function.
def bayesian_Qlearning(env, num_observations, gamma=0.95, learning_rate=utility.polynomial_learning_rate):
	""" Implements the Bayesian Qlearning.

	For documentation on arguments see Qlearning function above
	"""
	n = np.zeros((env.nS, env.nA))
	mu0 = np.ones((env.nS, env.nA))*2
	lam = np.ones((env.nS, env.nA))*2
	alpha = np.ones((env.nS, env.nA))*2
	beta = np.ones((env.nS, env.nA))*2

	def cdf(x, mu, alpha, beta, lam):
		#Pr(mu<x)

		#scale location ????
		return stats.t.cdf((x-mu)*(lam*alpha/beta)**0.5, 2*alpha)



	def VPI(state):
		a_star = np.argmax(mu0[state,:])
		mu_star = mu0[state,a_star]
		mu_second = sorted(mu0[state,:])[-2]
		c = alpha[state,:]*gamma_fun(alpha[state,:]+1/2)*np.sqrt(beta[state])/(
		(alpha[state,:]-1/2)*gamma_fun(alpha[state,:])*gamma_fun(1/2)*
		alpha[state,:]*np.sqrt(2*lam[state,:]))*np.power(
		(1+mu0[state,:]**2)/(2*alpha[state,:]), -alpha[state,:]+1/2)

		vpi = []
		for action, (a, mu, b, l, c_i) in enumerate(zip(alpha[state,:], mu0[state,:], beta[state,:], lam[state,:], c )):
			if action == a_star:
				vpi.append(c_i + (mu_second-mu_star)*cdf(mu_second, mu, a, b, l))
			else:
				vpi.append(c_i + (mu-mu_star)*(1-cdf(mu_star, mu, a, b, l)))
		return vpi

	def update_posterior(state, action, M1, M2):
		n[state,action] += 1

		mu0[state, action] = (lam[state,action]*mu0[state,action]+
					n[state,action]*M1)/(lam[state,action]+n[state,action])
		lam[state,action] += n[state,action]
		alpha[state,action] += n[state,action]/2
		beta[state,action] +=n[state,action]/2*(M2-M1**2) + (
		n[state,action]*lam[state,action]*(M1-mu0[state,action])**2/(2*(lam[state,action] + n[state,action]))
		)


	def moments(r, next_state):
		next_action = np.argmax(mu0[next_state,:])
		if next_state == "done":
			E_r = 0
			E_r_sq = 0
		else:
			E_r = mu0[next_state,next_action]
			E_r_sq = (lam[next_state, next_action]+1)/lam[next_state,next_action]*(
			beta[next_state,next_action]/(alpha[next_state,next_action]-1)) + mu0[next_state,next_action]**2

		M1 = r + gamma * E_r
		M2 = r**2 + 2*gamma*r*E_r + gamma**2*E_r_sq
		return M1, M2


	observation = env.reset()
	for i in range(num_observations):
		action = np.argmax(VPI(observation)+mu0[observation,:])
		next_observation, reward, done,  info = env.step(action)

		if done:
			next_observation = "done"
		M1,M2 = moments(reward, next_observation)
		update_posterior(observation, action, M1, M2)

		if done:
			observation = env.reset()
		else:
			observation = next_observation

	#scipy.special.digamma(z)
	return mu0

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
