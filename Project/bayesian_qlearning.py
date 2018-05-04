import numpy as np
from collections import defaultdict
from scipy.special import gamma as gamma_fun
from scipy.special import digamma as digamma
import scipy.stats  as stats
from scipy.integrate import tplquad as tplquad
from scipy.optimize import brentq as brentq

class Bayesian_Qlearning():
	""" Implements the Bayesian Qlearning.
	"""
	def __init__(self, action_selection="q-sampling", update_method="mom", parameter_decay=0.99, explorer=None, exploration=None):
		self.action_selection=action_selection
		self.update_method = update_method
		self.parameter_decay = parameter_decay
		self.algorithm = "Bayesian_Qlearning"

	def initialize(self, num_states, num_action, discount, mu_init=1, lam_init=1, alpha_init=1.05, beta_init=1):
		self.discount = discount
		self.num_states = num_states
		self.num_action = num_action
		self.n = np.ones((self.num_states, self.num_action))
		self.mu0 = np.ones((self.num_states, self.num_action))*mu_init
		self.lam = np.ones((self.num_states, self.num_action))*lam_init
		self.alpha = np.ones((self.num_states, self.num_action))*alpha_init
		self.beta = np.ones((self.num_states, self.num_action))*beta_init

	def reset(self):
		"""
		This should reset the algorithm so that it is ready for a new environment
		"""
		pass

	@staticmethod
	def cdf(x, mu, alpha, beta, lam):
		#Pr(mu<x)

		#scale location ????
		return stats.t.cdf((x-mu)*(lam*alpha/beta)**0.5, 2*alpha)


	def	q_sampling(self, state):
		mu_sample = []
		for (a, mu, b, l) in zip(self.alpha[state,:], self.mu0[state,:], self.beta[state,:], self.lam[state,:]):
			tau = np.random.gamma(a,1/b)
			mu_sample.append(np.random.normal(mu, 1/(l*tau)))
		return np.argmax(mu_sample)


	def VPI(self, state):
		a_star = np.argmax(self.mu0[state,:])
		mu_star = self.mu0[state,a_star]
		mu_second = sorted(self.mu0[state,:])[-2]
		c1 = self.alpha[state,:]*gamma_fun(self.alpha[state,:]+1/2)*np.sqrt(self.beta[state])
		c2 =  (
		(self.alpha[state,:]-1/2)*gamma_fun(self.alpha[state,:])*gamma_fun(1/2)*
		self.alpha[state,:]*np.sqrt(2*self.lam[state,:]))
		c3 = np.power(
		(1+self.mu0[state,:]**2)/(2*self.alpha[state,:]), -self.alpha[state,:]+1/2)

		#print(n[state],c1, "c1")
		#print(n[state],c2, "c2")
		#print(n[state],c3, "c3")
		c = self.alpha[state,:]*gamma_fun(self.alpha[state,:]+1/2)*np.sqrt(self.beta[state])/(
		(self.alpha[state,:]-1/2)*gamma_fun(self.alpha[state,:])*gamma_fun(1/2)*
		self.alpha[state,:]*np.sqrt(2*self.lam[state,:]))*np.power(
		(1+self.mu0[state,:]**2)/(2*self.alpha[state,:]), -self.alpha[state,:]+1/2)
		#print("c", c)
		vpi = []
		for action, (a, mu, b, l, c_i) in enumerate(zip(self.alpha[state,:], self.mu0[state,:], self.beta[state,:], self.lam[state,:], c )):
			if action == a_star:
				vpi.append(c_i + (mu_second-mu_star)*self.cdf(mu_second, mu, a, b, l))
			else:
				#print(mu-mu_star,1-cdf(mu_star, mu, a, b, l))
				#print(a,mu,b,l,c_i)
				vpi.append(c_i + (mu-mu_star)*(1-self.cdf(mu_star, mu, a, b, l)))
		#print("vpi",vpi)
		return vpi

	def posterior(self, state, action, M1, M2, update=True):
		#self.n[state,action] += 1
		mu0_new = (self.lam[state,action]*self.mu0[state,action]+
					self.n[state,action]*M1)/(self.lam[state,action]+self.n[state,action])
		lam_new = self.lam[state,action] + self.n[state,action]
		alpha_new = self.alpha[state,action] + self.n[state,action]/2
		beta_new = self.beta[state,action] + self.n[state,action]/2*(M2-M1**2) + (
		self.n[state,action]*self.lam[state,action]*(M1-self.mu0[state,action])**2/(2*(self.lam[state,action] + self.n[state,action]))
		)
		if update==True:
			self.mu0[state,action] = mu0_new
			self.lam[state,action] = lam_new*self.parameter_decay
			self.alpha[state,action] = alpha_new*self.parameter_decay
			self.beta[state,action] = beta_new*self.parameter_decay
		return mu0_new, lam_new, alpha_new, beta_new

	def normal_gamma_pdf(x, tau, mu, l, a, b):
		#mu, lambda, alpha, beta
		p = b**a*np.sqrt(l)/(gamma_fun(a)*np.sqrt(2*np.pi))*tau**(a-1/2)*(
		np.exp(-b*tau)*np.exp(-(l*tau*(x-mu)**2)/2))
		return p


	def update_mixed(self, state, action, reward, next_state):
		def integrand(x, mu, tau):
			#M1, M2 = moments(reward, next_state)
			next_action = np.argmax(self.mu0[next_state,:])
			E_r = self.mu0[next_state,next_action]
			E_r_sq = (self.lam[next_state, next_action]+1)/self.lam[next_state,next_action]*(
			self.beta[next_state,next_action]/(self.alpha[next_state,next_action]-1)) + self.mu0[next_state,next_action]**2
			p_R = lambda x: stats.norm.pdf(x, E_r,np.sqrt(E_r_sq-E_r**2) )
			return p_R(x)*prob(mu,tau,state,action, x, next_action, next_state)

		def prob(mu,tau, state,action,x, next_action, next_state):
			M1 =  reward + self.discount*x
			M2 = reward**2 + 2*gamma*reward*x+self.discount**2*x**2
			mu0_new, lam_new, alpha_new, beta_new = posterior(state,action,M1,M2, update=False)
			return normal_gamma_pdf(mu,tau,mu0_new,lam_new, alpha_new, beta_new)

		def monte_carlo(fun, limits, repeats=10000):
			s=0
			vals=[0,0,0]
			for i in range(repeats):
				for j in range(3):
					vals[j]=np.random.random()*(limits[j][1]-limits[j][0]) + limits[j][0]
				s+=fun(vals[0], vals[1], vals[2])
				print(vals)
				print(fun(vals[0], vals[1], vals[2]))
			return s/repeats

		#E_tau,_ = tplquad(lambda x,mu,tau: tau*integrand(x,mu,tau),0,25,lambda x: 0., lambda x: 1.,lambda x,y: 0.,lambda x,y: 1000., epsabs=0.01, epsrel=0.1)
		#E_tau_mu,_ = tplquad(lambda x,mu,tau: mu*tau*integrand(x,mu,tau),0,25,lambda x: 0., lambda x: 1.,lambda x,y: 0.,lambda x,y: 1000.,epsabs=0.01, epsrel=0.1)
		#E_tau_mu_sq,_ = tplquad(lambda x,mu,tau: np.power(mu,2)*tau*integrand(x,mu,tau),0,25,lambda x: 0., lambda x: 1.,lambda x,y: 0.,lambda x,y: 1000.,epsabs=0.01, epsrel=0.1)
		#E_log_tau,_ = tplquad(lambda x,mu,tau: np.log(tau)*integrand(x,mu,tau),0,25,lambda x: 0., lambda x: 1.,lambda x,y: 0.,lambda x,y: 1000.,epsabs=0.01, epsrel=0.1)

		limits = [[0,25], [0.,1000],[0,1000]]
		E_tau = monte_carlo(lambda x,mu,tau: tau*integrand(x,mu,tau),limits)
		E_tau_mu = monte_carlo(lambda x,mu,tau: mu*tau*integrand(x,mu,tau),limits)
		E_tau_mu_sq = monte_carlo(lambda x,mu,tau: np.power(mu,2)*tau*integrand(x,mu,tau),limits)
		E_log_tau = monte_carlo(lambda x,mu,tau: np.log(tau)*integrand(x,mu,tau),limits)

		print(E_tau, E_tau_mu, E_tau_mu_sq, E_log_tau)
		new_alpha = np.max(1+0.001, brentq(lambda y: np.log(y)-digamma(y)-(np.log(E_tau)-E_log_tau),0.0001, 10**8))
		new_mu0 = E_tau_mu/E_tau
		new_lam = 1/(E_tau_mu_sq-E_tau*new_mu0**2)
		new_beta = new_alpha/E_tau

		self.alpha[state,action]=new_alpha
		self.mu0[state,action] = new_mu0
		self.lam[state,action] = new_lam
		self.beta[state,action] = new_beta

	def moments(self, r, next_state):
		if next_state == None:
			E_r = 0
			E_r_sq = 0
		else:
			next_action = np.argmax(self.mu0[next_state,:])
			E_r = self.mu0[next_state,next_action]
			E_r_sq = (self.lam[next_state, next_action]+1)/self.lam[next_state,next_action]*(
			self.beta[next_state,next_action]/(self.alpha[next_state,next_action]-1)) + self.mu0[next_state,next_action]**2

		M1 = r + self.discount * E_r
		M2 = r**2 + 2*self.discount*r*E_r + self.discount**2*E_r_sq
		return M1, M2

	def play(self, state):
		"""
		Returns the action to play at state

		Args:
			state the state
		Returns:
			The action to play
		"""

		# if state is None:
		# 	return 0
		if self.action_selection == "vpi":
			action = np.argmax(self.VPI(state)+self.mu0[state,:])
		elif self.action_selection == "q-sampling":
			action = self.q_sampling(state)
		else:
			raise ValueError("Incorrect method for selecting actions")
		return action

	def observe_transition(self, state, action, next_state, reward):
		"""
		Observe a new transition: state,action,next_state,reward
		This means at state and upon playing action you transition to
		next_state and obtains reward

		"""
		if self.update_method == "mom":
			M1,M2 = self.moments(reward, next_state)
			_, _,_,_ = self.posterior(state, action, M1, M2, update=True)

		elif self.update_method == "mixed":
			self.update_mixed(state, action, reward, next_state)

		else:
			raise ValueError("Incorrect method for update")

		#print("max", alpha.max(), beta.max(), lam.max(), mu0.max())
		#print("min",alpha.min(), beta.min(), lam.min(), mu0.min())

	#scipy.special.digamma(z)
	#return mu0
