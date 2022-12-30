import numpy as np

class Exp3:
	def __init__(self, k: int, eta=0.1):
		"""Exp3 algorithm from the [Bandit Algorithms book](https://tor-lattimore.com/downloads/book/book.pdf).

		## Parameters
		k: int
			The number of arms.
		eta: float
			The learning rate."""
		self.k = k
		self.eta = eta
		self.P_t = np.zeros(k)
		self.S_t = np.zeros(k)

	def choose_arm(self):
		"""Choose an arm to play.

		## Returns
		reward: float
			The reward obtained by playing the chosen arm."""
		x = self.eta * self.S_t
		z = x - np.max(x)
		numerator = np.exp(z)
		denominator = np.sum(numerator)
		self.P_t = numerator / denominator
		# print('P_t')
		# print(self.P_t)
		# print(np.sum(self.P_t))
		# print('S_t')
		# print(self.S_t)
		# print(np.sum(self.S_t))
		return np.random.choice(self.k, p=self.P_t)

	def update(self, i: int, X_t: float):
		"""Update the distributions.
		
		## Parameters
		i: int
			The index of the arm played.
		X_t: float
			The reward obtained by playing the arm."""
		A_t = np.zeros(self.k)
		A_t[i] = 1
		self.S_t += 1 - (A_t * (1 - X_t)) / self.P_t

	def reset(self):
		"""Reset the distributions."""
		self.P_t = np.zeros(self.k)
		self.S_t = np.zeros(self.k)
