import numpy as np

class Exp3TB:
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
		arm_idx: float
			The index of the selected arm."""
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

class Exp3OG:
	def __init__(self, k: int, gamma=0.1):
		"""Exp3 algorithm from the [Auer et al.](https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf).

		## Parameters
		k: int
			The number of arms.
		gamma: float
			The learning rate."""
		self.k = k
		self.gamma = gamma
		self.p_t = np.zeros(k)
		self.w_t = np.zeros(k)

	def choose_arm(self):
		"""Choose an arm to play.

		## Returns
		arm_idx: float
			The index of the selected arm."""
		x = self.w_t - np.max(self.w_t)
		self.p_t = (1 - self.gamma) * (x / np.sum(x)) + (self.gamma / self.k)
		return np.random.choice(self.k, p=self.p_t)

	def update(self, j: int, x_t: float):
		"""Update the distributions.
		
		## Parameters
		i: int
			The index of the arm played.
		x_t: float
			The reward obtained by playing the arm."""
		xhat_t = np.zeros(self.k)
		xhat_t[j] = x_t / self.p_t[j]
		self.w_t *= np.exp(self.gamma * xhat_t / self.k)

	def reset(self):
		"""Reset the distributions."""
		self.p_t = np.zeros(self.k)
		self.S_t = np.zeros(self.k)
