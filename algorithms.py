import numpy as np

class Exp3_TB:
	def __init__(self, k: int, eta=0.1) -> None:
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

	def choose_arm(self) -> int:
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

	def update(self, i: int, X_t: float) -> None:
		"""Update the distributions.

		## Parameters
		i: int
			The index of the arm played.
		X_t: float
			The reward obtained by playing the arm."""

		A_t = np.zeros(self.k)
		A_t[i] = 1
		self.S_t += 1 - (A_t * (1 - X_t)) / self.P_t

	def reset(self) -> None:
		"""Reset the distributions."""
		self.P_t = np.zeros(self.k)
		self.S_t = np.zeros(self.k)

class Exp3_OG:
	def __init__(self, k: int, gamma=0.1) -> None:
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

	def choose_arm(self) -> int:
		"""Choose an arm to play.

		## Returns
		arm_idx: float
			The index of the selected arm."""

		x = self.w_t - np.max(self.w_t)
		self.p_t = (1 - self.gamma) * (x / np.sum(x)) + (self.gamma / self.k)
		return np.random.choice(self.k, p=self.p_t)

	def update(self, j: int, x_t: float) -> None:
		"""Update the distributions.

		## Parameters
		j: int
			The index of the arm played.
		x_t: float
			The reward obtained by playing the arm."""

		xhat_t = np.zeros(self.k)
		xhat_t[j] = x_t / self.p_t[j]
		self.w_t *= np.exp(self.gamma * xhat_t / self.k)

	def reset(self) -> None:
		"""Reset the distributions."""

		self.p_t = np.zeros(self.k)
		self.S_t = np.zeros(self.k)

class LinUCB:
	def __init__(self, k: int, d: int, alpha=0.1) -> None:
		"""LinUCB algorithm

		## Parameters
		k: int
			The number of arms.
		d: int
			The number of features.
		alpha: float
			The learning rate."""

		self.k = k
		self.d = d
		self.alpha = alpha

		# initialize A (identity matrix) for each arm
		self.A = np.tile(np.eye(d, dtype=int), (k, 1, 1))

		# initialize b (zero vector) for each arm
		self.b = np.zeros((k, d))

	def choose_arm(self, x_t: np.ndarray) -> int:
		"""Choose an arm to play.

		## Parameters
		x_t: np.ndarray
			Covariates.

		## Returns
		arm_idx: int
			The index of the selected arm."""

		# apply ridge regression to get estimated coefficients "theta"
		thetas = np.array([np.dot(np.linalg.inv(self.A[i]), self.b[i]) for i in range(self.k)])

		# compute the upper confidence bound for each arm
		p_t = np.zeros(self.k)
		for i in range(self.k):
			p_t = np.dot(thetas[i].T, x_t) + self.alpha * np.sqrt(x_t.T @ np.linalg.inv(self.A) @ x_t)

		# choose the arm with the highest upper confidence bound using argmax and tie breaking
		return np.random.choice(np.flatnonzero(np.isclose(p_t, p_t.max())))

	def update(self, i: int, reward: float, x_t: np.ndarray) -> None:
		"""Update A and b for each arm.

		## Parameters
		i: int
			The index of the arm played.
		reward: float
			The reward received from choosing arm i.
		x_t: np.ndarray
			Covariates."""

		self.A[i] += x_t @ x_t.T
		self.b[i] += reward * x_t
