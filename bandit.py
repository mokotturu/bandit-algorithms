import numpy as np

class StochasticBandit:
	def __init__(self, means: np.ndarray, sds: np.ndarray):
		"""A stochastic bandit with k arms.

		## Parameters
		means: np.ndarray
			The means of the arms.
		sds: np.ndarray
			The standard deviations of the arms."""
		if means.size != sds.size:
			raise ValueError('Means and standard deviations must have the same size')
		self.k = means.size
		self.means = means
		self.sds = sds

	def play(self, i: int) -> tuple:
		"""Play the arm with index i.

		## Parameters
		i: int
			The index of the arm to play.
		
		## Returns
		reward: float
			The reward obtained by playing the arm.
		regret: float
			The difference between the highest true mean and the true mean of the selected arm."""
		if i < 0 or i >= self.k:
			raise ValueError(f'Invalid arm index: {i}')
		return np.random.normal(self.means[i], self.sds[i]), np.max(self.means) - self.means[i]
