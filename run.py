import numpy as np
import matplotlib.pyplot as plt
from bandit import StochasticBandit
from exp3 import Exp3TB, Exp3OG
from time import ctime, time

def main():
	# Run the algorithm for T iterations
	T = 1000
	runs = 100

	print(f'{ctime(time())}: Script started with {T} timesteps and {runs} runs')

	# Create a bandit with k arms
	k = 10
	means = np.random.normal(0, 1, k)
	sds = np.ones(k)
	bandit = StochasticBandit(means, sds)

	# Run Exp3
	# eta = np.sqrt(2 * np.log(k) / (T * k))
	# print(f'eta = {eta}')
	exp3 = Exp3OG(k, gamma=0.01)

	# Store the rewards
	rewards = np.zeros((runs, T))
	regrets = np.zeros((runs, T))

	for run in range(runs):
		exp3.reset()
		for t in range(T):
			# Choose an arm
			i = exp3.choose_arm()

			# Play the arm
			rewards[run, t], regrets[run, t] = bandit.play(i)

			# Update the distributions
			exp3.update(i, rewards[run, t])

	regrets = np.cumsum(np.mean(regrets, axis=0))
	plt.plot(regrets)
	print(f'{ctime(time())}: Script ended')
	plt.show()


if __name__ == '__main__':
	main()
