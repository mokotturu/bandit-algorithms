import numpy as np
import matplotlib.pyplot as plt
from bandit import StochasticBandit
from algorithms import Exp3_TB, Exp3_OG, LinUCB
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
	exp3 = Exp3_OG(k, gamma=0.01)

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

def run_linucb():
	N = 10
	d = 100
	alpha = 1.0
	data_path = 'data/dataset.txt'

	cumulative_reward = 0
	aligned_timesteps = 0
	aligned_ctr = np.array([])

	linucb = LinUCB(N, d, alpha=alpha)
	
	print(f'{ctime(time())}: Script started')

	with open(data_path, 'r') as f:
		for line in f:
			# break the line into a list of integers
			data = np.array(list(map(int, line.strip().split(' '))))

			# first element is the logged arm
			logged_arm = data[0]

			# second element is the logged reward for choosing that arm
			logged_reward = data[1]

			# rest of the array are covariates
			covariates = data[2:]

			chosen_arm = linucb.choose_arm(covariates)

			if chosen_arm + 1 == logged_arm:
				linucb.update(chosen_arm, logged_reward, covariates)

				aligned_timesteps += 1
				cumulative_reward += logged_reward
				aligned_ctr = np.append(aligned_ctr, cumulative_reward / aligned_timesteps)
	
	print(f'Aligned timesteps: {aligned_timesteps}, cumulative reward: {cumulative_reward}')
	print(f'{ctime(time())}: Script ended')

	plt.plot(aligned_ctr)
	plt.savefig('linucb.png', format='png')
	plt.show()



if __name__ == '__main__':
	# main()
	run_linucb()
