import numpy as np
from functools import partial
from rl_utils import *
import gym

class TabularPG(object):
	def __init__(self, num_states, num_actions, gamma=1, alpha=0.5, alpha_decay=0.9995):
		self.num_states = num_states
		self.num_actions = num_actions

		self.gamma = gamma
		self.start_alpha = alpha # learning rate
		self.alpha_decay = alpha_decay
		self.reset_policy()

	def reset_policy(self):
		self.theta = np.zeros((self.num_states, self.num_actions)) # action values
		self.alpha = self.start_alpha

	def curr_policy(self, copy=False):
		if copy:
			return partial(TabularSoftmaxPolicy, np.copy(self.theta))
		else:
			return partial(TabularSoftmaxPolicy, self.theta)

	def update(self, env, get_state):
		policy = self.curr_policy()
		[states, actions, rewards] = rollout(env, policy, get_state)

		discounted_rewards = get_discounted_rewards(np.array(rewards), self.gamma)

		# get baseline
		baseline = np.zeros(self.theta.shape[0])
		baseline_counts = np.zeros(self.theta.shape[0])
		for state, disc_rew in zip(states, discounted_rewards):
			baseline_counts[state] += 1
			baseline[state] += (disc_rew - baseline[state]) / baseline_counts[state]

		grad = np.zeros(self.theta.shape)
		for state, action, disc_rew in zip(states, actions, discounted_rewards):
			row = self.theta[state, :]
			exponents = np.exp(row)
			den = np.sum(exponents)

			grad_row = -exponents / den
			grad_row[action] += 1
			grad[state, :] += grad_row * (disc_rew - baseline[state])

		self.theta += grad * self.alpha
		self.alpha *= self.alpha_decay

		return np.sum(np.array(rewards))


if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	num_actions = env.action_space.n
	num_states = [1, 8, 8, 8]
	
	state_len = np.prod(np.array(num_states))
	lower_bounds = [-4.8, -3, -0.418, -2]
	upper_bounds = [4.8, 3, 0.418, 2]

	get_state = partial(obs_to_state, num_states, lower_bounds, upper_bounds)

	policy_grad = TabularPG(state_len, num_actions, alpha=0.1)

	single_test(env, policy_grad, get_state, 20000)

