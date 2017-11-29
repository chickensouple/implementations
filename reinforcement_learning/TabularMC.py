import numpy as np
from functools import partial
from rl_utils import *
import gym

class TabularMC(object):
	def __init__(self, num_states, num_actions, gamma=1, eps=0.1, eps_decay=0.9999, first_visit=True):
		self.num_states = num_states
		self.num_actions = num_actions
		self.first_visit = first_visit
		
		self.gamma = gamma
		self.start_eps = eps
		self.eps_decay = eps_decay

		self.reset_policy()

	def reset_policy(self):
		self.Q = np.zeros((self.num_states, self.num_actions))
		self.Q_num = np.zeros((self.num_states, self.num_actions))
		self.visited_num = np.zeros((self.num_states, self.num_actions))
		self.eps = self.start_eps

	def curr_policy(self, copy=False):
		if copy:
			return partial(TabularEpsilonGreedyPolicy, np.copy(self.Q), self.eps)
		else:
			return partial(TabularEpsilonGreedyPolicy, self.Q, self.eps)

	def update(self, env, get_state):
		policy = self.curr_policy()
		[states, actions, rewards] = rollout(env, policy, get_state)

		updated_states = set()

		discounted_rewards = get_discounted_rewards(np.array(rewards), self.gamma)
		for state, action, disc_rew in zip(states, actions, discounted_rewards):

			if self.first_visit:
				if state in updated_states:
					continue

				updated_states.add(state)

			# incremental averaging
			self.Q_num[state, action] += 1
			self.Q[state, action] += (disc_rew - self.Q[state, action]) / self.Q_num[state, action]
			# self.Q[state, action] += 0.1 * (disc_rew - self.Q[state, action])
			self.visited_num[state, action] += 1
			

		self.eps *= self.eps_decay

		return np.sum(np.array(rewards))


if __name__ == '__main__':

	env = gym.make('CartPole-v0')
	num_actions = env.action_space.n
	num_states = [1, 8, 8, 8]
	
	state_len = np.prod(np.array(num_states))
	lower_bounds = [-4.8, -3, -0.418, -2]
	upper_bounds = [4.8, 3, 0.418, 2]

	get_state = partial(obs_to_state, num_states, lower_bounds, upper_bounds)

	mc1 = TabularMC(state_len, num_actions, first_visit=True, eps_decay=0.99)
	mc2 = TabularMC(state_len, num_actions, first_visit=False, eps_decay=0.99)

	single_test(env, mc2, get_state, 20000)
