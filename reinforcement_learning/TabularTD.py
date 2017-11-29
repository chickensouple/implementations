import numpy as np
from functools import partial
from rl_utils import *
import gym

# tabular temporal difference methods with eligibility traces
# 


class TabularTD(object):
	def __init__(self, num_states, num_actions, lambd=0, qlearning=True, replaytrace=True, online=False, gamma=1, alpha=0.5, eps=0.1, eps_decay=0.9999):
		self.num_states = num_states
		self.num_actions = num_actions
		self.lambd = lambd
		self.replaytrace = replaytrace
		self.qlearning = qlearning
		self.online = online

		self.alpha = alpha
		self.gamma = gamma 
		self.start_eps = eps
		self.eps_decay = eps_decay

		self.reset_policy()

	def reset_policy(self):
		self.Q = np.zeros((self.num_states, self.num_actions)) # action values
		self.visited_num = np.zeros((self.num_states, self.num_actions))
		self.eps = self.start_eps

	def curr_policy(self, copy=False):
		if copy:
			return partial(TabularEpsilonGreedyPolicy, np.copy(self.Q), self.eps)
		else:
			# return partial(TabularGreedyPolicy, self.Q)
			return partial(TabularEpsilonGreedyPolicy, self.Q, self.eps)

	def update(self, env, get_state):
		max_iter = 1000
		obs = env.reset()
		env._max_episode_steps = 1000
		# print dir(env)
		# exit()

		elig_trace = np.zeros((self.num_states, self.num_actions))

		policy = self.curr_policy()

		state = get_state(obs)
		action = policy(state)

		states = []
		actions = []

		total_reward = 0
		if (not self.online):
			total_delta_Q = 0

		for i in range(max_iter):
			self.visited_num[state, action] += 1

			[obs, reward, done, info] = env.step(action)
			total_reward += reward

			states.append(state)
			actions.append(action)

			next_state = get_state(obs)
			next_action = policy(next_state)


			if done:
				td_error = reward - self.Q[state, action]
			else:
				if self.qlearning:
					td_error = reward + (self.gamma * np.max(self.Q[next_state, :])) - self.Q[state, action]
				else:
					td_error = reward + (self.gamma * self.Q[next_state, next_action]) - self.Q[state, action]
			


			if self.replaytrace:
				elig_trace[state, action] = 1
			else:
				elig_trace[state, action] += 1

			delta_Q = elig_trace * self.alpha * td_error

			if self.online:
				self.Q += delta_Q
			else:
				total_delta_Q += delta_Q

			if self.qlearning and next_action != np.argmax(self.Q[next_state, :]):
				elig_trace = np.zeros((self.num_states, self.num_actions))
			else:
				elig_trace *= (self.gamma * self.lambd)

			# print "td_error: ", td_error
			# print "max Q: ", np.max(np.abs(self.Q))

			state = next_state
			action = next_action

			if done:
				break

		if (not self.online):
			self.Q += total_delta_Q

		self.eps *= self.eps_decay

		return total_reward

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	num_actions = env.action_space.n
	num_states = [1, 8, 8, 8]
	
	state_len = np.prod(np.array(num_states))
	lower_bounds = [-4.8, -3, -0.418, -2]
	upper_bounds = [4.8, 3, 0.418, 2]

	get_state = partial(obs_to_state, num_states, lower_bounds, upper_bounds)

	sarsa = TabularTD(state_len, num_actions, alpha=0.5, lambd=0, replaytrace=False, qlearning=False, online=True)
	qlearning = TabularTD(state_len, num_actions, alpha=0.5, lambd=0, replaytrace=False, qlearning=True, online=True)
	qlearning_replay = TabularTD(state_len, num_actions, alpha=0.5, lambd=0, replaytrace=True, qlearning=True, online=True, eps_decay=0.99)
	tdlambda = TabularTD(state_len, num_actions, alpha=0.5, lambd=0.8, replaytrace=True, qlearning=True, online=True, eps_decay=0.99)
	

	single_test(env, tdlambda, get_state, 20000)


