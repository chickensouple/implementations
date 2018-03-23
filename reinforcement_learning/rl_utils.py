"""
Utilities for reinforcement learning algorithms
"""
import time
import numpy as np
import scipy.signal
import copy
import math

def rollout(env, policy, get_state, max_iter=10000, render=False):
    """
    Simulates one episode of the environment following a policy
    
    Args:
        env (TYPE): openai gym env
        policy (function): function that takes in state and returns an action
        get_state (TYPE): function to get state from observation
        max_iter (int, optional): maximum number of iterations to simulate
        render (bool, optional): True if you want to render the environment
    
    Returns:
        TYPE: Description
    """
    obs = env.reset()
    rewards = []
    actions = []
    states = []
    for _ in range(max_iter):
        state = get_state(obs)
        action = policy(state)

        actions.append(action)
        states.append(state)

        if (render):
            env.render()
            time.sleep(0.01)

        [obs, reward, done, info] = env.step(action)
        # [obs, reward, done, info] = env.step(np.array([action]))
        rewards.append(reward)

        if done:
            break
    return [states, actions, rewards]

def single_test(env, method, get_state, num_episodes):
    for i in range(num_episodes):
        total_reward = method.update(env, get_state)
        print("Iteration " + str(i) + " reward: " + str(total_reward))
        if i % 500 == 0:
            [_, _, rewards] = rollout(env, method.curr_policy(), get_state, render=True)
            total_reward = np.sum(np.array(rewards))
            print("Test reward: " + str(total_reward))

    policy = method.curr_policy()
    rollout(env, policy, get_state, render=True)

class LinearAnnealing(object):
    def __init__(self, start_eps, end_eps, num_steps):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.num_steps = num_steps

        self.reset()

    def reset(self):
        self.eps = self.start_eps

    def update(self):
        if self.eps >= self.end_eps:
            self.eps -= (self.start_eps - self.end_eps) / self.num_steps


def TabularSoftmaxPolicy(theta, state, temperature=1):
    row = copy.deepcopy(theta[state, :])
    row = row / temperature
    probs = np.exp(row - np.max(row))
    probs /= np.sum(probs)

    cum_prob = np.cumsum(probs)

    sample = np.random.random_sample()
    num_actions = theta.shape[1]
    idx = 0
    while sample > cum_prob[idx]:
        idx += 1
        if idx >= num_actions:
            idx = num_actions-1
            break
    return idx

def TabularGreedyPolicy(Q, state):
    max_action = np.argmax(Q[state, :])
    return max_action

def TabularEpsilonGreedyPolicy(Q, eps, state):
    sample = np.random.random_sample()

    num_actions = Q.shape[1]

    if sample > eps:
        max_val = Q[state, :].max()

        max_indices = np.where(np.abs(Q[state, :] - max_val) < 1e-5)[0]
        rand_idx = np.random.randint(len(max_indices))
        max_action = max_indices[rand_idx]

        # max_action = np.argmax(Q[state, :])
        return max_action
    else:
        return np.random.randint(num_actions)

class RunningAverage(object):
    def __init__(self, N):
        self.N = N
        self.vals = []
        self.num_filled = 0

    def push(self, val):
        if self.num_filled == self.N:
            self.vals.pop(0)
            self.vals.append(val)
        else:
            self.vals.append(val)
            self.num_filled += 1

    def get(self):
        return float(sum(self.vals)) / self.num_filled

def discretize_val(val, min_val, max_val, num_states):
    """
    Discretizes a single float
    if val < min_val, it gets a discrete value of 0
    if val >= max_val, it gets a discrete value of num_states-1
    
    Args:
        val (float): value to discretize
        min_val (float): lower bound of discretization
        max_val (float): upper bound of discretization
        num_states (int): number of discrete states
    
    Returns:
        float: discrete value
    """
    state = int(num_states * (val - min_val) / (max_val - min_val))
    if state >= num_states:
        state = num_states - 1
    if state < 0:
        state = 0
    return state

def obs_to_state(num_states, lower_bounds, upper_bounds, obs):
    """
    Turns an observation in R^N, into a discrete state
    
    Args:
        num_states (list): list of number of states for each dimension of observation
        lower_bounds (list): list of lowerbounds for discretization
        upper_bounds (list): list of upperbounds for discretization
        obs (list): observation in R^N to discretize
    
    Returns:
        int: discrete state
    """
    state_idx = []
    for ob, lower, upper, num in zip(obs, lower_bounds, upper_bounds, num_states):
        state_idx.append(discretize_val(ob, lower, upper, num))

    return np.ravel_multi_index(state_idx, num_states)


def acrobot_obs_to_state(num_states, upper_bounds, lower_bounds, obs):
    state_idx = []

    angle1 = math.atan2(obs[1], obs[0])
    angle2 = math.atan2(obs[3], obs[2])

    angle_idx1 = int((angle1 + math.pi) * num_states[0] / (math.pi * 2))
    angle_idx2 = int((angle2 + math.pi) * num_states[1] / (math.pi * 2))
    state_idx.append(angle_idx1)
    state_idx.append(angle_idx2)
    state_idx.append(discretize_val(obs[4], lower_bounds[0], upper_bounds[0], num_states[2]))
    state_idx.append(discretize_val(obs[5], lower_bounds[1], upper_bounds[1], num_states[3]))

    return np.ravel_multi_index(state_idx, num_states)


def get_discounted_rewards(rewards, gamma):
    """
    Gets Discounted rewards at every timestep
    
    Args:
        rewards (numpy array): a list of [r_1, r_2, ...]
        where r_i = r(x_i, u_i)
        gamma (float): discount factor
    
    Returns:
        numpy array: a list of [R_1, R_2, ...]
        where R_i = \sum_{n=i} r_n * gamma^{n-i}
    """
    return scipy.signal.lfilter([1],[1,-gamma], rewards[::-1], axis=0)[::-1]

