import numpy as np
from functools import partial
import tensorflow as tf
from replay_buffer import ReplayBuffer
import pdb

class DQN(object):
    def __init__(self, state_dim, num_actions, eps_anneal, gamma=0.99, update_freq=100, sess=None):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.eps_anneal = eps_anneal
        self.update_freq = update_freq

        self.batch_size = 64

        self.replay_buffer = ReplayBuffer(3000, state_dim=state_dim, action_dim=1)
        self.__build_model()

        if sess == None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess = sess

        self.reset_policy()

        writer = tf.summary.FileWriter('logs', self.sess.graph)
        writer.close()

    def reset_policy(self):
        tf.global_variables_initializer().run()
        self.train_idx = 0
        self.replay_buffer.clear()
        self.eps_anneal.reset()

    def __build_q_func(self, input_var, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse) as scope:
            layer1 = tf.contrib.layers.fully_connected(input_var, 
                32, 
                activation_fn=tf.nn.relu, 
                scope='layer1')
            layer2 = tf.contrib.layers.fully_connected(layer1, 
                16, 
                activation_fn=tf.nn.relu, 
                scope='layer2')
            q_vals = tf.contrib.layers.fully_connected(layer2, 
                self.num_actions, 
                activation_fn=None, 
                scope='q_vals')
        return q_vals

    def __build_model(self):
        # forward model
        self.states = tf.placeholder(tf.float32, [None, self.state_dim], name='states')
        self.actions = tf.placeholder(tf.int32, [None], name='actions')
        self.action_q_vals = self.__build_q_func(self.states, name='action_q_func')
        self.output_actions = tf.argmax(self.action_q_vals, axis=1, name='output_actions')
        self.sampled_q_vals = tf.reduce_sum(tf.multiply(self.action_q_vals, tf.one_hot(self.actions, self.num_actions)), 1, name='sampled_q_vals')
        

        self.target_q_vals = self.__build_q_func(self.states, name='target_q_func')
        self.max_q_vals = tf.reduce_max(self.target_q_vals, axis=1, name='max_q_vals')

        # loss
        self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
        self.terminal = tf.placeholder(tf.float32, [None], name='terminal')
        self.q_vals_next_state = tf.placeholder(tf.float32, [None], name='q_vals_next_state')

        self.terminal_mask = tf.subtract(1.0, self.terminal)

        self.disc_return = tf.add(self.rewards, 
            tf.multiply(self.terminal_mask,
                tf.multiply(self.gamma, self.q_vals_next_state)), 
            name='disc_return')

        self.td_error = tf.subtract(self.disc_return, self.sampled_q_vals, name='td_error')
        self.loss = tf.reduce_mean(tf.square(self.td_error), name='loss')
        self.optimizer = tf.train.RMSPropOptimizer(0.00025).minimize(self.loss)

        # updating target network
        var_sort_lambd = lambda x: x.name
        self.action_q_vars = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='action_q_func'), key=var_sort_lambd)
        self.target_q_vars = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_q_func'), key=var_sort_lambd)

        update_target_ops = []
        for action_q, target_q in zip(self.action_q_vars, self.target_q_vars):
            update_target_ops.append(target_q.assign(action_q))
        self.update_target_ops = tf.group(*update_target_ops, name='update_target_ops')

    def __update_target_network(self):
        self.sess.run(self.update_target_ops)

    def get_action(self, state):
        sample = np.random.random_sample()
        if sample > self.eps_anneal.eps:
            fd = {self.states: np.array([state])}
            output_action = self.sess.run(self.output_actions, feed_dict=fd)
            action = np.asscalar(output_action)
        else:
            action = np.random.randint(self.num_actions)

        return action

    def curr_policy(self):
        return partial(DQN.get_action, self)

    def save_model(self, filename='/tmp/model.ckpt'):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, filename)
        print("Model saved in file: %s" % filename)

    def load_model(self, filename='/tmp/model.ckpt'):
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)
        print("Model loaded from file: %s" % filename)

    def update(self, env, get_state, max_iter=1000):
        state = env.reset()

        action = self.get_action(state)

        total_reward = 0
        for i in range(max_iter):
            [new_state, reward, done, _] = env.step(action)
            total_reward += reward

            self.replay_buffer.insert(state, action, reward, new_state, done)

            state = new_state

            if self.train_idx >= self.batch_size:
                sample = self.replay_buffer.sample(self.batch_size)

                # get max q values of next state
                fd = {self.states: sample['next_state']}
                max_q_vals = self.sess.run(self.max_q_vals, feed_dict=fd)

                fd = {self.states: sample['state'],
                    self.actions: sample['action'].squeeze(),
                    self.rewards: sample['reward'],
                    self.terminal: sample['terminal'],
                    self.q_vals_next_state: max_q_vals}

                loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=fd)


                if self.train_idx % self.update_freq == 0:
                    self.__update_target_network()
            if done:
                break

            action = self.get_action(state)
            self.train_idx += 1

        self.eps_anneal.update()
        return total_reward



if __name__ == '__main__':
    from rl_utils import rollout, LinearAnnealing
    import gym
    import argparse
    import sys
    import pickle

    parser = argparse.ArgumentParser(description="DQN")
    parser.add_argument('--type', dest='type', action='store',
        required=True,
        choices=['train', 'test'],
        help="type")
    args = parser.parse_args(sys.argv[1:])


    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    eps_anneal = LinearAnnealing(1, 0.005, 1000)
    dqn = DQN(state_dim, env.action_space.n, eps_anneal)

    if args.type == 'train':
        get_state = lambda x: x
        for i in range(10000):
            total_reward = dqn.update(env, get_state)
            print("Iteration " + str(i) + " reward: " + str(total_reward))
            if i % 20 == 0:
                [_, _, rewards] = rollout(env, dqn.curr_policy(), get_state, render=True)
                total_reward = np.sum(np.array(rewards))
                print("Test reward: " + str(total_reward))

            if i % 100 == 0:
                dqn.save_model()

        policy = dqn.curr_policy()
        rollout(env, policy, get_state, render=True)
        dqn.save_model()
    elif args.type == 'test':
        dqn.load_model()
        get_state = lambda x: x
        for i in range(20):
            [_, _, rewards] = rollout(env, dqn.curr_policy(), get_state, render=True)
            total_reward = np.sum(np.array(rewards))
            print("Test reward: " + str(total_reward))

