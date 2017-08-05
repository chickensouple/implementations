import numpy as np
import tensorflow as tf
from functools import partial
from tensorflow_utils import fully_connected
from replay_buffer import ReplayBuffer
import pdb

class OrnsteinUhlenbeckProcess(object):
    def __init__(self, dt, theta, sigma, mu, x0):
        self.dt = dt
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.x0 = x0
        self.reset()

    def get_next(self):
        dW = np.random.randn(*self.x.shape) * np.sqrt(self.dt)
        self.x = self.theta * (self.mu - self.x) * self.dt + self.sigma * dW
        return self.x

    def reset(self):
        self.x = self.x0


def create_update_op(network_vars, target_network_vars, tau, name):
    var_sort_lambd = lambda x: x.name
    network_vars = sorted(network_vars, key=var_sort_lambd)
    target_network_vars = sorted(target_network_vars, key=var_sort_lambd)

    update_target_ops = []
    for var, target_var in zip(network_vars, target_network_vars):
        updated_var = tau * var + (1.0 - tau) * target_var
        update_target_ops.append(target_var.assign(updated_var))
    update_target_ops = tf.group(*update_target_ops, name=name)
    return update_target_ops


class ActorModel(object):
    def __init__(self, state_dim, action_dim, action_mean, action_scale, sess):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_mean = action_mean
        self.action_scale = action_scale
        self.sess = sess


        # create network and target network
        self.state_inputs, self.actions_out = \
            self._create_model('actor', state_dim, action_dim, action_mean, action_scale)
        self.target_state_inputs, self.target_actions_out = \
            self._create_model('target_actor', state_dim, action_dim, action_mean, action_scale)

        # get variables for network and target networks
        self.network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        self.target_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')

        tau = 0.001
        self.update_target_ops = create_update_op(self.network_vars, 
            self.target_network_vars, tau, name='actor_update_target')

        self.reset_q_ops = create_update_op(self.network_vars, self.target_network_vars, 1.0, name='actor_reset_target')

        # loss
        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_dim], name='action_grads')
        self.grads = tf.gradients(self.actions_out, self.network_vars, -self.action_gradients)
        self.optimize = tf.train.AdamOptimizer(1e-4).apply_gradients(zip(self.grads, self.network_vars))

    def reset_target_model(self):
        self.sess.run(self.reset_q_ops)

    def get_action(self, states):
        fd = {self.state_inputs: np.reshape(states, (-1, self.state_dim))}
        actions = self.sess.run(self.actions_out, feed_dict=fd)
        return actions

    def get_target_action(self, states):
        fd = {self.target_state_inputs: np.reshape(states, (-1, self.state_dim))}
        actions = self.sess.run(self.target_actions_out, feed_dict=fd)
        return actions

    def train(self, states, action_grads):
        fd = {self.state_inputs: states, self.action_gradients: action_grads}
        self.sess.run(self.optimize, feed_dict=fd)

    def update_target_model(self):
        self.sess.run(self.update_target_ops)

    def _create_model(self, name, state_dim, action_dim, action_mean, action_scale):
        with tf.variable_scope(name) as scope:
            state_inputs = tf.placeholder(tf.float32, [None, state_dim], name='state')

            xavier_init = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope('layer1'):
                layer1 = fully_connected(state_inputs, 400, activation=tf.nn.relu, var_init=xavier_init)
            with tf.variable_scope('layer2'):
                layer2 = fully_connected(layer1, 300, activation=tf.nn.relu, var_init=xavier_init)
            with tf.variable_scope('actions'):
                rand_init = partial(tf.random_uniform, minval=-3e-3, maxval=3e-3)
                actions_out = fully_connected(layer2, action_dim, activation=tf.nn.tanh, var_init=rand_init)
                actions_out = tf.add(tf.multiply(actions_out, action_scale), action_mean)
        return state_inputs, actions_out

class CriticModel(object):
    def __init__(self, state_dim, action_dim, sess):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sess = sess

        # create network and target network
        self.state_inputs, self.action_inputs, self.q_vals_out, self.layer1 = \
            self._create_model('critic1', state_dim, action_dim)
        self.target_state_inputs, self.target_action_inputs, self.target_q_vals_out, _ = \
            self._create_model('target_critic', state_dim, action_dim)

        # get variables for network and target networks
        self.network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        self.target_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')

        # loss
        self.disc_return = tf.placeholder(tf.float32, [None], name='disc_return')
        self.td_error = self.disc_return - tf.squeeze(self.q_vals_out)
        self.loss = tf.reduce_mean(tf.square(self.td_error), name='loss')

        self.l2_reg = tf.add_n([tf.nn.l2_loss(var) for var in self.network_vars])

        self.total_loss = tf.add(self.loss, 0.00*self.l2_reg, name='total_loss')

        self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.total_loss)

        # action gradients for use by actor
        self.action_gradients = tf.gradients(self.q_vals_out, self.action_inputs)

        tau = 0.001
        self.update_target_ops = create_update_op(self.network_vars, 
            self.target_network_vars, tau, name='critic_update_target')

        self.reset_q_ops = create_update_op(self.network_vars, self.target_network_vars, 1.0, name='critic_reset_target')


    def reset_target_model(self):
        """
        Sets all weights in target network to match normal network
        """
        self.sess.run(self.reset_q_ops)

    def get_target_q_val(self, states, actions):
        """
        Gets q values from target network
        
        Args:
            states (TYPE): Description
            actions (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        fd = {self.target_state_inputs: states, self.target_action_inputs: actions}
        q_vals = self.sess.run(self.target_q_vals_out, feed_dict=fd)
        return q_vals

    def train(self, states, actions, disc_return):
        fd = {self.state_inputs: states, self.action_inputs: actions, self.disc_return: disc_return}
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=fd)
        return loss

    def get_action_grads(self, states, actions):
        fd = {self.state_inputs: states, self.action_inputs: actions}
        action_grads = self.sess.run(self.action_gradients, feed_dict=fd)
        return action_grads

    def update_target_model(self):
        self.sess.run(self.update_target_ops)

    def _create_model(self, name, state_dim, action_dim):
        with tf.variable_scope(name) as scope:
            state_inputs = tf.placeholder(tf.float32, [None, state_dim], name='state')
            action_inputs = tf.placeholder(tf.float32, [None, action_dim], name='action')

            xavier_init = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("layer1_state"):
                layer1_state = fully_connected(state_inputs, 400, activation=tf.nn.relu, var_init=xavier_init)
            with tf.variable_scope("layer1"):
                layer1 = tf.concat([layer1_state, action_inputs], axis=1)
            with tf.variable_scope("layer2"):
                layer2 = fully_connected(layer1, 300, activation=tf.nn.relu, var_init=xavier_init)
            with tf.variable_scope("q_vals"):
                rand_init = partial(tf.random_uniform, minval=-3e-3, maxval=3e-3)
                q_vals = fully_connected(layer2, 1, var_init=rand_init)
        return state_inputs, action_inputs, q_vals, q_vals


class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bounds, gamma=0.99, sess=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.action_mean = (action_bounds[0] + action_bounds[1]) * 0.5
        self.action_scale = (action_bounds[1] - action_bounds[0]) * 0.5

        self.batch_size = 5

        self.replay_buffer = ReplayBuffer(1000000, state_dim=state_dim, action_dim=action_dim)

        if sess == None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess = sess

        self.actor = ActorModel(state_dim, action_dim, self.action_mean, self.action_scale, self.sess)
        self.critic = CriticModel(state_dim, action_dim, self.sess)

        self.reset_policy()

        writer = tf.summary.FileWriter('logs', self.sess.graph)
        writer.close()

    def reset_policy(self):
        tf.global_variables_initializer().run()

        self.actor.reset_target_model()
        self.critic.reset_target_model()

        self.train_idx = 0
        self.replay_buffer.clear()

    def curr_policy(self):
        return self.actor.get_action

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

        total_reward = 0
        rand_process = OrnsteinUhlenbeckProcess(dt=1.0, theta=0.15, sigma=0.2, 
            mu=np.zeros(self.action_dim), x0=np.zeros(self.action_dim))
        for i in range(max_iter):
            # get action
            action = self.actor.get_action(state)
            # generate random noise for action
            action_noise = rand_process.get_next()
            action += action_noise
            action = np.clip(action, self.action_mean - self.action_scale, self.action_mean + self.action_scale)
            # action = np.array([action.squeeze()])

            [new_state, reward, done, _] = env.step(action)
            new_state = np.reshape(new_state, (1, self.state_dim))
            self.replay_buffer.insert(state, action, reward, new_state, done)

            total_reward += reward
            state = new_state

            if self.train_idx >= (self.batch_size * 3):
                sample = self.replay_buffer.sample(self.batch_size)

                # get target actions
                target_actions = self.actor.get_target_action(sample['next_state'])
                target_q_vals = self.critic.get_target_q_val(sample['next_state'], target_actions)

                disc_return = sample['reward'] + \
                    self.gamma * target_q_vals.squeeze() * (1.0 - sample['terminal'])


                # update critic network
                loss = self.critic.train(sample['state'], sample['action'], disc_return)

                # get actions grads from critic network
                action_grads = self.critic.get_action_grads(sample['state'], sample['action'])[0]

                # update actor network
                self.actor.train(sample['state'], action_grads)
                

                # # update target networks
                self.actor.update_target_model()
                self.critic.update_target_model()

            if done:
                break

            self.train_idx += 1

        return total_reward





if __name__ == '__main__':
    from rl_utils import rollout
    import gym
    import argparse
    import sys
    import pickle

    parser = argparse.ArgumentParser(description="DDPG")
    parser.add_argument('--type', dest='type', action='store',
        required=True,
        choices=['train', 'test'],
        help="type")
    args = parser.parse_args(sys.argv[1:])


    env = gym.make('Pendulum-v0')
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    ddpg = DDPG(state_dim, action_dim, [env.action_space.low, env.action_space.high])


    if args.type == 'train':
        get_state = lambda x: x
        for i in range(10000):
            total_reward = ddpg.update(env, get_state)
            print("Iteration " + str(i) + " reward: " + str(total_reward))
            if i % 20 == 0:
                [_, _, rewards] = rollout(env, ddpg.curr_policy(), get_state, render=True)
                total_reward = np.sum(np.array(rewards))
                print("Test reward: " + str(total_reward))

            if i % 100 == 0:
                ddpg.save_model()

        policy = ddpg.curr_policy()
        rollout(env, policy, get_state, render=True)
        ddpg.save_model()
    elif args.type == 'test':
        ddpg.load_model()
        get_state = lambda x: x
        for i in range(20):
            [_, _, rewards] = rollout(env, ddpg.curr_policy(), get_state, render=True)
            total_reward = np.sum(np.array(rewards))
            print("Test reward: " + str(total_reward))

