"""
Replay Buffer used for in deep RL algorithms.
Buffer can store tuples of (state, action, reward, next_state, terminal)
"""
import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_entries, state_dim, action_dim):
        """
        Initializes Replay Buffer
        
        Args:
            max_entries (int): maximum number of entries in buffer
            state_dim (int): number of dimensions in state vector
            action_dim (int): number of dimensions in action vector
        """
        self.max_entries = max_entries 
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clear()

    def clear(self):
        """
        Clears the replay buffer of all entries
        """
        self.buffer = {'state': np.zeros((self.max_entries, self.state_dim)),
            'action': np.zeros((self.max_entries, self.action_dim)),
            'reward': np.zeros(self.max_entries),
            'next_state': np.zeros((self.max_entries, self.state_dim)),
            'terminal': np.zeros(self.max_entries)}
        self.__buffer_len = 0
        self.__erase_idx = 0

    def __insert(self, dict_entry, idx):
        for key, val in dict_entry.iteritems():
            self.buffer[key][idx] = val

    def insert(self, state, action, reward, next_state, terminal):
        """
        inserts a tuple of (state, action, reward, next_state, terminal)
        
        Args:
            state (numpy array): array of length state_dim
            action (numpy array): array of length action_dim
            reward (float or numpy array): reward
            next_state (numpy array): array of length state_dim
            terminal (bool or int): whether or not episode ended
        """
        dict_entry = {'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'terminal': terminal}

        if self.__buffer_len >= self.max_entries:
            self.__insert(dict_entry, self.__erase_idx)

            self.__erase_idx += 1
            if self.__erase_idx >= self.max_entries:
                self.__erase_idx = 0
        else:
            self.__insert(dict_entry, self.__buffer_len)

            self.__buffer_len += 1

    def sample(self, num_samples):
        """
        Draws 'num_samples' random samples from Replay Buffer
        
        Args:
            num_samples (int): number of samples to draw
        
        Returns:
            dict: dict of numpy arrays
            dict has the following keys: 'state', 'action', 'reward', 'next_state', 'terminal'
            'state', 'next_state' dict entries have size 'num_samples' by state_dim
            'action' dict entries have size 'num_samples' by action_dim
            'terminal' and 'reward' dict entries have size 'num_samples'
        """
        rand_idx = np.random.randint(self.__buffer_len, size=num_samples)
        
        dict_entry = {}
        for key in self.buffer:
            dict_entry[key] = self.buffer[key][rand_idx]

        return dict_entry
