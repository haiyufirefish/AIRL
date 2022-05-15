import numpy as np
import random
import scipy.signal


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.size = size
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer = 0
        self.trajectory_start_index = 0

    def discounted_cumulative_sums(self, x, discount):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def append(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = self.discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self,batch_size = 64):
        # Get all data of the buffer and normalize the advantages
        # self.pointer, self.trajectory_start_index = 0, 0
        start = (self.pointer - batch_size) % self.size
        idxes = slice(start, start + batch_size)
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer[idxes],
            self.action_buffer[idxes],
            self.advantage_buffer[idxes],
            self.return_buffer[idxes],
            self.logprobability_buffer[idxes],
        )


class ReplayBuffer(object):

        '''
        apply PER
        '''

        def __init__(self, buffer_size, embedding_dim):
            self.buffer_size = buffer_size
            self.crt_idx = 0
            self.is_full = False

            '''
                state : (300,), 
                next_state : (300,) 
                actions : (100,), 
                rewards : (1,), 
                dones : (1,)
            '''
            self.states = np.zeros((buffer_size, 3 * embedding_dim), dtype=np.float32)
            self.actions = np.zeros((buffer_size, embedding_dim), dtype=np.float32)
            self.dones = np.zeros(buffer_size, np.bool)

        def append(self, state, action, done):
            self.states[self.crt_idx] = state
            self.actions[self.crt_idx] = action
            self.dones[self.crt_idx] = done

            self.crt_idx = (self.crt_idx + 1) % self.buffer_size
            if self.crt_idx == 0:
                self.is_full = True
