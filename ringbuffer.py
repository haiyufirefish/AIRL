import numpy as np


class RingBuffer(object):
    def __init__(self, buffer_size, shape, dtype='float32'):
        self.buffer_size = buffer_size
        self.start = 0
        self.length = 0
        self.data = np.zeros((buffer_size, shape),dtype=dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.buffer_size]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.buffer_size]

    def append(self, v):
        if self.length < self.buffer_size:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.buffer_size:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.buffer_size
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.buffer_size] = v



class Memory(object):
    def __init__(self, size, action_shape, observation_shape):
        self.size = size
        # state
        self.states = RingBuffer(size, shape=observation_shape)
        self.actions = RingBuffer(size, shape=action_shape)
        self.rewards = RingBuffer(size, shape=1)
        self.dones = RingBuffer(size, shape=1,dtype = bool)
        self.ids = RingBuffer(size, shape=1,dtype = np.int32)
        self.next_states = RingBuffer(size, shape=observation_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.randint(self.nb_entries - 2, size=batch_size)

        states_batch = self.states.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        dones_batch = self.dones.get_batch(batch_idxs)
        ids_batch = self.ids.get_batch(batch_idxs)

        result = (
            states_batch,
            action_batch,
            reward_batch,
            next_states_batch,
            dones_batch,
            ids_batch
        )
        return result

    def append(self, state, action, reward, next_state, done, id):

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.ids.append(id)

    @property
    def nb_entries(self):
        return len(self.states)