from abc import ABC, abstractmethod
import abc
import os
import numpy as np
import tensorflow as tf


class Algorithm(ABC):
    def __init__(self, state_shape, action_shape, seed, gamma,epsilon = 1.,std = 1.5):
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = (self.epsilon - 0.1) / 500000
        self.std = std

    def explore(self, state):
        action = self.actor.network(state)
        # ε-greedy exploration hyperparameter
        if self.epsilon > np.random.uniform():
            self.epsilon -= self.epsilon_decay
            action += np.random.normal(0, self.std, size=action.shape)
        return action

    def exploit(self, state):
        action = self.actor.network(state)
        return action

    def evaluate(self, epoch):
        pass

    def get_eval_statistics(self):
        return {}

    def get_snapshot(self):
        return {}

    @property
    @abc.abstractmethod
    def networks(self):
        pass

    # @abstractmethod
    # def is_update(self, step):
    #     pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    @abstractmethod
    def load_weights(self, path):
        pass
