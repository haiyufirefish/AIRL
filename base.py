from abc import ABC, abstractmethod
import abc
import os
import numpy as np
import torch


class Algorithm(ABC):

    def __init__(self, state_shape, action_shape, device, seed, gamma):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state.unsqueeze_(0))

        return action.squeeze_(1), log_pi.item()

    def exploit(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            #action = self.actor(state.unsqueeze_(0))#

            action = self.actor(state)
        return action

    def evaluate(self,epoch):
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
