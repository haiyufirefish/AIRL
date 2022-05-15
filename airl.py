import numpy as np
import tensorflow as tf
from ddpg_ import DDPG
from utils import get_label_batch


# class AIRL(DDPG):
#
#     def __init__(self,action_shape, seed, gail,dict,epoch_disc, gamma = 0.9, ac_lr = 1e-3, cr_lr = 1e-3,tau = 0.001,
#                  state_size = 10, embedding_dim = 100, actor_hidden_dim = 128, critic_hidden_dim = 128,
#                  replay_memory_size = 1000000, batch_size = 32, is_test=False, max_episode_num = 8000,
#                  epsilon = 1.,std = 1.5):
#         super().__init__(action_shape,seed,gamma,ac_lr,cr_lr,tau,state_size,embedding_dim,actor_hidden_dim,
#                          critic_hidden_dim,replay_memory_size,batch_size,is_test,max_episode_num,epsilon,std)
#
#         self.name = 'ddpg_airl'
#         self.epoch_disc = epoch_disc
#         self.epoch_policy = epoch_disc
#         self.gail = gail
#         self.dict = dict
#
#     def is_update(self):
#         return super.is_update()
#
#     def step(self):
#         pass
#
#     def update(self):
#         self.learning_steps +=1
#         # update discriminator
#         for _ in range(self.epoch_disc):
#             states, actions, rewards, next_states, dones, weight_batch, index_batch,batch_ids= \
#                 self.buffer.sample(batch_size=self.batch_size)
#             states_exp,actions_exp,marks = get_label_batch(self.dict,batch_ids,self.embedding_dim)
#             self.gail.update(states,actions,states_exp, actions_exp,marks)
#
#         states, actions, _, next_states, dones, weight_batch, index_batch,ids = \
#             self.buffer.sample(batch_size=self.batch_size)
#         q_loss = 0.0
#         for _ in range(self.epoch_policy):
#             states, actions, _, next_states, dones, weight_batch, index_batch, __ = \
#                 self.buffer.sample(batch_size=self.batch_size)
#             rewards = self.gail.get_reward(states,actions)
#             q_loss+=self.update_ddpg(states, actions, rewards, next_states, dones, weight_batch, index_batch)
#
#         return q_loss/self.epoch_policy

class InverseDDPG(DDPG):

    def __init__(self,action_shape, seed, disc,dict,epoch_disc, gamma = 0.9, ac_lr = 1e-3, cr_lr = 1e-3,tau = 0.001,
                 state_size = 10, embedding_dim = 100, actor_hidden_dim = 128, critic_hidden_dim = 128,
                 replay_memory_size = 1000000, batch_size = 32, is_test=False, max_episode_num = 8000,
                 epsilon = 1.,std = 1.5):
        super().__init__(action_shape,seed,gamma,ac_lr,cr_lr,tau,state_size,embedding_dim,actor_hidden_dim,
                         critic_hidden_dim,replay_memory_size,batch_size,is_test,max_episode_num,epsilon,std)

        self.name = 'ddpg_'+disc.gail_loss

        self.epoch_disc = epoch_disc
        self.epoch_policy = epoch_disc
        self.disc = disc
        self.dict = dict

    def is_update(self):
        return super.is_update()

    def step(self):
        pass

    def update(self):
        self.learning_steps +=1
        # update discriminator
        for _ in range(self.epoch_disc):
            states, actions, _, next_states, dones, batch_ids= \
                self.buffer.sample(batch_size=self.batch_size)
            states_exp,actions_exp,_ = get_label_batch(self.dict,batch_ids,self.embedding_dim)
            self.disc.update(states,actions,states_exp, actions_exp)

        q_loss = 0.0
        for _ in range(self.epoch_policy):
            states, actions, __, next_states, dones, _ = \
                self.buffer.sample(batch_size=self.batch_size)
            rewards = self.disc.get_reward(states,actions)
            q_loss += self.update_ddpg(states, actions, rewards, next_states, dones)
        return q_loss/self.epoch_policy