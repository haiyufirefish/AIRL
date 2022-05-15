from base import Algorithm
from ringbuffer import Memory
from network_ import Actor,Critic
import os
import tensorflow as tf
from utils import calculate_td_target


class DDPG(Algorithm):
    def __init__(self, action_shape, seed, gamma = 0.9, ac_lr = 1e-3, cr_lr = 1e-3,tau = 0.001,
                 state_size = 10, embedding_dim = 100, actor_hidden_dim = 128, critic_hidden_dim = 128,
                 replay_memory_size = 1000000, batch_size = 32, is_test=False, max_episode_num = 8000,
                 epsilon = 1.,std = 1.5):

        super().__init__(state_size, action_shape, seed, gamma,epsilon,std)

        self.name = 'DDPG_Ring'
        self.ac_lr = ac_lr
        self.cr_lr = cr_lr
        self.tau = tau
        self.embedding_dim = embedding_dim
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size
        self.max_episode_num = max_episode_num

        self.actor = Actor(self.embedding_dim, self.actor_hidden_dim, self.ac_lr, state_size, self.tau)
        self.critic = Critic(self.critic_hidden_dim, self.cr_lr, self.embedding_dim, self.tau)
        self.actor.build_networks()
        self.critic.build_networks()
        print("initial! ")
        self.buffer = Memory(self.replay_memory_size, self.embedding_dim, self.embedding_dim*3)
        self.epsilon_for_priority = 1e-6

    def is_update(self):
        return self.buffer.nb_entries > self.batch_size

    def step(self,state,t):
        pass

    def update(self):
        states, actions, rewards, next_states, dones, _ = \
            self.buffer.sample(batch_size=self.batch_size)
        return self.update_ddpg(states, actions, rewards, next_states, dones)

    def update_ddpg(self,states, actions, rewards, next_states, dones):
        q_loss = 0
        target_next_action = self.actor.target_network(next_states)
        qs = self.critic.network([target_next_action, next_states])
        target_qs = self.critic.target_network([target_next_action, next_states])
        # Dueling DQN
        min_qs = tf.raw_ops.Min(input=tf.concat([target_qs, qs], axis=1), axis=1, keep_dims=True)
        td_targets = calculate_td_target(self.gamma, rewards, min_qs, dones)
        # td_targets = utils.calculate_td_target(rewards, min_qs, dones)
        # Update priority

        q_loss += self.critic.train([actions, states], td_targets)
        # Update actor network
        s_grads = self.critic.dq_da([actions, states])
        self.actor.train(states, s_grads)
        self.actor.update_target_network()
        self.critic.update_target_network()
        return q_loss

    # def set_wandb(self, use_wandb=False, wandb=None):
    #     self.use_wandb = use_wandb
    #     self.wandb = wandb

    def load_weights(self, path):
        print('load weights success!')
        self.actor.load_weights('{}/ddpg_actor.h5'.format(path))
        self.critic.load_weights('{}/ddpg_critic.h5'.format(path))

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.actor.save_weights('{}/ddpg_actor.h5'.format(save_dir))
        self.critic.save_weights('{}/ddpg_critic.h5'.format(save_dir))

    @property
    def networks(self):
        return [
            self.actor,
            self.critic,
            self.actor_target,
            self.critic_target,
        ]