import torch
from torch import nn
from torch.optim import Adam

from base import Algorithm
from buffer import RolloutBuffer
from network import Actor,Critic
from utils import soft_update,hard_update


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class DDPG(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ddpg=10, clip_eps=0.2, lambd=0.97,tau = 0.001, coef_ent=0.0,
                 max_grad_norm=10.0):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Rollout buffer. 2 state, 1 action,1 reward
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )
        # Actor. network input s, output a
        # self.actor = StateIndependentPolicy(
        #     state_shape=state_shape,
        #     action_shape=action_shape,
        #     hidden_units=units_actor,
        #     hidden_activation=nn.Tanh()
        # ).to(device)
        self.actor = Actor(state_shape,action_shape,hidden1=units_actor[0],hidden2=units_actor[1],init_w=0.3).to(device)
        self.actor_target = Actor(state_shape,action_shape,hidden1=units_actor[0],hidden2=units_actor[1],init_w=0.3).to(device)
        # Critic. network input [s,a],output x
        # self.critic = StateFunction(
        #     state_shape=state_shape,
        #     hidden_units=units_critic,
        #     hidden_activation=nn.Tanh()
        # ).to(device)
        self.critic = Critic(state_shape,action_shape,hidden1=units_critic[0],hidden2=units_critic[0],init_w=0.3).to(device)
        self.critic_target = Critic(state_shape,action_shape,hidden1=units_critic[0],hidden2=units_critic[0],init_w=0.3).to(device)
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.criterion = nn.MSELoss()
        self.learning_steps_ddpg = 0
        self.rollout_length = rollout_length
        self.epoch_ddpg = epoch_ddpg
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.tau = tau
        self.max_grad_norm = max_grad_norm

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1

        action = self.exploit(state)
        log_pi = 0
        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done

        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ddpg(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ddpg(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states,actions)
            next_actions = self.actor_target(next_states)
            target_values = self.critic_target(next_states,next_actions)

        self.update_critic(states,next_states,actions,next_actions,rewards, writer)
        self.update_actor(states, actions, writer)

        soft_update(self.actor_target,self.actor,self.tau)
        soft_update(self.critic_target,self.critic,self.tau)
        # for _ in range(self.epoch_ddpg):
        #     self.learning_steps_ddpg += 1
        #     self.update_critic(states, targets, writer)
        #     self.update_actor(states, actions, log_pis, gaes, writer)

    def update_critic(self, states,next_states,actions,next_actions, rewards,writer):
       # loss_critic = (self.critic(states) - targets).pow_(2).mean()
        self.critic.zero_grad()
        next_q_values = self.critic_target(next_states,next_actions)
        q_values = self.critic(states,actions)
        self.optim_critic.zero_grad()
        target_q_batch = rewards + self.gamma  * next_q_values
        value_loss = self.criterion(q_values,target_q_batch)
        value_loss.backward()
        self.optim_critic.step()

        # if self.learning_steps_ddpg % self.epoch_ddpg == 0:
        #     writer.add_scalar(
        #         'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, writer):

        self.actor.zero_grad()
        policy_loss = -self.critic(states,actions)
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.optim_actor.step()


    def save_models(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/ddpg_actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/ddpg_critic.pkl'.format(output)
        )

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/ddpg_actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/ddpg_critic.pkl'.format(output))
        )
###########################################################################

#############################testing#############################################
import gym
import numpy as np
ENV_NAME = 'Pendulum-v1'
RANDOMSEED = 1
from trainer import Trainer
import os
import argparse
import datetime
from env import make_env
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

MAX_EPISODES = 200
MAX_EP_STEPS = 200
TEST_PER_EPISODES = 10
VAR = 3 #control exploration

################################################################################
def run(args):
    env = make_env(args.env_id)
    env_test = make_env(args.env_id)

    algo = DDPG(
        state_shape=env.observation_space.shape[0],
        action_shape=env.action_space.shape[0],
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed
    )

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, 'sac', f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()
if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=10 ** 6)
    p.add_argument('--eval_interval', type=int, default=10 ** 4)
    p.add_argument('--env_id', type=str, default=ENV_NAME)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=RANDOMSEED)
    args = p.parse_args()
    run(args)



