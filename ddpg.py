import torch
from torch import nn
from torch.optim import Adam
import numpy as np

from base import Algorithm
from buffer import RolloutBuffer, Buffer
from network import Actor,Critic
from utils import soft_update,hard_update, disable_gradient
from replay_buffer import PriorityExperienceReplay

class DDPG(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995,
                 memory_size=1000000, lr_actor=3e-4,embedding_dim = 100, batch_size = 32,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ddpg = 100,max_episode_num = 8000, lambd=0.97, tau = 0.001):
        super().__init__(state_shape, action_shape, device, seed, gamma)
        self.name = 'DDPG'

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.learning_steps_ddpg = 0
        self.memory_size = memory_size
        self.epoch_ddpg = epoch_ddpg
        self.lambd = lambd
        self.tau = tau

        self.buffer = Buffer(self.memory_size,state_shape,action_shape,device = device)
        self.epsilon_for_priority = 1e-6

        self.actor = Actor(state_shape,action_shape,hidden1=units_actor[0],hidden2=units_actor[1],init_w=0.3).to(device)
        self.actor_target = Actor(state_shape,action_shape,hidden1=units_critic[0],hidden2=units_critic[1],init_w=0.3).to(device)

        self.critic = Critic(state_shape,action_shape,hidden1=units_critic[0],hidden2=units_critic[0],init_w=0.3).to(device)
        self.critic_target = Critic(state_shape,action_shape,hidden1=units_critic[0],hidden2=units_critic[0],init_w=0.3).to(device)
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)
        self.max_episode_num = max_episode_num

    def is_update(self, step):
        return step % self.memory_size == 0

    def step(self, env, state, t, step):
        t += 1
        episode_reward = 0
        correct_count = 0
        steps = 0
        done = False
        #episodic_precision_history = []
        while not done:

       # action, log_pi = self.explore(state)
        #print(action.size())
        # action = np.expand_dims(action,axis=1)
        # action = np.transpose(action,(1,0))
            action = self.exploit(state)
            next_state, reward, done_, _ = env.step(action)
            done = done_
            self.buffer.append(state, action, reward, done, next_state)
            steps += 1
            if reward > 0:
                correct_count += 1
            episode_reward += reward

        #mask = False if t == env._max_episode_steps else done
        #print("current reward: ",reward)
        if done:
            t = 0
            precision = int(correct_count / steps * 100)
            print(
                f'{step}/{self.max_episode_num}, precision : {precision:2}%, total_reward:{episode_reward}')
                # f'{step}/{max_episode_num}, precision : {precision:2}%, total_reward:{episode_reward}, q_loss : {q_loss / steps}')
            # if self.use_wandb:
            #     wandb.log({'precision': precision, 'total_reward': episode_reward, 'epsilone': self.epsilon,
            #                'q_loss': q_loss / steps, 'mean_action': mean_action / steps})
            #episodic_precision_history.append(precision)
            next_state = env.reset()

        return next_state, t

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = self.buffer.sample(batch_size= self.batch_size)

        states = torch.squeeze(states,1)
        next_states = torch.squeeze(next_states,1)
        actions = torch.squeeze(actions,1)

        #rewards = torch.squeeze(next_states,1)

        self.update_ddpg(states, actions, rewards, dones,next_states, writer)


    def update_ddpg(self, states, actions, rewards, dones, next_states,writer):

        next_actions = self.actor(next_states)

        self.update_critic(states, next_states, actions, next_actions, rewards, writer)
        self.update_actor(states, actions, writer)

        soft_update(self.actor_target,self.actor,self.tau)
        soft_update(self.critic_target,self.critic,self.tau)
        self.eval()

    def update_critic(self, states,next_states,actions,next_actions, rewards,writer):
        loss_critic = (self.critic(states) - self.critic_target(states)).pow_(2).mean()
        self.critic.zero_grad()
        next_q_values = self.critic_target(next_states,next_actions)
        q_values = self.critic(states,actions)

        target_q_batch = rewards + self.gamma  * next_q_values
        value_loss = self.criterion(q_values,target_q_batch)
        value_loss.backward()
        self.optim_critic.step()
        self.learning_steps_ddpg += 1
        if self.learning_steps_ddpg % self.epoch_ddpg == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps_ddpg)

    def update_actor(self, states, actions, writer):

        self.actor.zero_grad()
        policy_loss = -self.critic(states,actions)
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.optim_actor.step()


    def save_models(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/ddpg_actor.pth'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/ddpg_critic.pth'.format(output)
        )

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/ddpg_actor.pth'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/ddpg_critic.pth'.format(output))
        )

    @property
    def networks(self):
        return [
            self.actor,
            self.critic,
            self.actor_target,
            self.critic_target,
        ]
###########################################################################
class DDPGExpert(DDPG):
    def __init__(self,state_shape, action_shape, device, path,
                 units_actor=(64, 64)):
        self.device = device
        self.actor = Actor(state_shape,action_shape,hidden1=units_actor[0],hidden2=units_actor[1],init_w=0.3).to(device)
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()
        disable_gradient(self.actor)







