import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import autograd

import numpy as np
from base import Algorithm
# from ppo import PPO
from disc import MLPDisc
from collections import OrderedDict



class AIRL(Algorithm):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,policy,mode,
                 gamma=0.995,hid_act = 'relu',state_only=False, rollout_length=10000,batch_size=64, lr_disc=3e-4,
                 units_disc=100,disc_momentum = 0.0,use_grad_pen = False,grad_pen_weight=10,
                 num_update_loops_per_train_call = 1,  num_disc_epoch = 50,
                 num_policy_epoch = 50,policy_optim_batch_size_from_expert=0,
                clip_eps=0.2, lambd=0.97, max_grad_norm=10.0):
        assert mode in [
            "airl",
            "gail",
        ], "Invalid adversarial irl algorithm!"
        super().__init__(
            state_shape, action_shape, device, seed, gamma
        )
        self.num_disc_epoch = num_disc_epoch
        self.num_policy_epoch = num_policy_epoch
        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.rollout_length = rollout_length
        self.mode = mode

        self.state_only = state_only
        # Expert's buffer.
        self.buffer_exp = buffer_exp
        #penalty
        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight
        #policy
        self.policy = policy
        self.buffer = self.policy.buffer
        self.buffer_policy =self.policy.buffer
        self.policy_optim_batch_size_from_expert = policy_optim_batch_size_from_expert

        self.disc = MLPDisc(state_shape=state_shape,hid_dim=units_disc,
                            hid_act=hid_act,clamp_magnitude=max_grad_norm).to(device)
        # print(self.disc)
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(batch_size, 1),
                torch.zeros(batch_size, 1),
            ],
            dim=0,
        )
        self.bce.to(device)
        self.bce_targets.to(device)
        self.learning_steps_disc = 0
        self.disc_optimizer = Adam(self.disc.parameters(), lr=lr_disc,betas=(disc_momentum, 0.999))
        self.batch_size = batch_size
        #self.epoch_disc = epoch_disc
        # evalute
        self.disc_eval_statistics = None

    def update(self, writer):
        """
                Train the discriminator and policy
        """
        self.learning_steps += 1
        for _ in range(self.num_update_loops_per_train_call):

            for epoch in range(1,self.num_disc_epoch+1):
                self.learning_steps_disc += 1
                self.update_disc(
                    epoch=epoch, writer=writer
                )
        # Train policy
            for epoch in range(self.num_policy_epoch):
                self.policy.update(writer = writer)

    def update_disc(self, epoch, writer):

        # Samples from current policy's trajectories.
        states, actions, _, dones, next_states = self.buffer.sample(self.batch_size)
        # Samples from expert's demonstrations.
        states_exp, actions_exp, _, dones_exp, next_states_exp = \
            self.buffer_exp.sample(self.batch_size)
        # Calculate log probabilities of expert actions.
        policy_obs = states
        expert_obs = states_exp

        if self.state_only:
            policy_disc_input = torch.cat([policy_obs, next_states], dim=1)
            expert_disc_input = torch.cat([expert_obs, next_states_exp], dim=1)
        else:

            policy_disc_input = torch.cat([policy_obs, actions], dim=2)
            expert_disc_input = torch.cat([expert_obs, actions_exp], dim=2)
        disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0)
        disc_input = torch.squeeze(disc_input,dim = 1)
        disc_logits = self.disc(disc_input)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())
        disc_ce_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = torch.rand(expert_obs.size(0), 1, device=self.device)
            interp_obs = eps * expert_disc_input + (1 - eps) * policy_disc_input
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)

            gradients = autograd.grad(
                outputs=self.disc(interp_obs).sum(),
                inputs=[interp_obs],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight
        else:
            disc_grad_pen_loss = 0.0

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        disc_total_loss.backward()
        self.disc_optimizer.step()
        # with torch.no_grad():
        #     log_pis_exp = self.actor.evaluate_log_pi(
        #         states_exp, actions_exp)
        # Update discriminator.

        if self.learning_steps_disc % epoch == 0:
            writer.add_scalar(
                'loss/disc', disc_total_loss.item(), self.learning_steps)

            # Discriminator's accuracies.
            # with torch.no_grad():
            #     acc_pi = (disc_logits < 0).float().mean().item()
            #     acc_exp = (disc_preds > 0).float().mean().item()
            # writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            # writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)

        """
                Save some statistics for eval
        """
        if self.disc_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()

            self.disc_eval_statistics["Disc CE Loss"] = np.mean(
                disc_ce_loss.detach().numpy()
            )
            self.disc_eval_statistics["Disc Acc"] = np.mean(accuracy.detach().numpy())
            if self.use_grad_pen:
                self.disc_eval_statistics["Grad Pen"] = np.mean(
                    gradient_penalty.detach().numpy()
                )
                self.disc_eval_statistics["Grad Pen W"] = np.mean(self.grad_pen_weight)

    def update_policy(self, epoch):
        if self.policy.name == 'PPO':
            states, actions, _, dones, log_pis, next_states, = self.buffer.sample(self.batch_size)
        else:
            states, actions, _, dones, next_states = self.buffer.sample(self.batch_size)

        self.disc.eval()
        if self.state_only:
            disc_input = torch.cat([states, next_states], dim=1)
        else:
            disc_input = torch.cat([states, actions], dim=1)
        disc_logits = self.disc(disc_input).detach()
        self.disc.train()

        if self.mode == "airl":
            # compute log(D) - log(1-D) then just get the logits
            rewards = disc_logits
        else:  # -log (1-D) > 0 #gail
            rewards = F.softplus(
                disc_logits, beta=1
            )  # F.softplus(disc_logits, beta=-1)
        if self.policy.name == 'PPO':
            self.policy.update(states, actions, rewards, dones, log_pis, next_states)
        else:
            self.policy.update(states, actions, rewards, dones, next_states)

        self.disc_eval_statistics["Disc Rew Mean"] = np.mean(
            rewards.detach().numpy()
        )
        self.disc_eval_statistics["Disc Rew Std"] = np.std(
            rewards.detach().numpy()
        )
        self.disc_eval_statistics["Disc Rew Max"] = np.max(
            rewards.detach().numpy()
        )
        self.disc_eval_statistics["Disc Rew Min"] = np.min(
            rewards.detach().numpy()
        )
    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.disc_eval_statistics)
        self.eval_statistics.update(self.policy_trainer.get_eval_statistics())
        super().evaluate(epoch)

    def is_update(self, step):
        return step % self.rollout_length == 0

    def load_weights(self, path):
        if path is None: return
        self.disc.load_state_dict(torch.load('{}/airl_disc.pth'.format(path)))
        self.policy.load_state_dict(torch.load('{}/airl_policy.pth'.format(path)))

    def save_models(self, save_dir):

        torch.save(
            self.disc.state_dict(),
            '{}/airl_disc.pth'.format(save_dir)
        )
        torch.save(
            self.policy.state_dict(),
            '{}/airl_policy.pth'.format(save_dir)
        )

    @property
    def networks(self):
        return [self.disc] + self.policy.networks