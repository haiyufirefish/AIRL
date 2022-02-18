import os
from time import time, sleep
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import wandb


class Trainer:

    def __init__(self, env, env_test, algo, log_dir, user_num, item_num,use_wandb = False,load = False,load_step = 1000, seed=0, num_steps=10**5,
                 eval_interval=500, num_eval_episodes=5,num_steps_before_train = 5000):
        super().__init__()
        np.random.seed(seed)
        # Env to collect samples.
        self.env = env
        #self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        #self.env_test.seed(2**31-seed)

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.num_steps_before_train = num_steps_before_train

        self.user_num = user_num
        self.item_num = item_num

        if load:
            path = os.path.join(self.model_dir, f'step_{load_step}').replace("\\", "/")
            self.algo.load_weights(path)

        if use_wandb:
            wandb.login()
            wandb.init(project="AIRL",
                       # entity="diominor",
                       config={'users_num': self.user_num,
                               'items_num': self.item_num,
                               'state_size': self.env.state_size,
                               'embedding_dim': self.algo.embedding_dim,
                               'actor_hidden_dim': self.algo.actor_hidden_dim[0],
                               'actor_learning_rate': self.algo.lr_actor,
                               'critic_hidden_dim': self.algo.critic_hidden_dim[0],
                               'critic_learning_rate': self.algo.lr_critic,
                               'discount_factor': self.algo.gamma,
                               'tau': self.algo.tau,
                               'replay_memory_size': self.algo.memory_size,
                               'batch_size': self.algo.batch_size}
                               )
            algo.set_wandb(True,wandb)

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset().float()

        for step in tqdm(range(1, self.num_steps + 1)):

            #print("current epislon: ",step)
            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step(self.env, state, t, step)
            #state = state.float()
            # Update the algorithm whenever ready.
            #print("update: ",self.algo.is_update(step))
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)
                path = os.path.join(self.model_dir, f'step_{step}').replace("\\","/")
                if not os.path.exists(path):
                    os.makedirs(path)
                self.algo.save_models(
                    path)

        # Wait for the logging to be finished.
        wandb.finish()
        sleep(10)

    def train_imitation(self):
        self.start_time = time()
        # Episode's timestep.
        t = 0
        start_epoch = 0

        # Initialize the environment.
        state = self.env.reset()
        # fill the buffer of policy
        for step in tqdm(range(self.num_steps_before_train)):
            state, t = self.algo.policy.step(self.env, state, t, step)
        for step in tqdm(range(start_epoch,self.num_steps)):
            state, t = self.algo.policy.step(self.env, state, t, step)
            #state = state.float()
            if self.algo.is_update(step):
                #print("step: ",step)
                self.algo.update(self.writer)
            # Evaluate regularly.
            # if step % self.eval_interval == 0:
            #     self.evaluate(step)
            #     path = os.path.join(self.model_dir, f'step_{step}').replace("\\","/")
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #     self.algo.save_models(
            #         path)
    def evaluate(self, step):
        mean_return = 0.0

        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset().float()
            episode_return = 0.0
            done = False

            while (not done):
                action = self.algo.exploit(state)
                # log_pi = 0
                next_state, reward, done, _ = self.env_test.step(action)
                #state, reward, done, _ = self.env_test.step(action)

                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        self.writer.add_scalar('return/test', mean_return, step)
        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
