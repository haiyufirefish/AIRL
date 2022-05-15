import os
from time import time, sleep
from datetime import timedelta
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, env, env_test, algo, user_num, item_num,
                 use_wandb=False, load=False, load_step=8000,
                 load_memory=False, seed=0, num_steps=4000,
                 eval_interval=500, num_eval_episodes=5, num_steps_before_train=5000, model_dir = './model',
                 memory_dir = './memory',top_k = False):
        super().__init__()
        self.num_steps_before_train = num_steps_before_train
        self.num_eval_episodes = num_eval_episodes
        self.eval_interval = eval_interval
        self.num_steps = num_steps
        self.seed = seed
        self.load_memory = load_memory
        self.item_num = item_num
        self.user_num = user_num
        self.algo = algo
        self.env_test = env_test
        self.env = env
        self.memory_dir = memory_dir
        self.top_k = top_k

        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if load:
            path = os.path.join(self.model_dir, f'epoch_{load_step}').replace("\\", "/")
            self.algo.load_weights(path)
            print('Load weights!')

        if load_memory:
            memory_path = os.path.join(self.memory_dir, self.algo.name).replace("\\", "/")
            self.algo.buffer.load(memory_path)
            print('Load memory!')

        if use_wandb:
            wandb.login()
            if self.algo.name == 'ppo2':
                wandb.init(project="airl_movie_tf",
                           # entity="diominor",
                           config={'users_num': self.user_num,
                                   'items_num': self.item_num,
                                   'state_size': self.env.state_size,
                                   'algo': self.algo.name,
                                   'embedding_dim': self.algo.action_dim,
                                   'actor_learning_rate': self.algo.actor_lr,
                                   'critic_hidden_dim': 128,
                                   'critic_learning_rate': self.algo.critic_lr,
                                   'discount_factor': self.algo.gamma,
                                   'batch_size': self.algo.update_interval}
                           )
            else:
                wandb.init(project="airl_movie_tf",
                           # entity="diominor",
                           config={'users_num': self.user_num,
                                   'items_num': self.item_num,
                                   'state_size': self.env.state_size,
                                   'algo': self.algo.name,
                                   'embedding_dim': self.algo.embedding_dim,
                                   'actor_hidden_dim': self.algo.actor_hidden_dim,
                                   'actor_learning_rate': self.algo.ac_lr,
                                   'critic_hidden_dim': self.algo.critic_hidden_dim,
                                   'critic_learning_rate': self.algo.cr_lr,
                                   'discount_factor': self.algo.gamma,
                                   'batch_size': self.algo.batch_size}
                           )
           # algo.set_wandb(use_wandb, wandb)
        self.use_wandb = use_wandb


    # ddpg train method
    def train(self):
        if(self.algo.name == 'ppo2'):
            self.train_ppo()
        else:
        # Time to start training.

            episodic_precision_history = []

            for epoch in tqdm(range(1, self.num_steps + 1)):
                episode_reward = 0
                correct_count = 0
                step = 0
                q_loss = 0
                mean_action = 0
                # Initialize the environment.
                state, _, done_ = self.env.reset()
                while not done_:
                    step += 1
                    action = self.algo.explore(state)
                    next_state, reward, done_, _ = self.env.step(action, top_k=self.top_k)

                    self.algo.buffer.append(state, action, reward, next_state, done_,self.env.user)
                    mean_action += np.sum(action[0]) / (len(action[0]))
                    if self.top_k:
                        reward = np.sum(reward)
                    if reward > 0:
                        correct_count += 1
                    episode_reward += reward

                    if self.algo.is_update():
                        q_loss += self.algo.update()

                    print(
                        f'recommended items : {len(self.env.recommended_items)},  epsilon : {self.algo.epsilon:0.3f}, reward : {reward:+}',
                        end='\r')
                    if done_:
                        precision = int(correct_count / step * 100)
                        print(
                            f'{epoch}/{self.num_steps}, precision : {precision:2}%, total_reward:{episode_reward}, mean_action : {mean_action / step}')
                        if self.use_wandb:
                            wandb.log({'precision': precision, 'total_reward': episode_reward, 'epsilone': self.algo.epsilon,
                                        'q_loss':q_loss/step,'mean_action': mean_action / step})
                        episodic_precision_history.append(precision)

                if (epoch) % 500 == 0:
                    plt.plot(episodic_precision_history)
                    plt.savefig('./images/training_precision_%_top_5.png')

                # Update the algorithm whenever ready.

                episode_reward += reward
                mean_action += np.sum(action[0]) / (len(action[0]))


                # Evaluate regularly.
                if epoch % self.eval_interval == 0:
                    #self.evaluate(step)
                    path = os.path.join(self.model_dir, f'step_{epoch}').replace("\\","/")
                    if not os.path.exists(path):
                        os.makedirs(path)
                    self.algo.save_models(
                        path)

            # Wait for the logging to be finished.
            memory_path = os.path.join(self.memory_dir,self.algo.name).replace("\\","/")
            if not os.path.exists(memory_path ):
                os.makedirs(memory_path)
            #self.algo.buffer.save(memory_path)
            wandb.finish()
            sleep(10)

    # ppp train method
    def train_ppo(self):
        for ep in tqdm(range(self.num_steps)):
            state_batch = []
            action_batch = []
            reward_batch = []
            old_policy_batch = []

            episode_reward = 0

            state, _, done = self.env.reset()
            t = 0
            count = 0
            q_loss_critic = 0
            q_loss_actor = 0
            while not done:
                t +=1
                # self.env.render()
                log_old_policy, action = self.algo.actor.get_action(state)
                #action 100,1 1*100
                next_state, reward, done, _ = self.env.step(np.expand_dims(action,axis=1).transpose())

                state = np.reshape(state, [1,  self.algo.state_dim])
                action = np.reshape(action, [1, self.algo.action_dim])
                next_state = np.reshape(next_state, [1,  self.algo.state_dim])
                #reward = np.reshape(reward, [1, 1])
                log_old_policy = np.reshape(log_old_policy, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                old_policy_batch.append(log_old_policy)

                if len(state_batch) >= self.algo.update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    #print(reward_batch)
                    rewards = reward_batch
                    old_policys = self.list_to_batch(old_policy_batch)

                    v_values = self.algo.critic.model.predict(states)
                    next_v_value = self.algo.critic.model.predict(next_state)

                    gaes, td_targets = self.algo.gae_target(
                        rewards, v_values, next_v_value, done)

                    for epoch in range(self.algo.update_epochs):
                        q_loss_actor += self.algo.actor.train(
                            old_policys, states, actions, gaes)
                        q_loss_critic += self.algo.critic.train(states, td_targets)
                    q_loss_critic /= self.algo.update_epochs
                    q_loss_actor /= self.algo.update_epochs
                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    old_policy_batch = []
                if reward > 0:
                    count += 1
                episode_reward += reward
                state = next_state
            precision = int(count/t *100)
            print(f'{ep}/{self.num_steps}, precision : {precision:2}%, total_reward:{episode_reward},actor loss: {q_loss_actor/t}, critic loss {q_loss_critic/t}')

            #wandb.log({'Reward': episode_reward})
            if self.use_wandb:
                wandb.log({'precision': precision, 'cumulative reward': episode_reward, 'epsilone': ep,'Actor Loss':(q_loss_actor/t),'Critic Loss': (q_loss_critic/t)})
            if ep % 500 == 0:
                path = os.path.join(self.model_dir, f'epoch_ppo_{ep}').replace("\\", "/")
                if not os.path.exists(path):
                    os.makedirs(path)
                self.algo.save_models(
                    path)

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch


    # imitation train method
    def train_imitation(self):
        self.start_time = time()


        for _ in tqdm(range(self.num_steps_before_train)):
            state, _, done = self.env.reset()
            while not done:
                action = self.algo.explore(state)
                next_state, reward, done, _ = self.env.step(action, top_k=self.top_k)
                self.algo.buffer.append(state, action, reward, next_state, done, self.env.user)

        print("pretrain done!")
        episodic_precision_history = []
        for epoch in tqdm(range(1, self.num_steps + 1)):
            episode_reward = 0
            correct_count = 0
            step = 0
            q_loss = 0
            mean_action = 0
            # Initialize the environment.
            state, _, done_ = self.env.reset()
            while not done_:
                step += 1
                action = self.algo.explore(state)
                next_state, reward, done_, _ = self.env.step(action, top_k=self.top_k)

                self.algo.buffer.append(state, action, reward, next_state, done_, self.env.user)
                mean_action += np.sum(action[0]) / (len(action[0]))
                if reward > 0:
                    correct_count += 1
                episode_reward += reward

                q_loss += self.algo.update()
                print(
                    f'recommended items : {len(self.env.recommended_items)},  epsilon : {self.algo.epsilon:0.3f}, reward : {reward:+}',
                    end='\r')
                if done_:
                    precision = int(correct_count / step * 100)
                    print(
                        f'{epoch}/{self.num_steps}, precision : {precision:2}%, total_reward:{episode_reward}, mean_action : {mean_action / step}')
                    if self.use_wandb:
                        wandb.log(
                            {'precision': precision, 'cumulative reward': episode_reward, 'epsilone': self.algo.epsilon,
                             'critic loss': q_loss / step, 'mean_action': mean_action / step})
                    episodic_precision_history.append(precision)

            if (epoch) % 500 == 0:
                plt.plot(episodic_precision_history)
                plt.savefig('./images/training_precision_%_top_5.png')

            # Update the algorithm whenever ready.

            episode_reward += reward
            mean_action += np.sum(action[0]) / (len(action[0]))

            # Evaluate regularly.
            if epoch % self.eval_interval == 0:
                # self.evaluate(step)
                path = os.path.join(self.model_dir, f'epoch_{epoch}').replace("\\", "/")
                if not os.path.exists(path):
                    os.makedirs(path)
                self.algo.save_models(
                    path)

        # Wait for the logging to be finished.
        memory_path = os.path.join(self.memory_dir, self.algo.name).replace("\\", "/")
        if not os.path.exists(memory_path):
            os.makedirs(memory_path)
        # self.algo.buffer.save(memory_path)
        wandb.finish()
        sleep(10)

    # def evaluate(self, step):
    #     mean_return = 0.0
    #
    #     for _ in range(self.num_eval_episodes):#5
    #         state,_, done = self.env_test.reset()
    #         episode_return = 0.0
    #
    #         for _ in range(10):
    #             action = self.algo.exploit(state)
    #             # log_pi = 0
    #             next_state, reward, done, _ = self.env_test.step(action)
    #             episode_return += reward
    #
    #         mean_return += episode_return / self.num_eval_episodes
    #
    #     #self.writer.add_scalar('return/test', mean_return, step)
    #     print(f'Num steps: {step:<6}   '
    #           f'Return: {mean_return:<5.1f}   '
    #           f'Time: {self.time}')

    # for single or top N recommendation
    def evaluate_final(self,check_items = False, top_k = False):
        mean_precision = 0
        mean_ndcg = 0
        state,_,done = self.env_test.reset()
        if check_items:
            print(f'user_id : {self.env_test.user}, rated_items_length:{len(self.env_test.user_items)}')
            #print('items : \n', np.array(self.env_test.get_items_names(self.env_test.items_ids)))
        action = self.algo.exploit(state)
        recommended_item = self.env_test.recommend_item(action,self.env_test.recommended_items,top_k)
        if check_items:
            print(f'recommended items ids : {recommended_item}')
        next_state, reward, done_, _ = self.env_test.step(action, top_k)

        if top_k:
            correct_list = [1 if r > 0 else 0 for r in reward]
            # ndcg
            dcg, idcg = self.calculate_ndcg(correct_list, [1 for _ in range(len(reward))])
            mean_ndcg += dcg / idcg

            # precision
            correct_num = top_k - correct_list.count(0)
            mean_precision += correct_num / top_k

        reward = np.sum(reward)

        if check_items:
            if top_k:
                print(f'precision : {correct_num / top_k}, dcg : {dcg:0.3f}, '
                      f'idcg : {idcg:0.3f}, ndcg : {dcg / idcg:0.3f}, reward : {reward}')
        if top_k:
            return mean_precision, mean_ndcg
        else:
            return reward > 0, mean_ndcg

    # ndcg calculation
    def calculate_ndcg(rel, irel):
        dcg = 0
        idcg = 0
        rel = [1 if r > 0 else 0 for r in rel]
        for i, (r, ir) in enumerate(zip(rel, irel)):
            dcg += (r) / np.log2(i + 2)
            idcg += (ir) / np.log2(i + 2)
        return dcg, idcg
    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))