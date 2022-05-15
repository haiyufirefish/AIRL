import numpy as np
import os
# from td3 import TD3
from ddpg_ import DDPG
from env import OfflineEnv
from ppo2 import PPO2
from trainer import Trainer
from state import DRRAveStateRepresentation
from embedding import UserMovieEmbedding

STATE_SIZE = 10
SEED = 155
EMBEDDING_DIM = 100
NUMBER_STEPS = 8000

if __name__ == '__main__':

    users_dict = np.load("./data/user_dict_100k.npy", allow_pickle=True).item()
    users_history_lens = np.load("./data/users_histroy_len_100k.npy")

    users_num = len(users_dict) + 1
    items_num_list = set()
    for i in users_dict.items():
        for j in i[1]:
            items_num_list.add(j[0])
    items_num = len(items_num_list) + 1

    # state
    state_representation = DRRAveStateRepresentation(EMBEDDING_DIM)
    state_representation([np.zeros((1, EMBEDDING_DIM,)), np.zeros((1, STATE_SIZE, EMBEDDING_DIM))])

    # embedding net
    emb_net = UserMovieEmbedding(users_num, items_num, EMBEDDING_DIM)
    emb_net([np.zeros((1,)), np.zeros((1,))])
    emb_net.load_weights('./weights/user_movie_embedding_100k.h5')

    train_users_num = int(users_num * 0.8)
    train_users_dict = {k: users_dict.get(k) for k in range(1, train_users_num + 1)}
    train_users_history_lens = users_history_lens[:train_users_num]

    # train_users_dict = {key:value[:20] for key,value in [x for x in users_dict.items()][0:30]}
    # train_users_history_lens = {key:value for key,value in[x for x in users_history_lens.items()][0:30]}
    # items_num_list = []
    # for k,v in train_users_dict.items():
    #     for vv in v:
    #         items_num_list.append(vv[0])

    env = OfflineEnv(train_users_dict, train_users_history_lens, items_num_list, STATE_SIZE, state_representation, emb_net,
                     seed=SEED)
    # algo = TD3(state_shape=(1, 300),
    #             action_shape=(1, 100),
    #            units_actor=(64, 32),
    #            units_critic=(64, 32),
    #             lr_actor=3e-3,
    #             lr_critic= 3e-3,
    #             memory_size=1000000,
    #             batch_size=32,
    #             embedding_dim=100,
    #              seed=SEED)

    # algo = DDPG(action_shape = EMBEDDING_DIM,seed = SEED,gamma = 0.9, ac_lr = 1e-3, cr_lr = 1e-3,tau = 0.001,
    #              state_size = STATE_SIZE, embedding_dim = EMBEDDING_DIM, actor_hidden_dim = 128, critic_hidden_dim = 128,
    #              replay_memory_size = 1000000, batch_size = 64, is_test=False, max_episode_num = NUMBER_STEPS,epsilon = 1.,std = 1.5)
    algo = DDPG(action_shape=EMBEDDING_DIM, seed=SEED, gamma=0.9, ac_lr=1e-3, cr_lr=1e-3, tau=0.001,
                     state_size=STATE_SIZE, embedding_dim=EMBEDDING_DIM, actor_hidden_dim=128, critic_hidden_dim=128,
                     replay_memory_size=1000000, batch_size=64, is_test=False, max_episode_num=NUMBER_STEPS, epsilon=1.,
                     std=1.5)
    # algo = PPO2(state_dim = EMBEDDING_DIM*3, action_shape=EMBEDDING_DIM, seed = SEED, env = env, gamma=0.99, clip_ratio=0.2, lamdb=0.95,
    #              actor_lr=3e-4, critic_lr=1e-3,update_interval = 63,update_epochs = 5)

    # mode = 'ppo'

    print("data load done.")
    print('processing!')
    # print("done !!")
    recommender = Trainer(env, env, algo, users_num, items_num,
                 use_wandb= False, load=False, load_step=9000,
                 load_memory= False, seed=0, num_steps= NUMBER_STEPS,
                 eval_interval= 500, num_eval_episodes=5,num_steps_before_train=5000,
                 model_dir = './model', memory_dir = './memory',top_k = False)


    recommender.train()
    print("done!")