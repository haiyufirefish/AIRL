import pandas as pd
import numpy as np
import os
import tensorflow as tf
# from algo import ALGOS
# from td3 import TD3
from ppo2 import PPO2
from env import OfflineEnv
from trainer import Trainer
from state import DRRAveStateRepresentation
from embedding import UserMovieEmbedding
from ddpg_ import DDPG


STATE_SIZE = 10
SEED = 155
EMBEDDING_DIM = 100
NUMBER_STEPS = 8000


if __name__ == '__main__':
    users_dict = np.load("data/user_dict_100k.npy", allow_pickle=True).item()
    users_history_lens = np.load("data/users_histroy_len_100k.npy")

    users_num = len(users_dict) + 1
    items_num_list = set()
    for i in users_dict.items():
        for j in i[1]:
            items_num_list.add(j[0])
    items_num = len(items_num_list) + 1

    eval_users_num = int(users_num * 0.2)
    eval_items_num = items_num
    print(eval_users_num, eval_items_num)

    eval_users_dict = {k: users_dict[k] for k in range(users_num - eval_users_num, users_num)}
    eval_users_history_lens = users_history_lens[-eval_users_num:]

    # eval_users_num = int(users_num * 0.8)
    # eval_items_num = items_num
    # print(eval_users_num, eval_items_num)
    #
    # eval_users_dict = {k: users_dict[k] for k in range(1,eval_users_num+1)}
    # eval_users_history_lens = users_history_lens[:eval_users_num]
    print(len(eval_users_dict), len(eval_users_history_lens))

    print('begin evaluation')
    tf.keras.backend.set_floatx('float64')

    state_representation = DRRAveStateRepresentation(EMBEDDING_DIM)
    state_representation([np.zeros((1, EMBEDDING_DIM,)), np.zeros((1, STATE_SIZE, EMBEDDING_DIM))])

    # embedding net
    emb_net = UserMovieEmbedding(users_num, items_num, EMBEDDING_DIM)
    emb_net([np.zeros((1,)), np.zeros((1,))])
    emb_net.load_weights('./weights/user_movie_embedding_100k.h5')

    env = OfflineEnv(eval_users_dict, users_history_lens, items_num_list, STATE_SIZE, state_representation,
                     emb_net,
                     seed=SEED)
    # algo = DDPG(action_shape=EMBEDDING_DIM, seed=SEED, gamma=0.9, ac_lr=1e-3, cr_lr=1e-3, tau=0.001,
    #             state_size=STATE_SIZE, embedding_dim=EMBEDDING_DIM, actor_hidden_dim=128, critic_hidden_dim=128,
    #             replay_memory_size=1000000, batch_size=64, is_test=False, max_episode_num=NUMBER_STEPS, epsilon=1.,
    #             std=1.5)
    sum_precision = 0
    algo =  PPO2(state_dim = EMBEDDING_DIM*3, action_shape=EMBEDDING_DIM, seed = SEED, env = env, gamma=0.99, clip_ratio=0.2, lamdb=0.95,
                 actor_lr=3e-4, critic_lr=1e-3,update_interval = 63,update_epochs = 5)
    sum_ndcg = 0
    TOP_K = 10
    #
    # 4000 step is airl model only 0.3
    recommender = Trainer(env, env, algo, users_num, items_num,
                          use_wandb=False, load = True, load_step=20000,
                          load_memory=False, seed=0, num_steps=NUMBER_STEPS,
                          eval_interval=500, num_eval_episodes=5, num_steps_before_train=5000,
                          model_dir='./model', memory_dir='./memory', top_k=TOP_K)

    predictions = []
    for user_id in eval_users_dict.keys():
        precision, ndcg = recommender.evaluate_final(top_k=TOP_K)
        predictions.append(precision)
        sum_precision += precision
        sum_ndcg += ndcg
    labels = np.ones_like(predictions)
    m = tf.keras.metrics.RootMeanSquaredError()
    if TOP_K:
        print(
            f'precision@{TOP_K} : {sum_precision / len(eval_users_dict)}, ndcg@{TOP_K} : {sum_ndcg / len(eval_users_dict)}',
            f'log loss@{TOP_K}:{tf.compat.v1.losses.log_loss(labels,predictions)}',
            f'RMSE@{TOP_K}:{m(labels,predictions)}')
    else:
        print(
            f'precision: {sum_precision / len(eval_users_dict)}',
        f'log loss:{tf.compat.v1.losses.log_loss(labels,predictions)}',
        f'RMSE:{m(labels,predictions)}')
