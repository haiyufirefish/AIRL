import os
import argparse
import torch


from algo import DDPGExpert
from utils import collect_demo
from envs import Offline_env
import json

def run(args):

    env = Offline_env

    # algo = SACExpert(
    #     state_shape=env.observation_space.shape,
    #     action_shape=env.action_space.shape,
    #     device=torch.device("cuda" if args.cuda else "cpu"),
    #     path=args.weight
    # )
    algo = DDPGExpert(
        state_shape=(1,300),
        action_shape=(1,100),
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=args.weight
    )

    buffer = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed
    )
    buffer.save(os.path.join(
        'buffers',
        args.env_id,
        f'size{args.buffer_size}_std{args.std}_prand{args.p_rand}.pth'
    ))

#########################################################################################
import pandas as pd
from utils import addSamplelabel
from Embedding_loader import Embedding_loader
import numpy as np
from envs.Offline_env import OfflineEnv
from state_representation import AveStateRepresentation

def main():
    STATE_SIZE = 10
    ratings_df = pd.read_csv("./data/ratings.csv")
    movies_df = pd.read_csv("./data/movies.csv")
    ratings_df = addSamplelabel(ratings_df)
    movies_df['movieId'] = movies_df['movieId'].apply(pd.to_numeric)

    # embedding
    item_em = pd.read_csv("./Embedding/item_embedding.csv")
    user_em = pd.read_csv("./Embedding/user_embedding.csv")

    item_em['features'] = item_em['features'].map(lambda x: np.array(json.loads(x)))
    user_em['features'] = user_em['features'].map(lambda x: np.array(json.loads(x)))
    Emb_loader = Embedding_loader(user_em, item_em)
    print("Data loading complete!")
    print("Data preprocessing...")

    movies_id_to_movies = movies_df.set_index('movieId').T.to_dict('list')
    ratings_df = ratings_df.applymap(int)

    users_dict = np.load("./data/user_dict.npy",allow_pickle=True).item()
    users_history_lens = np.load("./data/user_hist_len.npy",allow_pickle=True).item()

    # here also need some modifications
    users_num = len(users_dict) + 1
    items_num_list = item_em['id'].tolist()

    train_users_num = int(users_num * 0.8)
    train_items_num = len(items_num_list)+1
    # train_users_dict = {k: users_dict.item().get(k) for k in range(1, train_users_num + 1)}
    # train_users_history_lens = {k:users_history_lens.item().get(k) for k in range(1,train_users_num+1)}
    train_users_dict = {key:value for key,value in [x for x in users_dict.items()][0:train_users_num]}
    train_users_history_lens = {key:value for key,value in[x for x in users_history_lens.items()][0:train_users_num]}

    Avg_representation = AveStateRepresentation(100)
    env = OfflineEnv(train_users_dict, train_users_history_lens,items_num_list ,movies_id_to_movies,Avg_representation, STATE_SIZE,Emb_loader)

    algo = DDPGExpert(
        state_shape=(1, 300),
        action_shape=(1, 100),
        device=torch.device( "cpu"),
        path='./model/step_1000/ddpg_actor.pth'
    )

    buffer = collect_demo(
        env=env,
        algo=algo,
        buffer_size=10**2,
        device=torch.device("cpu"),
    )

    buffer.save(os.path.join(
        'buffers',
        "movie_env",
        f'size{10**2}_std{0}_prand{0}.pth'
    ))



if __name__ == '__main__':
    # p = argparse.ArgumentParser()
    # p.add_argument('--weight', type=str, required=True)
    # p.add_argument('--env_id', type=str, default='Hopper-v3')
    # p.add_argument('--buffer_size', type=int, default=10**6)
    # p.add_argument('--std', type=float, default=0.0)
    # p.add_argument('--p_rand', type=float, default=0.0)
    # p.add_argument('--cuda', action='store_true')
    # p.add_argument('--seed', type=int, default=0)
    # args = p.parse_args()
    # run(args)
    main()
