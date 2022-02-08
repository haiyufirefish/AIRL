import json
import os
from datetime import datetime
import torch as torch

from Embedding_loader import Embedding_loader
from buffer import SerializedBuffer
from algo import ALGOS

from ddpg import DDPG
from envs.Offline_env import OfflineEnv
from state_representation import AveStateRepresentation
from trainer import Trainer
import numpy as np
import pandas as pd

from utils import addSamplelabel


def main():
    # parser = ArgumentParser('parameters')
    # args = parser.parse_args()
    # parser = ConfigParser()
    # parser.read('config.ini')
    # agent_args = Dict(parser, args.algo)
    STATE_SIZE = 10
    SEED = 0
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

    users_dict = np.load("./data/user_dict.npy", allow_pickle=True).item()
    users_history_lens = np.load("./data/user_hist_len.npy", allow_pickle=True).item()

    # here also need some modifications
    users_num = len(users_dict) + 1
    items_num_list = item_em['id'].tolist()

    train_users_num = int(users_num * 0.8)
    train_items_num = len(items_num_list) + 1
    # train_users_dict = {k: users_dict.item().get(k) for k in range(1, train_users_num + 1)}
    # train_users_history_lens = {k:users_history_lens.item().get(k) for k in range(1,train_users_num+1)}
    train_users_dict = {key: value for key, value in [x for x in users_dict.items()][0:train_users_num]}
    train_users_history_lens = {key: value for key, value in [x for x in users_history_lens.items()][0:train_users_num]}

    Avg_representation = AveStateRepresentation(100)
    env = OfflineEnv(train_users_dict, train_users_history_lens, items_num_list, movies_id_to_movies,
                     Avg_representation, STATE_SIZE, Emb_loader, seed=SEED)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    buffer_exp = SerializedBuffer(
        path='./buffers/movie_env/ddpg_expert_trajectories_size_100.pth',
        device=torch.device(device)
    )
    policy = DDPG(state_shape=(1,300),
        action_shape=(1,100),
        memory_size= 100,
        device=device,seed=SEED)
    #state only
    mode = 'airl'
    algo = ALGOS[mode](
        buffer_exp=buffer_exp,
        state_shape=(1, 400),
        action_shape=(1, 100),
        device=device,
        seed=SEED,
        policy=policy,
        rollout_length=10,
        mode=mode,
        state_only=False
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir='./'

    trainer = Trainer(
        env=env,
        env_test=env,
        algo=algo,
        log_dir=log_dir,
        seed=SEED,
        num_steps=200,
        eval_interval=5,
        num_steps_before_train=50
    )
    trainer.train_imitation()

if __name__ == '__main__':
    main()
