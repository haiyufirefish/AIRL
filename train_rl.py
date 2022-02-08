import pandas as pd
import numpy as np
import json
import torch as torch
import os

from Embedding.embedding_net import EmbeddingNet
from algo import ALGOS
from ddpg import DDPG
from envs.Offline_env import OfflineEnv
from trainer import Trainer
from Embedding_loader import Embedding_loader
from state_representation import AveStateRepresentation
from utils import addSamplelabel


STATE_SIZE = 10
SEED = 0


def create_dataset(ratings, top=None):
    if top is not None:
        ratings.groupby('userId')['rating'].count()

    unique_users = ratings.userId.unique()
    user_to_index = {old: new for new, old in enumerate(unique_users)}
    new_users = ratings.userId.map(user_to_index)

    unique_movies = ratings.movieId.unique()
    movie_to_index = {old: new for new, old in enumerate(unique_movies)}
    new_movies = ratings.movieId.map(movie_to_index)

    n_users = unique_users.shape[0]
    n_movies = unique_movies.shape[0]

    X = pd.DataFrame({'user_id': new_users, 'movie_id': new_movies})
    y = ratings['rating'].astype(np.float32)
    return (n_users, n_movies), (X, y), (user_to_index, movie_to_index)

if __name__ == '__main__':

###################################################
    # ROOT_DIR = os.getcwd()
    # DATA_DIR = os.path.join(ROOT_DIR, './data/ml-1m/')
    #ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'ratings.dat'), 'r').readlines()]
    # users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'users.dat'), 'r').readlines()]
    # movies_list = [i.strip().split("::") for i in
    #                open(os.path.join(DATA_DIR, 'movies.dat'), encoding='latin-1').readlines()]
    #ratings = pd.DataFrame(ratings_list, columns=['userId', 'movieId', 'rating', 'timestamp'], dtype=np.uint32)
    #movies = pd.DataFrame(movies_list, columns=['movieId', 'title', 'genres'])
    #movies['movieId'] = movies['movieId'].apply(pd.to_numeric)
    # ratings['rating'] = ratings['rating'].apply(pd.to_numeric)
    # ratings['userId'] = ratings['userId'].apply(pd.to_numeric)
    #users_df = pd.DataFrame(users_list, columns=['userId', 'gender', 'age', 'occupation', 'zip-code'])
    #(n, m), (X, y), _ = create_dataset(ratings)
############################################################################################
    #embedding
    item_em = pd.read_csv("./Embedding/item_embedding_1m.csv")
    user_em = pd.read_csv("./Embedding/user_embedding_1m.csv")

    item_em['features'] = item_em['features'].map(lambda x: np.array(json.loads(x)))
    user_em['features'] = user_em['features'].map(lambda x: np.array(json.loads(x)))
    Emb_loader = Embedding_loader(user_em, item_em)
    print("Data loading complete!")
    print("Data preprocessing...")

    #movies_id_to_movies = movies.set_index('movieId').T.to_dict('list')
    #ratings_df = ratings.applymap(int)

    users_dict = np.load("./data/user_dict_1m.npy",allow_pickle=True).item()
    users_history_lens = np.load("./data/user_hist_len_1m.npy",allow_pickle=True).item()

    # here also need some modifications
    users_num = len(users_dict) + 1
    items_num_list = item_em['id'].values.tolist()

    train_users_num = int(users_num * 0.8)
    train_items_num = len(items_num_list)
    # train_users_dict = {k: users_dict.item().get(k) for k in range(1, train_users_num + 1)}
    # train_users_history_lens = {k:users_history_lens.item().get(k) for k in range(1,train_users_num+1)}
    train_users_dict = {key:value for key,value in [x for x in users_dict.items()][0:train_users_num]}
    train_users_history_lens = {key:value for key,value in[x for x in users_history_lens.items()][0:train_users_num]}

    Avg_representation = AveStateRepresentation(100)
    env = OfflineEnv(train_users_dict, train_users_history_lens,items_num_list , Avg_representation, STATE_SIZE,Emb_loader,seed = SEED)
    #env = OfflineEnv(train_users_dict, train_users_history_lens,items_num_list ,movies_id_to_movies,Avg_representation, STATE_SIZE,Emb_loader,seed = SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = env.reset()

    algo = DDPG(state_shape=(1,300),
        action_shape=(1,100),
        memory_size = 100,
        device=device,seed=SEED)
    # mode = 'ppo'
    # algo = ALGOS[mode](
    #     #buffer_exp=buffer_exp,
    #     state_shape=(1,300),
    #     action_shape=(1, 100),
    #     device=device,
    #     seed=SEED,
    #     #policy=policy,
    #     #rollout_length= 2048,
    # )
    #
    recommender = Trainer(env,env,algo,log_dir='./',num_steps = 8000)

    recommender.train()