import pandas as pd
import numpy as np
import json

from ddpg import DDPG
from envs.Offline_env import OfflineEnv
from trainer import Trainer
from Embedding_loader import Embedding_loader
from state_representation import AveStateRepresentation
from utils import addSamplelabel


STATE_SIZE = 10
SEED = 0

if __name__ == '__main__':

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


    state = env.reset()
    print(state.size())
    algo = DDPG(state_shape=(1,300),
        action_shape=(1,100),
        device='cpu',seed=SEED)
    #
    recommender = Trainer(env,env,algo,log_dir='./')
    #
    recommender.train()