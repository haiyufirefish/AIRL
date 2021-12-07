import pandas as pd
import numpy as np

from envs.Offline_env import OfflineEnv

STATE_SIZE = 10

if __name__ == '__main__':

    ratings_df = pd.read_csv("./data/ratings.csv")
    movies_df = pd.read_csv("./data/movies.csv")

    movies_df['movieId'] = movies_df['movieId'].apply(pd.to_numeric)

    print("Data loading complete!")
    print("Data preprocessing...")

    movies_id_to_movies = movies_df.set_index('movieId').T.to_dict('list')
    ratings_df = ratings_df.applymap(int)

    users_dict = np.load("./data/user_dict.npy",allow_pickle=True)
    users_history_lens = np.load("./data/user_hist_len.npy")

    users_num = max(ratings_df["userId"]) + 1
    items_num = max(ratings_df["movieId"]) + 1

    train_users_num = int(users_num * 0.8)
    train_items_num = items_num
    train_users_dict = {k: users_dict.item().get(k) for k in range(1, train_users_num + 1)}
    train_users_history_lens = users_history_lens[:train_users_num]

    print('DONE!')
    env = OfflineEnv(train_users_dict, train_users_history_lens, movies_id_to_movies, STATE_SIZE)