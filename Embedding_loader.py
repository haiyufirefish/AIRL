#import pandas as pd
import numpy as np
#import json
#from state_representation import AveStateRepresentation


# item_em = pd.read_csv("Pytorch_models/src/com/item_embedding.csv")
# user_em = pd.read_csv("Pytorch_models/src/com/user_embedding.csv")
#
# item_em['features'] = item_em['features'].map(lambda x : np.array(json.loads(x)))
# user_em['features'] = user_em['features'].map(lambda x : np.array(json.loads(x)))
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

from Embedding.embedding_net import EmbeddingNet
import os
import pandas as pd
import torch


# ROOT_DIR = os.getcwd()
# DATA_DIR = os.path.join(ROOT_DIR, './data/ml-1m/')
# ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'ratings.dat'), 'r').readlines()]
# users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'users.dat'), 'r').readlines()]
# movies_list = [i.strip().split("::") for i in
#                open(os.path.join(DATA_DIR, 'movies.dat'), encoding='latin-1').readlines()]
# ratings = pd.DataFrame(ratings_list, columns=['userId', 'movieId', 'rating', 'timestamp'], dtype=np.uint32)
# movies = pd.DataFrame(movies_list, columns=['movieId', 'title', 'genres'])
# movies['movieId'] = movies['movieId'].apply(pd.to_numeric)
# ratings['rating'] = ratings['rating'].apply(pd.to_numeric)
# ratings['userId'] = ratings['userId'].apply(pd.to_numeric)
# users_df = pd.DataFrame(users_list, columns=['userId', 'gender', 'age', 'occupation', 'zip-code'])
#
# (n, m), (X, y), (user_to_index, movie_to_index) = create_dataset(ratings)


# em_net = EmbeddingNet(
#     n_users=n, n_movies=m,
#     n_factors=100, hidden=[100, 200, 300],
#     embedding_dropout=0.05, dropouts=[0.5, 0.5, 0.25])
# #3706 m 6040 n
# em_net.load_state_dict(torch.load('./Embedding/user_item_embedding.pth'))
# items = [3362, 1301, 3471, 1580, 1270, 2028, 1953, 2130, 1387, 3576]
# items_eb = em_net.m.weight[items]



class Embedding_loader:
    def __init__(self, user_em, item_em):
        self.user_em = user_em
        self.item_em = item_em

    def get_item_em(self,item_ids):

        return np.array([self.item_em[self.item_em["id"] == id].iloc[0, 1] for id in item_ids])

    def get_user_em(self,id):

        return np.array([self.user_em[self.user_em["id"] == id].iloc[0, 1]])

    def check_item_em_(self, id):

        return id in self.item_em['id'] and not self.item_em[self.item_em['id'] == id].empty

    def check_user_em(self, id):

        return id in self.user_em['id'] and not self.user_em[self.user_em["id"] == id].empty






