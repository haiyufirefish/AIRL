import pandas as pd
import numpy as np
import os


# ratings Data processing
def data_movie():
    ROOT_DIR = '/'
    DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/')

    ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'ratings.dat'), 'r').readlines()]
    ratings = pd.DataFrame(ratings_list, columns=['userId', 'movieId', 'rating', 'timestamp'], dtype=np.uint32)
    # ratings = addSamplelabel(ratings)
    user_dict = {}
    for index, row in ratings.iterrows():
        # userId, movieId,rating,_,__ = val
        userId = int(row["userId"])
        movieId = int(row["movieId"])
        rating = row["rating"]

        if userId in user_dict:
            user_dict[userId].append((movieId, rating))
        else:
            user_dict[userId] = [(movieId, rating)]

    # np.save("user_dict_1m",user_dict)
    # hist_len = {k:user_dict.item().get}
    # user_dict = {ratings['userId']: [(movieId, rating)] for movieId, rating in ratings['userId']}
    hist_len = {}
    for k, v in user_dict.items():
        hist_len[k] = int(len(v))

    # hist_len = {k:ratings['userId']}
    # np.array(ratings.groupby("userId").userId.count().values.tolist())

    np.save("user_hist_len_1m", hist_len)
    print("data processing done")
def data_movie_k():

    ratings = pd.read_csv('ratings_embedding_100k.csv',header = 0,index_col=False, names=['userId', 'movieId', 'rating', 'timestamp'])
    #ratings.to_csv('ratings_embedding_100k.csv',index=False)
    # data['userId', 'movieId', 'rating','timestamp'] = data['userId'].str.split('\t',expand=True)
    user_dict = {}
    print(ratings.head(5))
    for index, row in ratings.iterrows():
        # userId, movieId,rating,_,__ = val
        userId = int(row["userId"])
        movieId = int(row["movieId"])
        rating = row["rating"]

        if userId in user_dict:
            user_dict[userId].append((movieId, rating))
        else:
            user_dict[userId] = [(movieId, rating)]

    hist_len = []
    for k, v in user_dict.items():
        hist_len.append(len(v))

    np.save("users_hist_len_100k", hist_len)
    print("data processing done")


def data_Yahoo_music():
    ROOT_DIR = '/'
    DATA_DIR = os.path.join(ROOT_DIR, 'dataset/')
    train_data = pd.read_csv(os.path.join(DATA_DIR,'ydata-ymusic-rating-study-v1_0-train.txt'),delimiter = "\t", header=None,
                             index_col = False)

    test_data = pd.read_csv(os.path.join(DATA_DIR, 'ydata-ymusic-rating-study-v1_0-test.txt'), delimiter="\t",
                             header=None,
                             index_col=False)

    train_data.columns = ['userId','musicId','rating']
    test_data.columns = ['userId','musicId','rating']
    ########################################################
    # bigdata = train_data.append(test_data, ignore_index=True)
    # bigdata.to_csv('Yahoo_music.csv',index=False)
    ##############################################################
    user_dict = {}
    for index, row in train_data.iterrows():
        # userId, movieId,rating,_,__ = val
        userId = int(row["userId"])
        musicId = int(row["musicId"])
        rating = row["rating"]

        if userId in user_dict:
            user_dict[userId].append((musicId, rating))
        else:
            user_dict[userId] = [(musicId, rating)]
    np.save("user_dict_Yahoo_music_train", user_dict)

    hist_len = {}
    for k, v in user_dict.items():
        hist_len[k] = int(len(v))
    np.save("user_hist_Yahoo_music_train", hist_len)
    print('train data done!')

    user_dict = {}
    for index, row in test_data.iterrows():
        # userId, movieId,rating,_,__ = val
        userId = int(row["userId"])
        musicId = int(row["musicId"])
        rating = row["rating"]

        if userId in user_dict:
            user_dict[userId].append((musicId, rating))
        else:
            user_dict[userId] = [(musicId, rating)]
    np.save("user_dict_Yahoo_music_test", user_dict)

    hist_len = {}
    for k, v in user_dict.items():
        hist_len[k] = int(len(v))
    np.save("user_hist_Yahoo_music_test", hist_len)
    print('test data done!')


def data_Jester():
    ROOT_DIR = '/'
    DATA_DIR = os.path.join(ROOT_DIR, 'JesterDataset3/')
    data = pd.read_excel(os.path.join(DATA_DIR, 'FINAL_jester_2006-15.xls'))
    data = data.T
    data.colums = ['usedId','rating']
    data = data[data['rating']<99]
    print(data.head(5))


def data_100k_100_users():
    ROOT_DIR = './'
    DATA_DIR = os.path.join(ROOT_DIR,'ratings.csv')
    ratings_df = pd.read_csv(DATA_DIR, header=0, index_col=False,
                          names=['userId', 'movieId', 'rating', 'timestamp'])

    ratings_df = ratings_df.sort_values(by='timestamp', ascending=True)
    users_dict = {user : [] for user in set(ratings_df["userId"])}

    ratings_df_gen = ratings_df.iterrows()
    users_dict_for_history_len = {user : [] for user in set(ratings_df["userId"])}
    for data in ratings_df_gen:
        users_dict[data[1]['userId']].append((data[1]['movieId'], data[1]['rating']))
        if data[1]['rating'] >= 4:
            users_dict_for_history_len[data[1]['userId']].append((data[1]['movieId'], data[1]['rating']))
    users_history_lens = [len(users_dict_for_history_len[u]) for u in set(ratings_df["userId"])]
    np.save("./user_dict_100k.npy", users_dict)
    np.save("./users_histroy_len_100k.npy", users_history_lens)


def deal_item():
    movie_df = pd.read_csv("./ml-100k/u.item", sep="|", encoding='latin-1', header=None)
    movie_df.columns = ['movieId', 'title', 'release date', 'video release date', 'IMDb URL', 'unknown',
                        'Action',
                        'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir',
                        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    gernes = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir',
              'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    # print(movie_df[gernes])
    movie_df['genres'] = 'n'
    for ind in movie_df.index:
        for gern in gernes:
            if movie_df[gern][ind] == 1:
                if movie_df['genres'][ind] == 'n':
                    movie_df['genres'][ind] = gern
                else:
                    movie_df['genres'][ind] += '|' + gern

    print(movie_df['genres'])

    df_save = movie_df[['movieId', 'title', 'genres']]
    df_save.to_csv('movie_100k.csv', index=False)


if __name__ == '__main__':
    # data_Yahoo_music()
    # data_movie_k()
    data_100k_100_users()






