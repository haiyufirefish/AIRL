import pandas as pd
import numpy as np
import os
def data_movie():
    ROOT_DIR = '../data/'
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

def data_Yahoo_music():
    ROOT_DIR = '../data/'
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
    ROOT_DIR = '../data/'
    DATA_DIR = os.path.join(ROOT_DIR, 'JesterDataset3/')
    data = pd.read_excel(os.path.join(DATA_DIR, 'FINAL_jester_2006-15.xls'))
    data = data.T
    data.colums = ['usedId','rating']
    data = data[data['rating']<99]
    print(data.head(5))
if __name__ == '__main__':
    #data_Yahoo_music()
    data_Jester()

