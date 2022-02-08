import pandas as pd
import numpy as np
import os

def addSamplelabel(ratingsamples):
    # if rating > 3.5 label 1 as recommend, 0 as not recommend.
    ratingsamples['label'] = (ratingsamples['rating']>3.5).astype(int)
    return ratingsamples

if __name__ == '__main__':

    ROOT_DIR = '../data/'
    DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/')

    ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
    ratings = pd.DataFrame(ratings_list, columns = ['userId', 'movieId', 'rating', 'timestamp'], dtype = np.uint32)
    #ratings = addSamplelabel(ratings)
    user_dict = {}
    for index,row in ratings.iterrows():
        #userId, movieId,rating,_,__ = val
        userId = int(row["userId"])
        movieId = int(row["movieId"])
        rating = row["rating"]

        if userId in user_dict:
            user_dict[userId].append((movieId, rating))
        else:
            user_dict[userId] = [(movieId, rating)]


    #np.save("user_dict_1m",user_dict)
    #hist_len = {k:user_dict.item().get}
    #user_dict = {ratings['userId']: [(movieId, rating)] for movieId, rating in ratings['userId']}
    hist_len = {}
    for k,v in user_dict.items():
        hist_len[k] = int(len(v))

    #hist_len = {k:ratings['userId']}
    #np.array(ratings.groupby("userId").userId.count().values.tolist())

    np.save("user_hist_len_1m", hist_len)
    print("data processing done")
