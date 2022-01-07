import pandas as pd
import numpy as np

def addSamplelabel(ratingsamples):
    # if rating > 3.5 label 1 as recommend, 0 as not recommend.
    ratingsamples['label'] = (ratingsamples['rating']>3.5).astype(int)
    return ratingsamples

if __name__ == '__main__':

    ratings = pd.read_csv(r"ratings.csv")
    ratings = addSamplelabel(ratings)

    # user_dict = {ratings['userId']:[(movieId,rating)] for movieId,rating in ratings['userId']}
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


    np.save("user_dict",user_dict)
    #hist_len = {k:user_dict.item().get}
    hist_len = {}
    for k,v in user_dict.items():

        hist_len[k] = int(len(v))

    #hist_len = {k:ratings['userId']}
    #np.array(ratings.groupby("userId").userId.count().values.tolist())

    np.save("user_hist_len", hist_len)
    print("data processing done")
