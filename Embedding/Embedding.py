import pandas as pd
import numpy as np
import json
from pyspark.sql import SparkSession
import os

spark = SparkSession \
    .builder \
    .appName("PySpark ALS") \
    .getOrCreate()

sc = spark.sparkContext


from pyspark.sql import types as T
from pyspark.ml.recommendation import ALS

customSchema = T.StructType([
    T.StructField("userId", T.IntegerType(), True),
    T.StructField("movieId", T.IntegerType(), True),
    T.StructField("rating", T.FloatType(), True),
    T.StructField("timestamp", T.LongType(), True),
])


def addSamplelabel(ratingsamples):
    # if rating > 3 label 1 as recommend, 0 as not recommend.
    ratingsamples['label'] = (ratingsamples['rating']>3).astype(int)
    return ratingsamples


ROOT_DIR = '../data/'
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/')

ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'users.dat'), 'r').readlines()]
movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'movies.dat'),encoding='latin-1').readlines()]
ratings_df = pd.DataFrame(ratings_list, columns = ['userId', 'movieId', 'rating', 'timestamp'], dtype = np.uint32)
movies_df = pd.DataFrame(movies_list, columns = ['movieId', 'title', 'genres'])

# ratings = pd.read_csv(r"..\data\ratings.csv")
# movies = pd.read_csv(r"..\data\movies.csv")

# ratings = addSamplelabel(ratings)
# ratings = ratings[ratings['label'] == 1]
#ratings_df.to_csv(r"ratings_embedding_1m.csv",index = False)

data = spark.read.csv(
    r"ratings_embedding_1m.csv",
    header=True,
    schema=customSchema
)

data.show(5)

als = ALS(
    maxIter=5,
    regParam=0.01,
    rank= 100,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop")
model = als.fit(data)

# # train

#
#
model.userFactors.select("id", "features") \
           .toPandas() \
           .to_csv(r"user_embedding_1m.csv", index=False)

model.itemFactors.select("id", "features") \
           .toPandas() \
           .to_csv(r"item_embedding_1m.csv", index=False)

