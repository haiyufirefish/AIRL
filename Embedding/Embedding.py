import pandas as pd
import numpy as np
import json
from pyspark.sql import SparkSession

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
    # if rating > 3.5 label 1 as recommend, 0 as not recommend.
    ratingsamples['label'] = (ratingsamples['rating']>3.5).astype(int)
    return ratingsamples

ratings = pd.read_csv(r"C:\Users\LiHaiyu\PycharmProjects\AIRL\data\ratings.csv")
movies = pd.read_csv(r"C:\Users\LiHaiyu\PycharmProjects\AIRL\data\movies.csv")

ratings = addSamplelabel(ratings)
ratings = ratings[ratings['label'] == 1]
ratings.to_csv(r"ratings_embedding.csv",index = False)

ratings_emb = spark.read.csv(
    r"ratings_embedding.csv",
    header=True,
    schema=customSchema
)

als = ALS(
    maxIter=5,
    regParam=0.01,
    rank= 100,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop")

# train
model = als.fit(ratings_emb)


model.userFactors.select("id", "features") \
           .toPandas() \
           .to_csv(r"C:\Users\LiHaiyu\PycharmProjects\AIRL\Embedding\user_embedding.csv", index=False)

model.itemFactors.select("id", "features") \
           .toPandas() \
           .to_csv(r"C:\Users\LiHaiyu\PycharmProjects\AIRL\Embedding\item_embedding.csv", index=False)

