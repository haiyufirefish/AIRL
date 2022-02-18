import pandas as pd
import numpy as np
import json
from pyspark.sql import SparkSession
import os
from pyspark.sql import types as T
from pyspark.ml.recommendation import ALS
spark = SparkSession \
    .builder \
    .appName("PySpark ALS") \
    .getOrCreate()

sc = spark.sparkContext

def movie_1m_embedding():
    customSchema = T.StructType([
        T.StructField("userId", T.IntegerType(), True),
        T.StructField("movieId", T.IntegerType(), True),
        T.StructField("rating", T.FloatType(), True),
        T.StructField("timestamp", T.LongType(), True),
    ])

    ROOT_DIR = '../data/'
    DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/')

    data = spark.read.csv(
        r"ratings_embedding_1m.csv",
        header=True,
        schema=customSchema
    )

    data.show(5)

    als = ALS(
        maxIter=5,
        regParam=0.01,
        rank=100,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop")
    model = als.fit(data)

    model.userFactors.select("id", "features") \
        .toPandas() \
        .to_csv(r"user_embedding_1m.csv", index=False)

    model.itemFactors.select("id", "features") \
        .toPandas() \
        .to_csv(r"item_embedding_1m.csv", index=False)

def jester_embedding():
    customSchema = T.StructType([
        T.StructField("userId", T.IntegerType(), True),
        T.StructField("jokeId", T.IntegerType(), True),
        T.StructField("rating", T.FloatType(), True),
    ])

    ROOT_DIR = '../data/'
    DATA_DIR = os.path.join(ROOT_DIR, 'jester_rating_sec.csv')

    data = spark.read.csv(
        DATA_DIR,
        header=True,
        schema=customSchema
    )
    # data.show(10)
    als = ALS(
        maxIter=5,
        regParam=0.01,
        rank=100,
        userCol="userId",
        itemCol="jokeId",
        ratingCol="rating",
        coldStartStrategy="drop")
    model = als.fit(data)
    print('processing fit done')
    model.userFactors.select("id", "features") \
        .toPandas() \
        .to_csv(r"user_embedding_jester.csv", index=False)

    model.itemFactors.select("id", "features") \
        .toPandas() \
        .to_csv(r"item_embedding_jester.csv", index=False)


def Yahoo_music_embedding():
    customSchema = T.StructType([
        T.StructField("userId", T.IntegerType(), True),
        T.StructField("musicId", T.IntegerType(), True),
        T.StructField("rating", T.FloatType(), True),
    ])

    ROOT_DIR = '../data/'
    DATA_DIR = os.path.join(ROOT_DIR, 'Yahoo_music.csv')

    data = spark.read.csv(
        DATA_DIR,
        header=True,
        schema=customSchema
    )
    als = ALS(
        maxIter=5,
        regParam=0.01,
        rank=100,
        userCol="userId",
        itemCol="musicId",
        ratingCol="rating",
        coldStartStrategy="drop")
    model = als.fit(data)

    model.userFactors.select("id", "features") \
        .toPandas() \
        .to_csv(r"user_embedding_Yahoo_music.csv", index=False)

    model.itemFactors.select("id", "features") \
        .toPandas() \
        .to_csv(r"item_embedding_Yahoo_music.csv", index=False)
    print("em processing done!")
if __name__ == '__main__':
    jester_embedding()
    #Yahoo_music_embedding()