import tensorflow as tf
import pandas as pd
import numpy as np

from tqdm import tqdm


class UserMovieEmbedding(tf.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(UserMovieEmbedding, self).__init__()
        self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users,
                                                     output_dim=embedding_dim)
        self.m_embedding = tf.keras.layers.Embedding(name='item_embedding', input_dim=len_movies,
                                                     output_dim=embedding_dim)
        self.m_u_merge = tf.keras.layers.Dot(name='item_user_dot', normalize=False, axes=1)
        self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        memb = self.m_embedding(x[1])
        m_u = self.m_u_merge([memb, uemb])
        return self.m_u_fc(m_u)


if __name__ == '__main__':
    import os

    genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
              'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    ROOT_DIR = os.getcwd()
    DATA_DIR = os.path.join(ROOT_DIR, './data/ml-100k/')
    STATE_SIZE = 10

    ratings = pd.read_csv('data/ratings.csv', header=0, index_col=False,
                          names=['userId', 'movieId', 'rating', 'timestamp'])

    movie_df = pd.read_csv(os.path.join(DATA_DIR, "u.item"), sep="|", encoding='latin-1', header=None)
    movie_df.columns = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown',
                        'Action',
                        'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir',
                        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movie_df['MovieID'] = movie_df['movie id'].apply(pd.to_numeric)
    movies_genres_df = movie_df[['MovieID']]


    MAX_EPOCH = 500
    INIT_USER_BATCH_SIZE = 64
    FINAL_USER_BATCH_SIZE = 1024

    user_movie_rating_df = ratings[['userId', 'movieId', 'rating']]

    user_movie_rating_df = user_movie_rating_df.apply(np.int32)
    LEN_MOVIES = max(movies_genres_df["MovieID"]) + 1
    LEN_USERS = max(user_movie_rating_df['userId']) + 1
    EMBEDDING_SIZE = 100

    test_model = UserMovieEmbedding(LEN_USERS, LEN_MOVIES, EMBEDDING_SIZE)

    test_model([np.zeros((1)), np.zeros((1))])
    print(test_model.summary())

    modified_user_movie_rating_df = user_movie_rating_df.apply(np.int32)
    index_names = modified_user_movie_rating_df[modified_user_movie_rating_df['rating'] < 4].index
    modified_user_movie_rating_df = modified_user_movie_rating_df.drop(index_names)
    modified_user_movie_rating_df = modified_user_movie_rating_df.drop('rating', axis=1)
    u_m_pairs = modified_user_movie_rating_df.to_numpy()
    u_m_pairs[:5]

    positive_user_movie_dict = {u: [] for u in range(1, max(modified_user_movie_rating_df['userId']) + 1)}
    for data in modified_user_movie_rating_df.iterrows():
        positive_user_movie_dict[data[1][0]].append(data[1][1])
    positive_user_movie_dict[1]


    def generate_user_movie_batch(positive_pairs, batch_size, negative_ratio=0.5):
        batch = np.zeros((batch_size, 3))
        positive_batch_size = batch_size - int(batch_size * negative_ratio)
        max_user_id = max(modified_user_movie_rating_df['userId']) + 1
        max_movie_id = max(modified_user_movie_rating_df['movieId']) + 1

        while True:
            idx = np.random.choice(len(positive_pairs), positive_batch_size)
            data = positive_pairs[idx]
            for i, d in enumerate(data):
                batch[i] = (d[0], d[1], 1)

            while i + 1 < batch_size:
                u = np.random.randint(1, max_user_id)
                m = np.random.randint(1, max_movie_id)
                if m not in positive_user_movie_dict[u]:
                    i += 1
                    batch[i] = (u, m, 0)

            np.random.shuffle(batch)
            yield batch[:, 0], batch[:, 1], batch[:, 2]


    optimizer = tf.keras.optimizers.Adam()
    # loss
    bce = tf.keras.losses.BinaryCrossentropy()

    test_train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')


    @tf.function
    def test_train_step(test_inputs, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = test_model(test_inputs, training=True)
            loss = bce(labels, predictions)
        gradients = tape.gradient(loss, test_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, test_model.trainable_variables))

        test_train_loss(loss)
        test_train_accuracy(labels, predictions)


    test_losses = []
    MAX_EPOCH = 500
    for epoch in tqdm(range(MAX_EPOCH)):

        batch_size = INIT_USER_BATCH_SIZE * (epoch + 1)
        if batch_size > FINAL_USER_BATCH_SIZE:
            batch_size = FINAL_USER_BATCH_SIZE
        test_generator = generate_user_movie_batch(u_m_pairs, batch_size)

        for step in range(len(user_movie_rating_df) // batch_size):
            # embedding layer update
            u_batch, m_batch, u_m_label_batch = next(test_generator)
            test_train_step([u_batch, m_batch], u_m_label_batch)

            print(
                f'{epoch} epoch, Batch size : {batch_size}, {step} steps, Loss: {test_train_loss.result():0.4f}, Accuracy: {test_train_accuracy.result() * 100:0.1f}')

        test_losses.append(test_train_loss.result())

        import matplotlib.pyplot as plt

        plt.plot(test_losses)

        test_model.save_weights('./user_movie_embedding_100k.h5')

