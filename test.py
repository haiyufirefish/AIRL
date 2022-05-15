
from embedding import UserMovieEmbedding
##############################################
# c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
# b = c
#
# print(tf.equal(b,c)) # return tensor
# print(b is c) # check the object equal
######################################
import numpy as np
# users_num = 944
# EMBEDDING_DIM = 100
# items_num = 1683
# emb_net = UserMovieEmbedding(users_num,items_num,EMBEDDING_DIM)
# emb_net([np.zeros((1,)), np.zeros((1,))])
# emb_net.load_weights('./weights/user_movie_embedding_100k.h5')

#####################################################
# users_dict = np.load('data/user_dict_100k.npy',allow_pickle = True).item()
# items_num = 0
# items_num_list = set()
# for i in users_dict.items():
#     for j in i[1]:
#         items_num_list.add(j[0])
# items_num += 1
# items_num_list = list(items_num_list)
users_history_lens = np.load("./data/users_histroy_len_100k.npy")
u_dic = np.load('./data/user_dict_100k.npy',allow_pickle=True).item()
# for key in [362, 400, 34, 502, 88, 166, 40, 317, 604, 93, 364, 143, 732, 36, 681, 50, 511, 47, 281, 443]:
#     good_ones = []
#     for items in u_dic[key][11:]:
#         if items[1] >3:
#             good_ones.append(items[0])
#     print(good_ones)
#print(np.average(users_history_lens))
#users_history_lenssss = np.load("data/users_histroy_len.npy")
#
# #users_history_lenssss[:10]
# users_history_lens[:10]
# u_dic = np.load('./data/user_dict_100k.npy',allow_pickle=True).item()
###########################################################
# train_users_dict = {k: users_dict.item().get(k) for k in range(1, train_users_num + 1)}
    # train_users_history_lens = {k:users_history_lens.item().get(k) for k in range(1,train_users_num+1)}
    #train_user_ids = [196,186,22,244,166,298,115,253,305,6]]
    #

    #,298,115,253,305,6]
    # train_users_history_lens = {key: users_history_lens[key] for key in train_user_ids}
    # train_users_dict = {key:users_dict[key][:14] for key in train_user_ids}
    # items_num_list = []
    # for k,v in train_users_dict.items():
    #     for vv in v:
    #         items_num_list.append(vv[0])
############################################################################
import os
# model_dir = './model'
# if not os.path.exists(model_dir):
#     print('not exist')

#########################################################################
# import tensorflow as tf
# samples = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 5)
# print(samples)
#################################
#
import tensorflow as tf

class Discriminator(tf.keras.Model):
  """Implementation of a discriminator network."""

  def __init__(self, input_dim):
      """Initializes a discriminator.
      Args:
         input_dim: size of the input space.
      """
      super(Discriminator, self).__init__()
      kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)

      self.main = tf.keras.Sequential([
          tf.keras.layers.Dense(
              units=256,
              input_shape=(input_dim,),
              activation='tanh',
              kernel_initializer=kernel_init),
          tf.keras.layers.Dense(
              units=256, activation='tanh', kernel_initializer=kernel_init),
          tf.keras.layers.Dense(units=1, kernel_initializer=kernel_init)
      ])
  def call(self, inputs):
    """Performs a forward pass given the inputs.
    Args:
      inputs: a batch of observations (tfe.Variable).
    Returns:
      Values of observations.
    """
    return self.main(inputs)

# input_dim = 100
# discriminator = Discriminator(input_dim)
# discriminator(np.zeros((1,input_dim)))
# with tf.name_scope('discriminator'):
#     discriminator_optimizer = tf.keras.optimizers.Adam()
#     discriminator_optimizer._create_slots(discriminator.variables)
#     print("ok")
# print(discriminator.summary())
# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
# inputs = tf.concat([t1, t2], -1)
# alpha = np.random.uniform(size=(inputs.get_shape()[0], 1))
# alpha = tf.Variable(alpha.astype('float32'))
# step = tf.Variable(
#           0, dtype=tf.int64, name='step')
# t2 = [[7, 8, 9], [10, 11, 12]]
# t = tf.constant(t2)
# print(t.shape)
# items = [[245,312,689,350,332],[749, 321, 895, 304, 343, 323],[990, 329, 324, 898, 245, 299, 899],[271, 342, 895, 343, 338, 323, 358, 687],
#              [354, 690, 898, 886, 881, 261, 1191, 326],[300, 322, 984, 343, 894],[268, 337, 321, 345, 294, 754],
#              [355, 350, 264, 748, 260],[164, 637, 448, 447, 670, 567],[815, 476, 235],[288, 990, 678, 948, 1048],[271, 326, 333, 322, 325, 347],
#              [245],[339, 882, 878, 883, 358, 261, 885, 1026],[990, 690, 539, 292, 294, 289],[508, 1084, 253, 1008, 544, 123],[682, 895, 271, 260, 908, 1527, 294],
#              [307, 303, 306, 301, 327],[538, 332, 877, 300, 748, 304, 322],[343, 309, 748, 12]]
# from utils import get_label_batch
#
# ids = [362, 400, 34, 502, 88, 166, 40, 317, 604, 93, 364, 143]
# dict = np.load('./demostrations_full.npy', allow_pickle=True).item()
# # a,b,c = get_label_batch(dict,ids,100)
# print(len(dict))

# len = 100
# for i in range(100):
#     batch_idxs = np.random.randint(len, size=32)
#     print(batch_idxs)

# import urllib.request
# i = 1
# url = 'https://images-na.ssl-images-amazon.com/images/M/MV5BMjkwNDEyODY4OF5BMl5BanBnXkFtZTcwODQyNjUyMQ@@..jpg'
# urllib.request.urlretrieve("{}".format(url), "{}.jpg".format(3))

#
