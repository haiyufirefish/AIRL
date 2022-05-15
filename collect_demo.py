import tensorflow as tf
import numpy as np
from tqdm import tqdm
from enum import Enum

from embedding import UserMovieEmbedding
from env import OfflineEnv
from state import DRRAveStateRepresentation
from ddpg_ import DDPG

EMBEDDING_DIM = 100
SEED = 12
STATE_SIZE = 10
NUMBER_STEPS = 8000

class Mask(Enum):
  ABSORBING = -1.0
  DONE = 0.0
  NOT_DONE = 1.0


def collect_demo_fixed_users(env,algo,seed = SEED):
    np.random.seed(seed)
    tf.random.set_seed(
        seed
    )
    ids = [19, 78, 86, 206, 260, 273, 335, 341, 375, 515, 575, 750]
    dict = np.load('./demostration.npy', allow_pickle=True).item()
    for _ in tqdm(range(2000)):
        for ind in range(len(ids)):
            state = env.get_fixed_user_state(ids[ind])
            done_ = False
            while not done_:
                action = algo.explore(state)
                next_state, reward, done_, _ = env.step(action, top_k=False)

                if next_state is not state:
                   # print("here", reward)
                   # mask = False if t == env.max_episodes else done_
                    state_ = next_state.numpy()

                    if env.user not in dict:
                        dict[env.user] = [(state_,action.numpy(),done_)]
                    else:
                        if len(dict[env.user]) > 9:
                            continue
                        dict[env.user].append([state_,action.numpy(),done_])
    print("len  ", len(dict))
    # print("they are ",dict)
    np.save('demostration', dict)

def collect_demo(env, algo, seed = SEED):

    np.random.seed(seed)
    tf.random.set_seed(
        seed
    )

    dict = np.load('./demostration.npy',allow_pickle=True).item()
    #dict = {}
    nums = []
    sum = 0
    for __ in tqdm(range(len(env.available_users)*20)):
        state, _, done_ = env.reset()
        t = 0
        if env.user not in dict:
            nums.append(env.user)
        while not done_:
           # print('user is ', env.user)
            action = algo.explore(state)
            next_state, reward, done_, _ = env.step(action, top_k=False)
            t += 1
            if next_state is not state:
               # print("here", reward)
               # mask = False if t == env.max_episodes else done_
                state_ = next_state.numpy()

                if env.user not in dict:
                    dict[env.user] = [(state_,action.numpy(),done_)]
                    sum += 1
                else:
                    if len(dict[env.user]) > 9:
                        continue
                    dict[env.user].append([state_,action.numpy(),done_])
                    sum += 1
        print("------")
    print("len  ",len(dict))
   # print("they are ",dict)
    np.save('demostration',dict)

def add_fixed_user_action(env, emb_net,seed = SEED):
    dict = np.load('./demostration_.npy', allow_pickle=True).item()

    ids = [362, 400, 34, 502, 88, 166, 40, 317, 604, 93, 364, 143, 732, 36, 681, 50, 511, 47, 281, 443]
    items = [[245,312,689,350,332],[749, 321, 895, 304, 343, 323],[990, 329, 324, 898, 245, 299, 899],[271, 342, 895, 343, 338, 323, 358, 687],
             [354, 690, 898, 886, 881, 261, 1191, 326],[300, 322, 984, 343, 894],[268, 337, 321, 345, 294, 754],
             [355, 350, 264, 748, 260],[164, 637, 448, 447, 670, 567],[815, 476, 235],[288, 990, 678, 948, 1048],[271, 326, 333, 322, 325, 347],
             [245],[339, 882, 878, 883, 358, 261, 885, 1026],[990, 690, 539, 292, 294, 289],[508, 1084, 253, 1008, 544, 123],[682, 895, 271, 260, 908, 1527, 294],
             [307, 303, 306, 301, 327],[538, 332, 877, 300, 748, 304, 322],[343, 309, 748, 12]]

    for id,item in zip(ids,items):
        actions = emb_net.get_layer('item_embedding')(np.array(item))
        state_ = env.get_fixed_user_state(id).numpy()
        for action in actions:
            if id not in dict:
                dict[id] = [[state_,action.numpy(),True]]
            else:
                dict[id].append([state_,action.numpy(),True])

    return dict
def main():

    users_dict = np.load("./data/user_dict_100k.npy", allow_pickle=True).item()
    users_history_lens = np.load("./data/users_histroy_len_100k.npy")

    users_num = len(users_dict) + 1
    items_num_list = set()
    for i in users_dict.items():
        for j in i[1]:
            items_num_list.add(j[0])
    items_num = len(items_num_list) + 1

    # state
    state_representation = DRRAveStateRepresentation(EMBEDDING_DIM)
    state_representation([np.zeros((1, EMBEDDING_DIM,)), np.zeros((1, STATE_SIZE, EMBEDDING_DIM))])

    # embedding net
    emb_net = UserMovieEmbedding(users_num, items_num, EMBEDDING_DIM)
    emb_net([np.zeros((1,)), np.zeros((1,))])
    emb_net.load_weights('./weights/user_movie_embedding_100k.h5')

    train_users_num = int(users_num * 0.8)
    train_users_dict = {k: users_dict.get(k) for k in range(1, train_users_num + 1)}
    train_users_history_lens = users_history_lens[:train_users_num]

    env = OfflineEnv(train_users_dict, train_users_history_lens, items_num_list, STATE_SIZE, state_representation,
                     emb_net,
                     seed=SEED)
    algo = DDPG(action_shape=EMBEDDING_DIM, seed=SEED, gamma=0.9, ac_lr=1e-3, cr_lr=1e-3, tau=0.001,
                state_size=STATE_SIZE, embedding_dim=EMBEDDING_DIM, actor_hidden_dim=128, critic_hidden_dim=128,
                replay_memory_size=1000000, batch_size=64, is_test=False, max_episode_num=NUMBER_STEPS, epsilon=1.,
                std=1.5)

    algo.load_weights('./model/step_8000')
    #collect_demo(env, algo, SEED)
    collect_demo_fixed_users(env,algo,SEED)

if __name__ == '__main__':
    users_dict = np.load("./data/user_dict_100k.npy", allow_pickle=True).item()
    users_history_lens = np.load("./data/users_histroy_len_100k.npy")

    users_num = len(users_dict) + 1
    items_num_list = set()
    for i in users_dict.items():
        for j in i[1]:
            items_num_list.add(j[0])
    items_num = len(items_num_list) + 1

    # state
    state_representation = DRRAveStateRepresentation(EMBEDDING_DIM)
    state_representation([np.zeros((1, EMBEDDING_DIM,)), np.zeros((1, STATE_SIZE, EMBEDDING_DIM))])

    # embedding net
    emb_net = UserMovieEmbedding(users_num, items_num, EMBEDDING_DIM)
    emb_net([np.zeros((1,)), np.zeros((1,))])
    emb_net.load_weights('./weights/user_movie_embedding_100k.h5')

    train_users_num = int(users_num * 0.8)
    train_users_dict = {k: users_dict.get(k) for k in range(1, train_users_num + 1)}
    train_users_history_lens = users_history_lens[:train_users_num]

    env = OfflineEnv(train_users_dict, train_users_history_lens, items_num_list, STATE_SIZE, state_representation,
                     emb_net,
                     seed=SEED)
    #
    # dict = add_fixed_user_action(env,emb_net,SEED)
    # np.save('demostrations_full', dict)
    dict_demo = np.load('./demostration.npy',allow_pickle=True).item()
    users = env.available_users
    keys = dict_demo.keys()
    ks = list(keys)
    ks.sort()
    users_ = []
    for user in users:
        if user not in ks:
            users_.append(user)
    print(users_)
    #main()

