import numpy as np
import tensorflow as tf


class OfflineEnv(object):
    def __init__(self, users_dict, users_history_lens, items_num_list, state_size, state_net, embedding_net, seed):

        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.embedding_net = embedding_net
        self.state_net = state_net
        self.users_dict = users_dict
        self.users_history_lens = users_history_lens
        self.items_num_list = items_num_list

        self.state_size = state_size
        self.available_users = self._generate_available_users()

        self.user = None
        self.user_items = {}
        self.items = []
        self.done = False
        self.recommended_items = set(self.items)
        # self.done_count = 30
        self.state = None
        self.idx = -1
        self.num_user = len(self.available_users)

        self.max_episodes = 100

    def _generate_available_users(self):
        available_users = []
        for i, length in zip(self.users_dict.keys(), self.users_history_lens):
            if length > self.state_size:
                available_users.append(i)
        return available_users

    def next_user(self):
        if (self.idx + 1) / self.num_user == 0:
            np.random.shuffle(self.available_users)
        user = self.available_users[self.idx % self.num_user]
        self.idx += 1
        return user

    def get_fixed_user_state(self,id):

        items = [data[0] for data in self.users_dict[id][:self.state_size]]
        user_eb = self.embedding_net.get_layer('user_embedding')(np.array(id))
        items_eb = self.embedding_net.get_layer('item_embedding')(np.array(items))
        return self.state_net([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])

    def reset(self):

        self.user = self.next_user()
        #self.user = np.random.choice(self.available_users)
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        user_eb = self.embedding_net.get_layer('user_embedding')(np.array(self.user))
        items_eb = self.embedding_net.get_layer('item_embedding')(np.array(self.items))
        self.state = self.state_net([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
        return self.state, self.items, self.done

    def step(self, action, top_k=False):
        reward = -0.5
        recommend_items = self.recommend_item(action,self.recommended_items,top_k=False, items_ids=None)
        if top_k:
            correctly_recommended = []
            rewards = []
            for act in recommend_items:
                # if action recommended item not in list, append it
                if act in self.user_items.keys() and act not in self.recommended_items:
                    correctly_recommended.append(act)
                    rewards.append((self.user_items[act] - 3) / 2)
                else:
                    # else, return -0.5 reward, duplicated recommended item
                    rewards.append(-0.5)
                self.recommended_items.add(act)
            if max(rewards) > 0:
                self.items = self.items[len(correctly_recommended):] + correctly_recommended
            reward = rewards
        else:
            if recommend_items in self.user_items.keys() and recommend_items  not in self.recommended_items:
                reward = (int(self.user_items[recommend_items]) - 3) / 2  # reward if rating bigger than 3 reward plus!

            self.recommended_items.add(recommend_items)
            if reward > 0:
                self.items = self.items[1:] + [recommend_items]

        if len(self.recommended_items) > self.max_episodes or len(self.recommended_items) >= self.users_history_lens[
            self.user - 1]:
            self.done = True
        if reward > 0:
            user_eb = self.embedding_net.get_layer('user_embedding')(np.array(self.user))
            items_eb = self.embedding_net.get_layer('item_embedding')(np.array(self.items))
            state = self.state_net([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
            return state, reward, self.done, self.recommended_items
        return self.state, reward, self.done, self.recommended_items

    # for discrete recommendation
    def step_(self, action, top_k=False):
        reward = -0.5
        recommend_items = action.numpy()
        if top_k:
            correctly_recommended = []
            rewards = []
            for act in recommend_items:
                # if action recommended item not in list, append it
                if act in self.user_items.keys() and act not in self.recommended_items:
                    correctly_recommended.append(act)
                    rewards.append((self.user_items[act] - 3) / 2)
                else:
                    # else, return -0.5 reward, duplicated recommended item
                    rewards.append(-0.5)
                self.recommended_items.add(act)
            if max(rewards) > 0:
                self.items = self.items[len(correctly_recommended):] + correctly_recommended
            reward = rewards
        else:
            if recommend_items in self.user_items.keys() and recommend_items not in self.recommended_items:
                reward = (int(self.user_items[recommend_items]) - 3) / 2  # reward if rating bigger than 3 reward plus!

            self.recommended_items.add(recommend_items)
            if reward > 0:
                self.items = self.items[1:] + [recommend_items]

        if len(self.recommended_items) > self.max_episodes or len(self.recommended_items) >= self.users_history_lens[
            self.user - 1]:
            self.done = True
        if reward > 0:
            user_eb = self.embedding_net.get_layer('user_embedding')(np.array(self.user))
            items_eb = self.embedding_net.get_layer('item_embedding')(np.array(self.items))
            state = self.state_net([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
            return state, reward, self.done, self.recommended_items
        return self.state, reward, self.done, self.recommended_items

    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None):
        if items_ids is None:
            items_ids = np.array(list(self.items_num_list - recommended_items))

        items_ebs = self.embedding_net.get_layer('item_embedding')(items_ids)
        action = tf.transpose(action, perm=(1, 0))
        if top_k:
            item_indice = np.argsort(tf.transpose(tf.keras.backend.dot(items_ebs, action), perm=(1, 0)))[0][-top_k:]
            return items_ids[item_indice]
        else:
            # （*，100） dot （100,1)
            #print(action)
            item_idx = np.argmax(tf.keras.backend.dot(items_ebs, action))
            return items_ids[item_idx]
