import numpy as np
import torch

class OfflineEnv(object):

    def __init__(self, users_dict, users_history_lens, items_num_list, state_representation,
                 state_size,embedding_loader, seed = 0,fix_user_id=None):
        np.random.seed(seed)
        self.users_dict = users_dict
        self.users_history_lens = users_history_lens
        #self.items_id_to_name = movies_id_to_movies
        self.items_num_list = items_num_list
        # embedding files and state representation
        self.embedding_loader = embedding_loader
        self.state_representation = state_representation
        # 10 state size
        self.state_size = state_size
        self.action_space = (1,100)
        self.available_users = self._generate_available_users()
        self.fix_user_id = fix_user_id

        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}

        # self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        #self.items = self._generate_available_items()
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)

        self.done_count = 400
        #np.random.seed(0)
        self._max_episode_steps = 10**3


    def _generate_available_users(self):
        available_users = []
        # select the users which rates the movies over 10+
        for user, length in self.users_history_lens.items():
            if length > self.state_size and self.embedding_loader.check_user_em(user):
                available_users.append(user)
        return available_users

    def _generate_available_items(self):
        available_items = []
        for data in self.users_dict[self.user]:
            if self.embedding_loader.check_item_em_(data[0]):
                available_items.append(data[0])
                if len(available_items) == 10:
                    break
        if len(available_items) != 10:
            self.users_dict.pop(self.user,None)
            self.available_users.remove(self.user)
            self.user = np.random.choice(self.available_users)
            return self._generate_available_items()
        return available_items

    def reset(self):
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        #self.items = self._generate_available_items()
        self.done = False

        self.recommended_items = set(self.items)

        user_eb = self.embedding_loader.get_user_em(id=self.user)
        items_eb = self.embedding_loader.get_item_em(item_ids=self.items)

        return self.state_representation([np.expand_dims(items_eb, axis=0), np.expand_dims(user_eb, axis=0)])


    def step(self, action, top_k=False):
        action = self.recommend_item(action, self.recommended_items, top_k)
        #print(action.shape," action s")
        reward = -0.5

        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:
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
            if action in self.user_items.keys() and action not in self.recommended_items:
                reward = int(self.user_items[action]) - 3  # reward if rating bigger than 3 reward plus!
            if reward > 0:
                self.items = self.items[1:] + [action]
            self.recommended_items.add(action)
        if len(self.recommended_items) > self.done_count or len(self.recommended_items) > self.users_history_lens[
            self.user]:
            self.done = True

        user_eb = self.embedding_loader.get_user_em(id = self.user)
        items_eb = self.embedding_loader.get_item_em(item_ids=self.items)
        next_state = self.state_representation([np.expand_dims(items_eb, axis=0), np.expand_dims(user_eb, axis=0)])
        return next_state, reward, self.done, self.recommended_items

    # def get_items_names(self, items_ids):
    #     items_names = []
    #     for id in items_ids:
    #         try:
    #             items_names.append(self.items_id_to_name[str(id)])
    #         except:
    #             items_names.append(list(['Not in list']))
    #     return items_names

    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None):
        #
        action  = action.cpu().clone()
        # print(type(action), "tp")
        if items_ids == None:
                        #3000+ items_num
            items_ids = np.array(list(set(self.items_num_list) - recommended_items))

        items_ebs = self.embedding_loader.get_item_em(items_ids)
        action = np.transpose(action, (1,0))
        #(100,1)
        if top_k:
            item_indice = np.argsort(np.transpose(np.dot(items_ebs, action), (1,0)))[0][-top_k:]
            return items_ids[item_indice]
        else:
            item_idx = np.argmax(np.dot(items_ebs, action))
            return items_ids[item_idx]
    # def test_embedding(self):
    #     #     for id in self.available_users:
    #     #         print(id)
    #     #         if id not in self.embedding_loader.user_em['id']:
    #     #             print("not in")
    # def test_item_embedding(self):
    #     for item in self.items_num_list:
    #         print(item)
    #         if item not in self.embedding_loader.item_em['id']:
    #             print("not in !!!!!!!!!!!!!!!!!!!!!!")