#import pandas as pd
import numpy as np
#import json
#from state_representation import AveStateRepresentation


# item_em = pd.read_csv("Pytorch_models/src/com/item_embedding.csv")
# user_em = pd.read_csv("Pytorch_models/src/com/user_embedding.csv")
#
# item_em['features'] = item_em['features'].map(lambda x : np.array(json.loads(x)))
# user_em['features'] = user_em['features'].map(lambda x : np.array(json.loads(x)))



class Embedding_loader:
    def __init__(self, user_em, item_em):
        self.user_em = user_em
        self.item_em = item_em

    def get_item_em(self,item_ids):

        return np.array([self.item_em[self.item_em["id"] == id].iloc[0, 1] for id in item_ids])

    def get_user_em(self,id):

        return np.array([self.user_em[self.user_em["id"] == id].iloc[0, 1]])

    def check_item_em_(self, id):

        return id in self.item_em['id'] and not self.item_em[self.item_em['id'] == id].empty

    def check_user_em(self, id):

        return id in self.user_em['id'] and not self.user_em[self.user_em["id"] == id].empty






