


class C(object):
    @property
    def x(self):
        """I'm the 'x' property."""
        print("getter of x called")
        return self._x

    @x.setter
    def x(self, value):
        print("setter of x called")
        self._x = value

    @x.deleter
    def x(self):
        print("deleter of x called")
        del self._x

c = C()

c.x = 'foo'
print(c.x)
c.x = 'hh'
print(c.x)

c1 = C()
print(c.__dict__)
print(c1.__dict__)
#

# c.x = 'foo'
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import os
# m = nn.Linear(300,64)
# input = torch.randn(1,300)
# output = m(input)
# output = output.float()
# print(output.size())
#==============================================================
# dfl = pd.DataFrame(np.random.randn(5, 4),
#                    columns=list('ABCD'),
#                    index=pd.date_range('20130101', periods=5))
#
# list_ = dfl['A'].tolist()
# list_.append(3)
# list_.append(4)
# s_ = set([3,4])
# s = list(set(list_) - s_ )

#----------------------------------
import torch.nn.functional as F
from torch import optim
# class TheModelClass(nn.Module):
#     def __init__(self):
#         super(TheModelClass, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
# # Initialize model
# model = TheModelClass()
#
# # Initialize optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#
# path = './model/model.pt'
# torch.save(model, path)
from tqdm import tqdm

from tqdm import tqdm
import timeit

# for _ in tqdm(range(10000)):
# 	t1 = torch.randn(128,300).to(device)
# 	t2 = torch.randn(300,128).to(device)
#
# 	t1.dot(t2)
# 	outputs.append(t1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# print(device)
# outputs = []
# start = timeit.default_timer()
# for _ in tqdm(range(10)):
# 	t1 = torch.randn(10000,10000).to(device)
# 	t2 = torch.randn(10000,10000).to(device)
#
# 	t = torch.mul(t1, t2).to(device)
# 	outputs.append(t)
#
# stop = timeit.default_timer()
#
# print('Time: ', stop - start)
import pandas as pd

item_em = pd.read_csv("./Embedding/item_embedding_1m.csv")
user_em = pd.read_csv("./Embedding/user_embedding_1m.csv")

items_num_list_v = item_em['id'].values.tolist()
items_num_list = item_em['id'].tolist()
user_num_list = user_em['id'].tolist()
print(len(items_num_list_v)) # 3706
print(len(items_num_list)) # 3706
print(max(items_num_list_v)) # 3952
print(max(items_num_list)) # 3952

#print(item_em[item_em['id'] == 3952].iloc[0, 1])

# print(min(user_num_list)) # 1
# print(min(items_num_list)) # 1
# for i,num in enumerate(items_num_list):
#     print(i)

