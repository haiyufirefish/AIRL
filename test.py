
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
# m = nn.Linear(300,64)
# input = torch.randn(1,300)
# output = m(input)
# output = output.float()
# print(output.size())
#==============================================================
dfl = pd.DataFrame(np.random.randn(5, 4),
                   columns=list('ABCD'),
                   index=pd.date_range('20130101', periods=5))

list_ = dfl['A'].tolist()
list_.append(3)
list_.append(4)
s_ = set([3,4])
s = list(set(list_) - s_ )


