
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


m = nn.Linear(300,64)
input = torch.randn(300,)
output = m(input)
print(output.size())

