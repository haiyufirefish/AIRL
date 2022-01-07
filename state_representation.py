import torch.nn as nn
import torch


class AveStateRepresentation(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        self.dim = embedding_dim #(1,100)->output (1,300)
        self.avgpool1 = nn.AvgPool1d(kernel_size = 10)
        self.flatten = nn.Flatten()

    def forward(self,x): #x[0]:items [1,10,100] ,x[1]: user [1,100]

        items_eb = torch.as_tensor(x[0]).permute(0,2,1)
        avg_p = self.avgpool1(items_eb)

        avg_p = torch.squeeze(avg_p,dim = 2)
        user_eb = torch.as_tensor(x[1])
        mul = torch.multiply(user_eb,avg_p)

        user_eb = torch.squeeze(user_eb,dim=1)
        mul = torch.squeeze(mul,dim=1)

        concat = torch.cat([user_eb,mul,avg_p],1)

        return self.flatten(concat)


