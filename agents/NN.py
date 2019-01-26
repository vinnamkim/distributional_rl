import torch
from torch import nn

class NN(nn.Module):
    def __init__(self):
        self.l1 = torch.nn.Linear(1,2)
        self.l2 = torch.nn.Linear(2, 10)

    def forward(self, input):
        o1 = torch.nn.RelU(self.l1(input))
        o2 = self.l2(o1)
        return o2



loss = torch.mean((NN.forward(batch_observations) - NN.forward(next_batch_observations))**2)
