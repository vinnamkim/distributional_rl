import torch
from torch import nn
import torch.nn.functional as F

class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()

        self.l1 = torch.nn.Linear(2 + 1, 200)
        self.l2 = torch.nn.Linear(200, 1)
        self.n_actions = 1

    def forward(self, input):
        o1 = F.relu(self.l1(input))
        o2 = self.l2(o1)
        return o2
