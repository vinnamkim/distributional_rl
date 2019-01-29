import torch
from torch import nn
import torch.nn.functional as F


class Distrib_QNetwork(nn.Module):
    def __init__(self, state_size, action_size, N, hiddens=[64], seed=0):
        super(Distrib_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hiddens = hiddens
        self.action_size = action_size
        self.N = N

        for i in range(len(hiddens)):
            if i == 0:
                setattr(self, "fc{}".format(i), nn.Linear(state_size, self.hiddens[i]))
            else:
                setattr(self, "fc{}".format(i), nn.Linear(self.hiddens[i-1], self.hiddens[i]))

        for i in range(action_size):
            setattr(self, "l_action{}".format(i), nn.Linear(hiddens[-1], self.N))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(getattr(self, "fc0")(state))
        for i in range(1, len(self.hiddens)):
            x = F.relu(getattr(self, "fc{}".format(i))(x))

        for i in range(self.action_size):
            if i == 0:
                output = F.softmax(getattr(self, "l_action{}".format(i))(x))
            else:
                output = torch.cat([output,F.softmax(getattr(self, "l_action{}".format(i))(x))], 1)
        return output

