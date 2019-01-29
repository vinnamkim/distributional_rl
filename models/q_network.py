import torch
from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hiddens=[64], seed=0):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hiddens = hiddens

        for i in range(len(hiddens)):
            if i == 0:
                setattr(self, "fc{}".format(i), nn.Linear(state_size, self.hiddens[i]))
            else:
                setattr(self, "fc{}".format(i), nn.Linear(self.hiddens[i-1], self.hiddens[i]))

        setattr(self, "fc{}".format(len(hiddens)), nn.Linear(self.hiddens[-1], action_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(getattr(self, "fc0")(state))
        for i in range(1, len(self.hiddens)):
            x = F.relu(getattr(self, "fc{}".format(i))(x))
        return getattr(self, "fc{}".format(len(self.hiddens)))(x)

