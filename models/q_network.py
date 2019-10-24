import torch
from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, input, output, hiddens=[64], seed=0):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hiddens = hiddens
        if len(self.hiddens) == 0:
            self.fc0 = nn.Linear(input, output)
        for i in range(len(hiddens)):
            if i == 0:
                setattr(self, "fc{}".format(i), nn.Linear(input, self.hiddens[i]))
            else:
                setattr(self, "fc{}".format(i), nn.Linear(self.hiddens[i-1], self.hiddens[i]))

        setattr(self, "fc{}".format(len(hiddens)), nn.Linear(self.hiddens[-1], output))

    def forward(self, x, zerocenter=False):
        if len(self.hiddens) == 0:
            return self.fc0(x)
        else:
            for i in range(0, len(self.hiddens)):
                x = getattr(self, "fc{}".format(i))(x)
                if zerocenter == True:
                    x = x - x.mean(1, keepdims=True)
                x = F.relu(x)
                if zerocenter == True:
                    x = x - x.mean(1, keepdims=True)
            return getattr(self, "fc{}".format(len(self.hiddens)))(x)

