import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ["mlp_mnist"]


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class MLP_MNIST(nn.Module):

    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.linear1 = nn.Linear(784, 1024, bias=False)
        self.linear2 = nn.Linear(1024, 256, bias=False)
        self.linear3 = nn.Linear(256, 128, bias=False)
        self.linear4 = nn.Linear(128, 64, bias=False)
        self.linear5 = nn.Linear(64, 10, bias=False)
        self.relu = nn.ReLU()

        self.apply(_weights_init)

    def forward(self, data):
        data = data.reshape(-1, 784)
        h1 = self.relu(self.linear1(data))
        h2 = self.relu(self.linear2(h1))
        h3 = self.relu(self.linear3(h2))
        h4 = self.relu(self.linear4(h3))
        h5 = self.linear5(h4)

        self.sigma_list = [
            (h1 > 0).float(),
            (h2 > 0).float(),
            (h3 > 0).float(),
            (h4 > 0).float()
        ]

        return h5


def mlp_mnist():
    return MLP_MNIST()
