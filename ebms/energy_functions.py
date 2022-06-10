import torch
import torch.nn as nn
import torch.nn.init as init


__all__ = ["E_3072d"]


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class E_3072d(nn.Module):
    def __init__(self):
        super(E_3072d, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3072, 32, kernel_size=1, stride=1, padding=0, bias=False),# 32 x 1 x 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),# 64 x 1 x 1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),# 128 x 1 x 1
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),# 256 x 1 x 1
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),# 512 x 1 x 1
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),# 512 x 1 x 1
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.dense = nn.Linear(512, 1)
        self.apply(_weights_init)

    def forward(self, input):
        x = input.view(-1, 3072, 1, 1)
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.dense(x) * 3072 * 1 * 1 / 1000