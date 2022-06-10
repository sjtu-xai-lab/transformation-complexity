import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ["resmlp10"]


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


DEFAULT_BETA = 100.0


class SoftGate(nn.Module):
    def __init__(self, beta=DEFAULT_BETA):
        super(SoftGate, self).__init__()
        self.beta = beta

    def forward(self, x):
        return torch.sigmoid(self.beta * x)


class Swish(nn.Module):
    # To enable the computation gradients of gating states, we can approximate the ReLU operation
    # using the Swish function (Ramachandran et al., 2017).
    def __init__(self, beta=DEFAULT_BETA):
        super(Swish, self).__init__()
        self.beta = beta
        self.soft_gate = SoftGate(beta=beta)

    def forward(self, x):
        return x * self.soft_gate(x)


class BasicResidualBlock(nn.Module):
    def __init__(self, in_dim, activation_type="relu", beta=DEFAULT_BETA):
        super(BasicResidualBlock, self).__init__()

        self.linear = nn.Linear(in_dim, in_dim, bias=False)

        self.activation_type = activation_type.lower()
        if self.activation_type == "relu":
            self.act = nn.ReLU()
        elif self.activation_type == "swish":
            self.act = Swish(beta=beta)
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.linear(x)
        out += x
        out = self.act(out) # out = F.relu(out)  # ReLU[x+f(x)]
        return out


class ResMLP(nn.Module):
    def __init__(self, n_layer, n_class=10, activation_type="relu"):
        super(ResMLP, self).__init__()
        # (n_layer) indicates the number of conv/linear layers
        # the number of ReLU layers is (n-1)
        self.n_class = n_class
        self.n_layer = n_layer
        self.activation_type = activation_type.lower()

        self.layers = self._make_layers()

    def _make_layers(self):
        layers = []
        for _ in range(self.n_layer - 1):
            layers.append(BasicResidualBlock(3072, self.activation_type))
        layers.append(nn.Linear(3072, self.n_class, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.layers(x)

    def set_store_soft_gating_states(self, save_layers=""):
        assert self.activation_type == "swish"

        save_layers = save_layers.split(",")

        self.sigma_list = {}
        def get_hook(name):
            def store_soft_gates(m, i, o):
                self.sigma_list[name] = o
            return store_soft_gates

        for i, save_layer in enumerate(save_layers):
            save_layer = save_layer.split(".")
            target_layer = self
            for item in save_layer:
                if item.isnumeric():
                    target_layer = target_layer[int(item)]
                else:
                    target_layer = getattr(target_layer, item)
            target_layer.soft_gate.register_forward_hook(get_hook(name=i))


def resmlp10(n_class=10, activation_type="relu"):
    return ResMLP(n_layer=10, n_class=n_class, activation_type=activation_type)


if __name__ == '__main__':
    # ResMLP with swish activations
    net = resmlp10(activation_type="swish")
    net.set_store_soft_gating_states(save_layers="layers.5.act,layers.6.act,layers.7.act,layers.8.act")
    # print(net)
    x = torch.randn(1, 3072)
    out = net(x)
    print(out.shape)
    print(net.sigma_list)

    # ResMLP with relu activations
    net = resmlp10(activation_type="relu")
    # print(net)
    x = torch.randn(1, 3072)
    out = net(x)
    print(out.shape)