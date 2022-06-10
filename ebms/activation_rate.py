import torch
import torch.nn as nn


class ActRateTracker(nn.Module):
    """
    For tracking log q(\sigma_l)
    """
    def __init__(self, n_channels, eps=1e-5):
        super(ActRateTracker, self).__init__()
        self.first_time = True
        self.n_channels = n_channels
        self.p_hat = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.eps = eps

    def _update_p(self, p_new):
        if self.first_time:
            self.first_time = False
            self.p_hat.data = 1.0 * p_new.data
        else:
            self.p_hat.data = 0.9 * self.p_hat.data + 0.1 * p_new.data
        self.p_hat.detach_()

    def forward(self, sigma_l):
        if len(sigma_l.shape) == 2:
            sigma_l = sigma_l.view(sigma_l.shape[0], sigma_l.shape[1], 1, 1)
        if self.training:
            p_new = (sigma_l >= 0.5).float().sum(dim=(0, 2, 3)) / (sigma_l.numel() / sigma_l.shape[1])
            p_new = p_new.to(self.p_hat.device)
            p_new = p_new.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self._update_p(p_new)
        res = sigma_l * (2 * self.p_hat.data - 1) - self.p_hat.data + 1
        res = torch.log(res + self.eps).sum(dim=(1, 2, 3))
        return res

    def calculate_mean_act_rate(self):
        return self.p_hat.mean().data.item()
