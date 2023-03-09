import torch
import torch.nn as nn
import ipdb

class FixedNormal(torch.distributions.Normal):
    def log_prob(self, actions):
        return super().log_prob(actions).sum(dim=-1)

    def entropy(self):
        return super().entropy().sum(dim=-1)

    def mode(self):
        return self.mean


class ScaleParameterizedNormal(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.logstd = nn.Parameter(torch.zeros(shape))

    def forward(self, logits):
        return FixedNormal(logits, self.logstd.exp())
