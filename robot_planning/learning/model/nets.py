import torch
import torch.nn as nn


class SimpleFFD(nn.Module):
    """Simple Feed Forward Neural Network

    """

    def __init__(self, input_dim, hidden_units):
        super(SimpleFFD, self).__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.Softplus(),
            nn.Linear(hidden_units, hidden_units),
            nn.Softplus(),
            nn.Linear(hidden_units, input_dim),
        )

    def forward(self, x):
        score = self.score(x)
        return score


class SimpleSigmaFFD(nn.Module):
    """Simple Feed Forward Neural Network

    """

    def __init__(self, input_dim, hidden_units):
        super(SimpleSigmaFFD, self).__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_units),
            nn.Softplus(),
            nn.Linear(hidden_units, hidden_units),
            nn.Softplus(),
            nn.Linear(hidden_units, hidden_units),
            nn.Softplus(),
            nn.Linear(hidden_units, input_dim),
        )

    def forward(self, x, sigma):
        x = torch.hstack((x, sigma))
        score = self.score(x)
        return score
