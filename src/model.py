import torch
import torch.nn as nn

class BuiltUpNet(nn.Module):
    """
    Simple Deep Learning model (MLP).
    Input: engineered patch features
    Output: built-up probability (logit)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)
