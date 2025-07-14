import torch
import torch.nn as nn

class BotNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BotNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 6),
            nn.ReLU(),
            nn.Linear(6, output_size)
        )

    def forward(self, x):
        out = self.network(x)
        return out
    