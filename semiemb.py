import torch
import torch.nn as nn

class SemiEmb(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SemiEmb, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index=None):  # edge_index unused, kept for compatibility
        return torch.log_softmax(self.linear(x), dim=1)