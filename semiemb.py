
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiEmb(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SemiEmb, self).__init__()
        self.embedding = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index=None):
        x = self.embedding(x)
        return F.log_softmax(x, dim=1)