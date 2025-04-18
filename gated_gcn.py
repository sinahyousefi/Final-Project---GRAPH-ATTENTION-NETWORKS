
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv


class GatedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GatedGCN, self).__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.gated_conv = GatedGraphConv(out_channels=hidden_channels, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input_proj(x)
        x = self.gated_conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_proj(x)
        return F.log_softmax(x, dim=1)
