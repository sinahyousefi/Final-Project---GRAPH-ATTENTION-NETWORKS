import torch
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv


class GatedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(GatedGCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GatedGraphConv(
            out_channels=hidden_channels, num_layers=num_layers))
        self.linear_in = torch.nn.Linear(in_channels, hidden_channels)
        self.linear_out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.linear_in(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.linear_out(x)
        return F.log_softmax(x, dim=1)
