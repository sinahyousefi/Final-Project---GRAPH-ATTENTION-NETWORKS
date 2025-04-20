import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, K=3):
        super(ChebNet, self).__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.conv2 = ChebConv(hidden_channels, out_channels, K)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
