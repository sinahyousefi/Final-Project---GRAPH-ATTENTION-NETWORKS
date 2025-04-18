

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=(8, 1), dropout=0.6):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attn_weights = None

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads[0], concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads[0], out_channels, heads=heads[1], concat=False, dropout=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, GATConv)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_uniform_(m.weight, gain=1.414)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)

        self.attn_weights = (attn1, attn2)
        return F.log_softmax(x, dim=1)
