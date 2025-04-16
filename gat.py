import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=(8, 1), dropout=0.6):
        super(GAT, self).__init__()
        self.dropout = dropout

        # First GAT layer: Multi-head attention, outputs concatenated
        self.conv1 = GATConv(in_channels, hidden_channels,heads=heads[0], concat=True, dropout=dropout)

        # Second GAT layer: Single-head attention, outputs averaged (not concatenated)
        self.conv2 = GATConv(hidden_channels * heads[0], out_channels, heads=heads[1], concat=False, dropout=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Xavier initialization as suggested in the original GAT paper."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, GATConv)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_uniform_(m.weight, gain=1.414)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        # Input feature dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # First GAT layer with ELU
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        # Dropout before final layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        # Log-softmax for classification
        return F.log_softmax(x, dim=1)