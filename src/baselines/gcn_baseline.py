"""GCN Baseline.

A deliberately plain 3-layer GCN over the same ego-graphs the OptimalBitcoinGNN
sees. Intended as a *graph-aware* counterpart to the XGBoost tabular baseline:
same node features, but message passing rather than just the center-node row.

Architecture:
- GCNConv(12 -> 64) -> ReLU -> Dropout
- GCNConv(64 -> 64) -> ReLU -> Dropout
- GCNConv(64 -> 64) -> ReLU
- global_mean_pool -> Linear(64 -> 2)

Edge attributes are intentionally ignored (vanilla GCN does not use them) so
that any lift over XGBoost is attributable to message passing alone, and any
gap below OptimalBitcoinGNN reflects the richer architecture there.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class BasicGCN(nn.Module):
    def __init__(
        self,
        num_node_features: int = 12,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, edge_attr=None):
        # edge_attr accepted for API symmetry with OptimalBitcoinGNN; ignored here.
        del edge_attr
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv3(h, edge_index))
        h_graph = global_mean_pool(h, batch)
        return self.classifier(h_graph)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
