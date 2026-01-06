"""
Optimal Bitcoin GNN Model
=========================
3-layer GATv2Conv with residual connections for Bitcoin wallet classification.

Architecture:
    - Layer 1: GATv2Conv (12 → 64, 4 heads) = 256 output
    - Layer 2: GATv2Conv (256 → 64, 4 heads) = 256 output + residual
    - Layer 3: GATv2Conv (256 → 32, 2 heads) = 64 output
    - Classification: Linear(64 → 32 → 2)

Expected performance: 92-94% accuracy (baseline: 89.86%)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm
from torch_geometric.data import Batch


class OptimalBitcoinGNN(nn.Module):
    """
    Optimized GATv2 model for cryptocurrency wallet classification.

    Features:
    - 3 GATv2Conv layers with multi-head attention
    - Residual connection between layers 1 and 2
    - LayerNorm for stability with variable graph sizes
    - ELU activation (smooth gradients)
    - Progressive dropout (0.2 → 0.2 → 0.1 → 0.3)
    - Edge feature utilization (3 features: amount, direction, timestamp)
    """

    def __init__(
        self,
        num_node_features: int = 12,
        num_edge_features: int = 3,
        hidden_dim: int = 64,
        num_heads_1: int = 4,
        num_heads_2: int = 4,
        num_heads_3: int = 2,
        dropout: float = 0.2,
        final_dropout: float = 0.3
    ):
        """
        Initialize the model.

        Args:
            num_node_features: Number of input node features (default: 12)
            num_edge_features: Number of edge features (default: 3)
            hidden_dim: Hidden dimension per head (default: 64)
            num_heads_1: Attention heads in layer 1 (default: 4)
            num_heads_2: Attention heads in layer 2 (default: 4)
            num_heads_3: Attention heads in layer 3 (default: 2)
            dropout: Dropout rate for GNN layers (default: 0.2)
            final_dropout: Dropout rate for classifier (default: 0.3)
        """
        super().__init__()

        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Layer 1: Initial feature transformation with attention
        # Input: 12 → Output: 64 * 4 = 256
        self.conv1 = GATv2Conv(
            in_channels=num_node_features,
            out_channels=hidden_dim,
            heads=num_heads_1,
            edge_dim=num_edge_features,
            concat=True,
            dropout=dropout
        )
        self.norm1 = LayerNorm(hidden_dim * num_heads_1)

        # Layer 2: Deep attention with residual
        # Input: 256 → Output: 64 * 4 = 256
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim * num_heads_1,
            out_channels=hidden_dim,
            heads=num_heads_2,
            edge_dim=num_edge_features,
            concat=True,
            dropout=dropout
        )
        self.norm2 = LayerNorm(hidden_dim * num_heads_2)

        # Residual projection (dimensions match, but we keep it for flexibility)
        self.residual_proj = nn.Linear(
            hidden_dim * num_heads_1,
            hidden_dim * num_heads_2
        )

        # Layer 3: Final attention aggregation
        # Input: 256 → Output: 32 * 2 = 64
        self.conv3 = GATv2Conv(
            in_channels=hidden_dim * num_heads_2,
            out_channels=hidden_dim // 2,
            heads=num_heads_3,
            edge_dim=num_edge_features,
            concat=True,
            dropout=dropout * 0.5  # Lighter dropout
        )
        self.norm3 = LayerNorm((hidden_dim // 2) * num_heads_3)

        # Classification head
        classifier_input_dim = (hidden_dim // 2) * num_heads_3  # 64
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),  # 64 → 32
            nn.ELU(),
            nn.Dropout(final_dropout),
            nn.Linear(hidden_dim // 2, 2)  # 32 → 2
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.dropout_light = nn.Dropout(dropout * 0.5)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, num_edge_features]
            batch: Batch assignment [num_nodes]

        Returns:
            Output logits [batch_size, 2]
        """
        # Layer 1
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.norm1(h)
        h = F.elu(h)
        h = self.dropout_layer(h)

        # Save for residual
        h_residual = h

        # Layer 2
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(h)
        h = F.elu(h)
        h = self.dropout_layer(h)

        # Add residual connection
        h = h + self.residual_proj(h_residual)

        # Layer 3
        h = self.conv3(h, edge_index, edge_attr=edge_attr)
        h = self.norm3(h)
        h = F.elu(h)
        h = self.dropout_light(h)

        # Extract center node embeddings (node 0 of each graph)
        h = self._get_center_embeddings(h, batch)

        # Classification
        out = self.classifier(h)

        return out

    def _get_center_embeddings(self, h, batch):
        """
        Extract center node (index 0) embedding for each graph in batch.

        In ego-graphs, the center node is always the first node (index 0)
        of each graph.

        Args:
            h: Node embeddings [num_nodes, embedding_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Center embeddings [batch_size, embedding_dim]
        """
        batch_size = batch.max().item() + 1
        center_embeddings = []

        # Find the start index of each graph
        for i in range(batch_size):
            mask = (batch == i)
            graph_nodes = h[mask]
            center_embeddings.append(graph_nodes[0])  # Center is always first

        return torch.stack(center_embeddings)

    def get_attention_weights(self, x, edge_index, edge_attr):
        """
        Get attention weights from all layers (for interpretability).

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features

        Returns:
            List of attention weight tensors per layer
        """
        attention_weights = []

        # Layer 1
        h, (edge_index_1, alpha_1) = self.conv1(
            x, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        attention_weights.append(alpha_1)
        h = self.norm1(h)
        h = F.elu(h)

        h_residual = h

        # Layer 2
        h, (edge_index_2, alpha_2) = self.conv2(
            h, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        attention_weights.append(alpha_2)
        h = self.norm2(h)
        h = F.elu(h)
        h = h + self.residual_proj(h_residual)

        # Layer 3
        h, (edge_index_3, alpha_3) = self.conv3(
            h, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        attention_weights.append(alpha_3)

        return attention_weights


class FocalLoss(nn.Module):
    """
    Focal Loss for hard example mining.

    Down-weights easy examples to focus training on hard cases.
    Useful even for balanced datasets where some samples are harder.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor (default: 0.5 for balanced)
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        mode: str = 'max'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like accuracy/F1, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        """
        Check if training should stop.

        Args:
            score: Current metric value
            model: Model to save state from

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif self._is_improvement(score):
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improvement(self, score):
        """Check if score is an improvement over best."""
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta

    def load_best_model(self, model):
        """Load the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
