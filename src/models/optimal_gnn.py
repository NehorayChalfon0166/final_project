"""
Optimal Bitcoin GNN Model
=========================
3-layer GATv2Conv with residual connections for Bitcoin wallet classification.

Architecture:
    - Ghost node handling: learnable embeddings + node-type encoding
    - Layer 1: GATv2Conv (12 → 64, 4 heads) = 256 output
    - Layer 2: GATv2Conv (256 → 64, 4 heads) = 256 output + residual
    - Layer 3: GATv2Conv (256 → 32, 2 heads) = 64 output
    - Hybrid readout: center node + global mean pool + global max pool
    - Classification: Linear(192 → 64 → 2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm, global_mean_pool, global_max_pool
from torch_geometric.utils import dropout_edge
from torch_geometric.data import Batch


class OptimalBitcoinGNN(nn.Module):
    """
    Optimized GATv2 model for cryptocurrency wallet classification.

    Features:
    - Learnable ghost node embeddings (replaces zero features for neighbors)
    - Node-type embedding (center vs ghost)
    - 3 GATv2Conv layers with multi-head attention
    - Residual connections (layer 1→2 and initial→final)
    - LayerNorm for stability with variable graph sizes
    - ELU activation (smooth gradients)
    - DropEdge regularization during training
    - Hybrid readout: center node + global mean pool + global max pool
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
        final_dropout: float = 0.3,
        drop_edge_rate: float = 0.1
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.drop_edge_rate = drop_edge_rate

        # Ghost node handling: learnable embedding + node type
        self.ghost_embedding = nn.Parameter(torch.randn(1, num_node_features) * 0.01)
        self.node_type_embedding = nn.Embedding(2, 8)  # 0=center, 1=ghost

        # Input projection: features (12) + has_features flag (1) + node_type (8) → 12
        self.input_proj = nn.Linear(num_node_features + 1 + 8, num_node_features)

        # Layer 1: Initial feature transformation with attention
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
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim * num_heads_1,
            out_channels=hidden_dim,
            heads=num_heads_2,
            edge_dim=num_edge_features,
            concat=True,
            dropout=dropout
        )
        self.norm2 = LayerNorm(hidden_dim * num_heads_2)

        # Residual projection
        self.residual_proj = nn.Linear(
            hidden_dim * num_heads_1,
            hidden_dim * num_heads_2
        )

        # Layer 3: Final attention aggregation
        self.conv3 = GATv2Conv(
            in_channels=hidden_dim * num_heads_2,
            out_channels=hidden_dim // 2,
            heads=num_heads_3,
            edge_dim=num_edge_features,
            concat=True,
            dropout=dropout * 0.5
        )
        self.norm3 = LayerNorm((hidden_dim // 2) * num_heads_3)

        # Initial connection residual: project input features to final GNN dim
        final_gnn_dim = (hidden_dim // 2) * num_heads_3  # 64
        self.initial_proj = nn.Linear(num_node_features, final_gnn_dim)

        # Hybrid readout: center (64) + mean pool (64) + max pool (64) = 192
        classifier_input_dim = final_gnn_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),  # 192 → 64
            nn.ELU(),
            nn.Dropout(final_dropout),
            nn.Linear(hidden_dim, 2)  # 64 → 2
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.dropout_light = nn.Dropout(dropout * 0.5)

    def _prepare_node_features(self, x):
        """Replace zero-feature ghost nodes with learned embeddings and add node-type info."""
        has_features = (x.abs().sum(dim=1) > 0).float()  # 1 for center, 0 for ghost

        # Replace ghost node features with learnable embedding
        ghost_mask = has_features == 0
        if ghost_mask.any():
            x = x.clone()
            x[ghost_mask] = self.ghost_embedding.expand(ghost_mask.sum(), -1)

        # Node type: 0=center (has features), 1=ghost
        node_types = ghost_mask.long()
        type_emb = self.node_type_embedding(node_types)

        # Concatenate: [features, has_features_flag, type_embedding]
        x_augmented = torch.cat([x, has_features.unsqueeze(1), type_emb], dim=1)

        # Project back to original feature dim
        return self.input_proj(x_augmented)

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
        # Prepare node features (handle ghost nodes)
        x = self._prepare_node_features(x)

        # Save original features for initial connection
        x_init = x

        # DropEdge during training
        if self.training and self.drop_edge_rate > 0 and edge_index.size(1) > 0:
            edge_index_drop, edge_mask = dropout_edge(edge_index, p=self.drop_edge_rate, training=self.training)
            edge_attr_drop = edge_attr[edge_mask] if edge_attr is not None else None
        else:
            edge_index_drop = edge_index
            edge_attr_drop = edge_attr

        # Layer 1
        h = self.conv1(x, edge_index_drop, edge_attr=edge_attr_drop)
        h = self.norm1(h)
        h = F.elu(h)
        h = self.dropout_layer(h)

        # Save for residual
        h_residual = h

        # Layer 2
        h = self.conv2(h, edge_index_drop, edge_attr=edge_attr_drop)
        h = self.norm2(h)
        h = F.elu(h)
        h = self.dropout_layer(h)

        # Add residual connection
        h = h + self.residual_proj(h_residual)

        # Layer 3 (use full edges for final layer)
        h = self.conv3(h, edge_index, edge_attr=edge_attr)
        h = self.norm3(h)
        h = F.elu(h)
        h = self.dropout_light(h)

        # Initial connection residual
        h = h + self.initial_proj(x_init)

        # Hybrid readout: center node + global mean pool + global max pool
        center_emb = self._get_center_embeddings(h, batch)
        mean_emb = global_mean_pool(h, batch)
        max_emb = global_max_pool(h, batch)
        h_readout = torch.cat([center_emb, mean_emb, max_emb], dim=1)

        # Classification
        out = self.classifier(h_readout)

        return out

    def _get_center_embeddings(self, h, batch):
        """Extract center node (index 0) embedding for each graph in batch."""
        batch_size = batch.max().item() + 1
        center_embeddings = []

        for i in range(batch_size):
            mask = (batch == i)
            graph_nodes = h[mask]
            center_embeddings.append(graph_nodes[0])

        return torch.stack(center_embeddings)

    def get_attention_weights(self, x, edge_index, edge_attr):
        """Get attention weights from all layers (for interpretability)."""
        x = self._prepare_node_features(x)

        attention_weights = []

        h, (edge_index_1, alpha_1) = self.conv1(
            x, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        attention_weights.append(alpha_1)
        h = self.norm1(h)
        h = F.elu(h)

        h_residual = h

        h, (edge_index_2, alpha_2) = self.conv2(
            h, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        attention_weights.append(alpha_2)
        h = self.norm2(h)
        h = F.elu(h)
        h = h + self.residual_proj(h_residual)

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


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for model calibration.

    Learns a single temperature parameter T on a validation set.
    At inference, logits are divided by T before softmax.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    @staticmethod
    def calibrate(model, val_loader, device, get_labels_fn, lr=0.01, max_iter=50):
        """
        Learn optimal temperature on a validation set.

        Args:
            model: Trained GNN model (eval mode)
            val_loader: Validation DataLoader
            device: Device
            get_labels_fn: Function to extract center labels from a batch
            lr: Learning rate for LBFGS
            max_iter: Max LBFGS iterations

        Returns:
            Fitted TemperatureScaler
        """
        scaler = TemperatureScaler().to(device)
        nll = nn.CrossEntropyLoss()

        # Collect all logits and labels
        all_logits = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                labels = get_labels_fn(batch)
                all_logits.append(logits)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll(scaler(all_logits), all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return scaler
