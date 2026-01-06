"""
Model Interpretability Module
=============================
Tools for understanding model decisions:
- Attention weight analysis
- Feature importance (gradient-based)
- Embedding visualization (t-SNE/UMAP)
"""
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AttentionInfo:
    """Container for attention weight information."""
    layer_idx: int
    head_idx: int
    source_node: int
    target_node: int
    weight: float


class ModelInterpreter:
    """
    Interpret GNN model decisions.

    Provides:
    - Attention weight extraction and analysis
    - Gradient-based feature importance
    - Embedding extraction for visualization
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize interpreter.

        Args:
            model: Trained GNN model
            feature_names: List of feature names
            device: Device to use
        """
        self.model = model
        self.feature_names = feature_names or [
            'lifetime_seconds_log', 'activity_rate_log', 'in_out_balance_log',
            'total_txs_log', 'send_receive_ratio_log', 'fee_per_tx_log',
            'blocks_btwn_txs_mean_log', 'fee_share_mean_log', 'avg_tx_size_log',
            'tx_size_range_log', 'max_sent_log', 'max_received_log'
        ]
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def extract_attention_weights(self, graph: Data) -> Dict[int, np.ndarray]:
        """
        Extract attention weights from all GAT layers.

        Args:
            graph: Single graph data object

        Returns:
            Dictionary mapping layer index to attention weight matrix
        """
        self.model.eval()

        # Create batch from single graph
        batch = Batch.from_data_list([graph]).to(self.device)

        attention_weights = {}

        # Hook to capture attention weights
        hooks = []
        layer_attentions = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                # GATv2Conv returns (out, attention_weights) when return_attention_weights=True
                # We need to run with return_attention_weights=True
                pass
            return hook

        # For now, we'll do a forward pass and extract what we can
        with torch.no_grad():
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr

            # Get attention from each layer
            for idx, conv_layer in enumerate(self._get_conv_layers()):
                if hasattr(conv_layer, 'return_attention_weights'):
                    # Run with attention weights
                    x_new, (edge_idx, attn) = conv_layer(
                        x, edge_index, edge_attr=edge_attr,
                        return_attention_weights=True
                    )
                    attention_weights[idx] = attn.cpu().numpy()
                    x = x_new

        return attention_weights

    def _get_conv_layers(self) -> List[nn.Module]:
        """Get list of convolutional layers from model."""
        conv_layers = []
        for name, module in self.model.named_modules():
            if 'conv' in name.lower() or 'gat' in name.lower():
                if hasattr(module, 'forward'):
                    conv_layers.append(module)
        return conv_layers

    def get_attention_summary(self, graph: Data) -> Dict:
        """
        Get summary of attention patterns for a graph.

        Args:
            graph: Graph data object

        Returns:
            Summary statistics
        """
        attention = self.extract_attention_weights(graph)

        if not attention:
            return {'error': 'Could not extract attention weights'}

        summary = {}
        for layer_idx, weights in attention.items():
            summary[f'layer_{layer_idx}'] = {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'max': float(np.max(weights)),
                'min': float(np.min(weights)),
                'center_node_attention': self._get_center_attention(weights, graph)
            }

        return summary

    def _get_center_attention(
        self,
        attention_weights: np.ndarray,
        graph: Data
    ) -> Dict[str, float]:
        """Get attention weights related to center node (node 0)."""
        edge_index = graph.edge_index.cpu().numpy()

        # Find edges connected to center node (node 0)
        incoming = edge_index[1] == 0
        outgoing = edge_index[0] == 0

        incoming_attn = attention_weights[incoming] if incoming.any() else np.array([0])
        outgoing_attn = attention_weights[outgoing] if outgoing.any() else np.array([0])

        return {
            'incoming_mean': float(np.mean(incoming_attn)),
            'incoming_max': float(np.max(incoming_attn)),
            'outgoing_mean': float(np.mean(outgoing_attn)),
            'outgoing_max': float(np.max(outgoing_attn))
        }

    def compute_feature_importance(
        self,
        graphs: List[Data],
        method: str = 'gradient'
    ) -> Dict[str, float]:
        """
        Compute feature importance using gradients.

        Args:
            graphs: List of graphs to analyze
            method: 'gradient' for gradient-based importance

        Returns:
            Dictionary of feature name to importance score
        """
        self.model.eval()

        # Collect gradients for each feature
        feature_grads = {name: [] for name in self.feature_names}

        for graph in graphs:
            batch = Batch.from_data_list([graph]).to(self.device)
            batch.x.requires_grad = True

            # Forward pass
            out = self.model(
                batch.x, batch.edge_index,
                batch.edge_attr, batch.batch
            )

            # Backward pass for predicted class
            pred_class = out.argmax(dim=1)
            out[0, pred_class].backward()

            # Get gradients for center node
            if batch.x.grad is not None:
                center_grad = batch.x.grad[0].cpu().numpy()
                for i, name in enumerate(self.feature_names):
                    if i < len(center_grad):
                        feature_grads[name].append(abs(center_grad[i]))

            # Clear gradients
            self.model.zero_grad()

        # Aggregate importance scores
        importance = {}
        for name, grads in feature_grads.items():
            if grads:
                importance[name] = float(np.mean(grads))

        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def compute_class_feature_importance(
        self,
        graphs: List[Data],
        labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute feature importance separately for each class.

        Args:
            graphs: List of graphs
            labels: Ground truth labels

        Returns:
            Dictionary with 'benign' and 'criminal' importance scores
        """
        benign_graphs = [g for g, l in zip(graphs, labels) if l == 0]
        criminal_graphs = [g for g, l in zip(graphs, labels) if l == 1]

        return {
            'benign': self.compute_feature_importance(benign_graphs[:100]),  # Limit for speed
            'criminal': self.compute_feature_importance(criminal_graphs[:100])
        }

    @torch.no_grad()
    def extract_embeddings(
        self,
        graphs: List[Data],
        batch_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract final embeddings for all graphs.

        Args:
            graphs: List of graph data objects
            batch_size: Batch size for processing

        Returns:
            Tuple of (embeddings, labels, sources)
        """
        self.model.eval()

        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

        all_embeddings = []
        all_labels = []
        all_sources = []

        for batch in loader:
            batch = batch.to(self.device)

            # Get embeddings before classification head
            embeddings = self._get_embeddings(batch)

            all_embeddings.append(embeddings.cpu().numpy())

            # Extract labels
            if hasattr(batch, 'ptr'):
                center_indices = batch.ptr[:-1]
            else:
                batch_size = batch.batch.max().item() + 1
                center_indices = []
                for i in range(batch_size):
                    mask = (batch.batch == i)
                    first_idx = mask.nonzero(as_tuple=True)[0][0]
                    center_indices.append(first_idx)
                center_indices = torch.tensor(center_indices, device=batch.y.device)

            labels = batch.y[center_indices].cpu().numpy()
            all_labels.append(labels)

            # Extract sources if available
            if hasattr(batch, 'source'):
                # Handle batched source attribute
                sources = [batch.source] if isinstance(batch.source, str) else ['unknown'] * len(labels)
            else:
                sources = ['unknown'] * len(labels)
            all_sources.extend(sources)

        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)
        sources = np.array(all_sources)

        return embeddings, labels, sources

    def _get_embeddings(self, batch) -> torch.Tensor:
        """
        Get embeddings from model before classification head.

        This requires knowledge of model architecture.
        For OptimalBitcoinGNN, we extract after the last GAT layer.
        """
        # Try to get embeddings by running through layers manually
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        # Run through conv layers (assuming model has conv1, conv2, conv3)
        if hasattr(self.model, 'conv1'):
            x = self.model.conv1(x, edge_index, edge_attr)
            if hasattr(self.model, 'norm1'):
                x = self.model.norm1(x)
            x = torch.nn.functional.elu(x)

        if hasattr(self.model, 'conv2'):
            residual = x
            x = self.model.conv2(x, edge_index, edge_attr)
            if hasattr(self.model, 'norm2'):
                x = self.model.norm2(x)
            x = torch.nn.functional.elu(x) + residual

        if hasattr(self.model, 'conv3'):
            x = self.model.conv3(x, edge_index, edge_attr)
            if hasattr(self.model, 'norm3'):
                x = self.model.norm3(x)
            x = torch.nn.functional.elu(x)

        # Extract center node embeddings
        if hasattr(batch, 'ptr'):
            center_indices = batch.ptr[:-1]
        else:
            batch_size = batch.batch.max().item() + 1
            center_indices = []
            for i in range(batch_size):
                mask = (batch.batch == i)
                first_idx = mask.nonzero(as_tuple=True)[0][0]
                center_indices.append(first_idx)
            center_indices = torch.tensor(center_indices, device=x.device)

        return x[center_indices]

    def reduce_embeddings(
        self,
        embeddings: np.ndarray,
        method: str = 'tsne',
        n_components: int = 2,
        **kwargs
    ) -> np.ndarray:
        """
        Reduce embedding dimensionality for visualization.

        Args:
            embeddings: High-dimensional embeddings
            method: 'tsne' or 'umap'
            n_components: Output dimensions
            **kwargs: Additional arguments for reducer

        Returns:
            Reduced embeddings
        """
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=kwargs.get('perplexity', 30),
                n_iter=kwargs.get('n_iter', 1000)
            )
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=42,
                    n_neighbors=kwargs.get('n_neighbors', 15),
                    min_dist=kwargs.get('min_dist', 0.1)
                )
            except ImportError:
                logger.warning("UMAP not installed, falling back to t-SNE")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        return reducer.fit_transform(embeddings)

    def get_embedding_stats(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Compute statistics about embedding space.

        Args:
            embeddings: Embedding vectors
            labels: Class labels

        Returns:
            Statistics dictionary
        """
        from sklearn.metrics import silhouette_score
        from scipy.spatial.distance import cdist

        stats = {}

        # Silhouette score (how well-separated are the classes)
        if len(np.unique(labels)) > 1:
            stats['silhouette_score'] = float(silhouette_score(embeddings, labels))

        # Intra-class and inter-class distances
        benign_mask = labels == 0
        criminal_mask = labels == 1

        if benign_mask.any() and criminal_mask.any():
            benign_emb = embeddings[benign_mask]
            criminal_emb = embeddings[criminal_mask]

            # Intra-class distances
            benign_intra = cdist(benign_emb, benign_emb).mean()
            criminal_intra = cdist(criminal_emb, criminal_emb).mean()

            # Inter-class distance
            inter_dist = cdist(benign_emb, criminal_emb).mean()

            stats['benign_intra_distance'] = float(benign_intra)
            stats['criminal_intra_distance'] = float(criminal_intra)
            stats['inter_class_distance'] = float(inter_dist)
            stats['separation_ratio'] = float(inter_dist / ((benign_intra + criminal_intra) / 2))

        return stats

    def print_feature_importance(self, importance: Dict[str, float]) -> str:
        """Generate printable feature importance summary."""
        lines = [
            "Feature Importance (Gradient-based)",
            "=" * 40,
            f"{'Feature':<30} {'Importance':>10}"
        ]

        for name, score in importance.items():
            bar = '█' * int(score * 50)
            lines.append(f"{name:<30} {score:>10.4f} {bar}")

        return '\n'.join(lines)
