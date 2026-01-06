"""
DataLoader for Training
=======================
PyG Dataset and DataLoader for loading cached ego-graphs.
"""
import os
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from typing import List, Optional
import pandas as pd
import logging

from .config import GRAPHS_DIR, TRAIN_DATASET_PATH, TEST_DATASET_PATH

logger = logging.getLogger(__name__)


class EgoGraphDataset(Dataset):
    """PyTorch Geometric Dataset for loading cached ego-graphs."""

    def __init__(
        self,
        split: str = 'train',
        transform=None,
        pre_transform=None
    ):
        """
        Initialize the dataset.

        Args:
            split: 'train' or 'test'
            transform: Optional transform to apply
            pre_transform: Optional pre-transform
        """
        self.split = split
        self.graphs_dir = os.path.join(GRAPHS_DIR, split)

        # Load address list from original dataset
        dataset_path = TRAIN_DATASET_PATH if split == 'train' else TEST_DATASET_PATH
        df = pd.read_csv(dataset_path)

        # Filter to only addresses with cached graphs
        self.addresses = [
            addr for addr in df['address'].tolist()
            if os.path.exists(os.path.join(self.graphs_dir, f"{addr}.pt"))
        ]

        logger.info(f"Found {len(self.addresses):,} cached graphs for {split} split")

        super().__init__(None, transform, pre_transform)

    def len(self) -> int:
        """Return number of graphs."""
        return len(self.addresses)

    def get(self, idx: int) -> Data:
        """
        Load a single graph.

        Args:
            idx: Index of graph to load

        Returns:
            PyG Data object
        """
        address = self.addresses[idx]
        graph_path = os.path.join(self.graphs_dir, f"{address}.pt")
        data = torch.load(graph_path, weights_only=False)
        return data


def get_train_loader(
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Get DataLoader for training set.

    Args:
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes

    Returns:
        PyG DataLoader
    """
    dataset = EgoGraphDataset(split='train')
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def get_test_loader(
    batch_size: int = 32,
    num_workers: int = 0
) -> DataLoader:
    """
    Get DataLoader for test set.

    Args:
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        PyG DataLoader
    """
    dataset = EgoGraphDataset(split='test')
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


def get_dataset_stats(split: str = 'train') -> dict:
    """
    Get statistics about the dataset.

    Args:
        split: 'train' or 'test'

    Returns:
        Dictionary with statistics
    """
    dataset = EgoGraphDataset(split=split)

    total_nodes = 0
    total_edges = 0
    total_ghost_nodes = 0
    label_counts = {0: 0, 1: 0}

    for i in range(len(dataset)):
        data = dataset.get(i)
        total_nodes += data.num_nodes
        total_edges += data.num_edges if hasattr(data, 'num_edges') else data.edge_index.size(1)
        total_ghost_nodes += data.num_ghost_nodes if hasattr(data, 'num_ghost_nodes') else data.num_nodes - 1

        # Count labels (center node only)
        label = data.y[0].item()
        if label in label_counts:
            label_counts[label] += 1

    return {
        'num_graphs': len(dataset),
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'total_ghost_nodes': total_ghost_nodes,
        'avg_nodes_per_graph': total_nodes / len(dataset) if len(dataset) > 0 else 0,
        'avg_edges_per_graph': total_edges / len(dataset) if len(dataset) > 0 else 0,
        'label_0_count': label_counts[0],
        'label_1_count': label_counts[1]
    }
