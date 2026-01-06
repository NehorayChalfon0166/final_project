"""
Graph Builder
=============
Constructs PyTorch Geometric Data objects from transaction data.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple
import logging

from .config import (
    FEATURE_COLUMNS,
    NUM_NODE_FEATURES,
    TIMESTAMP_MIN,
    TIMESTAMP_MAX
)

logger = logging.getLogger(__name__)


class EgoGraphBuilder:
    """Builds ego-graphs (depth=1) for wallet addresses."""

    def __init__(self):
        """Initialize the graph builder."""
        self.num_features = NUM_NODE_FEATURES

    def _normalize_timestamp(self, ts: int) -> float:
        """
        Normalize timestamp to [0, 1] range.

        Args:
            ts: Unix timestamp

        Returns:
            Normalized timestamp
        """
        if ts <= TIMESTAMP_MIN:
            return 0.0
        if ts >= TIMESTAMP_MAX:
            return 1.0
        return (ts - TIMESTAMP_MIN) / (TIMESTAMP_MAX - TIMESTAMP_MIN)

    def _log_amount(self, satoshis: int) -> float:
        """
        Apply log1p transformation to satoshi amount.

        Args:
            satoshis: Amount in satoshis

        Returns:
            log1p(satoshis)
        """
        return np.log1p(max(0, satoshis))

    def _parse_transactions(
        self,
        center_address: str,
        transactions: List[Dict]
    ) -> Tuple[List[str], List[Tuple[int, int, float, float, float]]]:
        """
        Parse transactions to extract neighbors and edges.

        Args:
            center_address: The wallet address being analyzed
            transactions: Raw transaction data from mempool API

        Returns:
            neighbors: List of unique neighbor addresses
            edges: List of (src_idx, dst_idx, amount_log, direction, ts_norm)
                   Note: indices are relative (0=center, 1..N=neighbors)
        """
        neighbor_set = set()
        edge_data = []  # (neighbor_addr, direction, amount_log, ts_norm)

        for tx in transactions:
            # Skip unconfirmed transactions
            if not tx.get('status', {}).get('confirmed', False):
                continue

            tx_time = tx['status'].get('block_time', 0)
            if tx_time == 0:
                continue

            ts_norm = self._normalize_timestamp(tx_time)

            # Check if center is sender (appears in inputs)
            is_sender = False
            for inp in tx.get('vin', []):
                prevout = inp.get('prevout', {})
                if prevout.get('scriptpubkey_address') == center_address:
                    is_sender = True
                    break

            # Check if center is receiver (appears in outputs)
            is_receiver = False
            received_amount = 0
            for out in tx.get('vout', []):
                if out.get('scriptpubkey_address') == center_address:
                    is_receiver = True
                    received_amount += out.get('value', 0)

            # Process outgoing edges (center -> recipients)
            if is_sender:
                for out in tx.get('vout', []):
                    recipient = out.get('scriptpubkey_address')
                    amount = out.get('value', 0)

                    if recipient and recipient != center_address and amount > 0:
                        neighbor_set.add(recipient)
                        edge_data.append((
                            recipient,
                            'out',  # direction: center sent
                            self._log_amount(amount),
                            ts_norm
                        ))

            # Process incoming edges (senders -> center)
            if is_receiver and received_amount > 0:
                for inp in tx.get('vin', []):
                    prevout = inp.get('prevout', {})
                    sender = prevout.get('scriptpubkey_address')

                    if sender and sender != center_address:
                        neighbor_set.add(sender)
                        edge_data.append((
                            sender,
                            'in',  # direction: center received
                            self._log_amount(received_amount),
                            ts_norm
                        ))

        return list(neighbor_set), edge_data

    def build_ego_graph(
        self,
        center_address: str,
        center_features: np.ndarray,
        center_label: int,
        transactions: List[Dict]
    ) -> Data:
        """
        Build a PyG Data object for an ego-graph.

        Args:
            center_address: The wallet address being analyzed
            center_features: Log-scaled features from dataset (shape: (NUM_NODE_FEATURES,))
            center_label: Ground truth label (0=benign, 1=criminal)
            transactions: Raw transaction data from mempool API

        Returns:
            PyG Data object with:
            - x: Node features [num_nodes, NUM_NODE_FEATURES]
            - edge_index: Edge connectivity [2, num_edges]
            - edge_attr: Edge features [num_edges, 3] (amount_log, direction, ts_norm)
            - y: Node labels (only center node has valid label)
        """
        # Parse transactions
        neighbors, edge_data = self._parse_transactions(center_address, transactions)

        # Build node index mapping
        # Index 0 = center node
        # Index 1..N = neighbor (ghost) nodes
        all_nodes = [center_address] + neighbors
        addr_to_idx = {addr: i for i, addr in enumerate(all_nodes)}
        num_nodes = len(all_nodes)

        # Build node feature matrix
        # Center node: actual features from dataset
        # Ghost nodes: zeros
        x = np.zeros((num_nodes, self.num_features), dtype=np.float32)
        x[0] = center_features

        # Build edge index and attributes
        if edge_data:
            edge_index_list = []
            edge_attr_list = []

            for neighbor_addr, direction, amount_log, ts_norm in edge_data:
                neighbor_idx = addr_to_idx.get(neighbor_addr)
                if neighbor_idx is None:
                    continue

                if direction == 'out':
                    # Center (0) -> Neighbor
                    edge_index_list.append([0, neighbor_idx])
                    edge_attr_list.append([amount_log, 1.0, ts_norm])
                else:
                    # Neighbor -> Center (0)
                    edge_index_list.append([neighbor_idx, 0])
                    edge_attr_list.append([amount_log, 0.0, ts_norm])

            if edge_index_list:
                edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 3), dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)

        # Build label tensor (-1 for ghost nodes)
        y = torch.full((num_nodes,), -1, dtype=torch.long)
        y[0] = center_label

        # Create PyG Data object
        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes
        )

        # Store metadata
        data.center_address = center_address
        data.num_ghost_nodes = num_nodes - 1
        data.num_edges = edge_index.size(1) if edge_index.numel() > 0 else 0

        return data

    def build_empty_graph(
        self,
        center_address: str,
        center_features: np.ndarray,
        center_label: int
    ) -> Data:
        """
        Build an ego-graph with only the center node (no transactions).

        Args:
            center_address: The wallet address
            center_features: Log-scaled features (shape: NUM_NODE_FEATURES)
            center_label: Ground truth label

        Returns:
            PyG Data object with single node
        """
        x = torch.from_numpy(center_features.reshape(1, -1).astype(np.float32))
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
        y = torch.tensor([center_label], dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=1
        )

        data.center_address = center_address
        data.num_ghost_nodes = 0
        data.num_edges = 0

        return data
