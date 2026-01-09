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
                prevout = inp.get('prevout') or {} #for if its mined then there is no sender
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
                    prevout = inp.get('prevout') or {} #for if its mined then there is no sender
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

    def compute_features_from_transactions(
        self,
        address: str,
        transactions: List[Dict]
    ) -> np.ndarray:
        """
        Compute wallet features from raw transaction data for addresses NOT in the dataset.
        
        Args:
            address: The wallet address
            transactions: Raw transaction data from mempool API
            
        Returns:
            features: numpy array of shape (NUM_NODE_FEATURES,) with log-scaled features
        """
        # Initialize counters
        total_sent = 0
        total_received = 0
        total_fees = 0
        num_send_txs = 0
        num_receive_txs = 0
        sent_amounts = []
        received_amounts = []
        tx_times = []
        
        for tx in transactions:
            if not tx.get('status', {}).get('confirmed', False):
                continue
                
            tx_time = tx['status'].get('block_time', 0)
            if tx_time > 0:
                tx_times.append(tx_time)
            
            # Check if address is sender
            is_sender = False
            for inp in tx.get('vin', []):
                prevout = inp.get('prevout') or {}
                if prevout.get('scriptpubkey_address') == address:
                    is_sender = True
                    break
            
            # Check if address is receiver
            received_in_tx = 0
            for out in tx.get('vout', []):
                if out.get('scriptpubkey_address') == address:
                    received_in_tx += out.get('value', 0)
            
            if is_sender:
                num_send_txs += 1
                # Sum sent amounts (outputs not going back to self)
                for out in tx.get('vout', []):
                    if out.get('scriptpubkey_address') != address:
                        amount = out.get('value', 0)
                        total_sent += amount
                        if amount > 0:
                            sent_amounts.append(amount)
                # Add fee
                total_fees += tx.get('fee', 0)
            
            if received_in_tx > 0:
                num_receive_txs += 1
                total_received += received_in_tx
                received_amounts.append(received_in_tx)
        
        # Calculate derived features
        total_txs = num_send_txs + num_receive_txs
        
        # Lifetime
        if len(tx_times) >= 2:
            lifetime_seconds = max(tx_times) - min(tx_times)
        else:
            lifetime_seconds = 0
        
        # Activity rate
        activity_rate = total_txs / max(lifetime_seconds, 1)
        
        # Send/receive ratio
        send_receive_ratio = total_sent / max(total_received, 1)
        
        # In/out balance
        in_out_balance = num_receive_txs / max(num_send_txs, 1)
        
        # Fee per tx
        fee_per_tx = total_fees / max(total_txs, 1)
        
        # Blocks between txs mean (approx: 1 block ≈ 600 seconds)
        if len(tx_times) >= 2:
            tx_times_sorted = sorted(tx_times)
            intervals = [tx_times_sorted[i+1] - tx_times_sorted[i] for i in range(len(tx_times_sorted)-1)]
            blocks_btwn_txs_mean = np.mean(intervals) / 600
        else:
            blocks_btwn_txs_mean = 0
        
        # Fee share mean
        total_transacted = total_sent + total_received
        fee_share_mean = total_fees / max(total_transacted, 1)
        
        # Avg tx size
        avg_tx_size = total_transacted / max(total_txs, 1)
        
        # Tx size range
        all_amounts = sent_amounts + received_amounts
        if all_amounts:
            tx_size_range = max(all_amounts) - min(all_amounts)
        else:
            tx_size_range = 0
        
        # Max sent/received
        max_sent = max(sent_amounts) if sent_amounts else 0
        max_received = max(received_amounts) if received_amounts else 0
        
        # Build feature vector (in same order as FEATURE_COLUMNS)
        features_raw = np.array([
            lifetime_seconds,       # lifetime_seconds_log
            activity_rate,          # activity_rate_log
            in_out_balance,         # in_out_balance_log
            total_txs,              # total_txs_log
            send_receive_ratio,     # send_receive_ratio_log
            fee_per_tx,             # fee_per_tx_log
            blocks_btwn_txs_mean,   # blocks_btwn_txs_mean_log
            fee_share_mean,         # fee_share_mean_log
            avg_tx_size,            # avg_tx_size_log
            tx_size_range,          # tx_size_range_log
            max_sent,               # max_sent_log
            max_received,           # max_received_log
        ], dtype=np.float32)
        
        # Apply log1p transformation (matching dataset preprocessing)
        features_log = np.log1p(np.abs(features_raw)).astype(np.float32)
        
        return features_log

    def build_graph_for_new_address(
        self,
        address: str,
        transactions: List[Dict],
        label: int = -1
    ) -> Data:
        """
        Build ego-graph for an address NOT in the dataset.
        Features are computed from raw transaction data.
        
        Args:
            address: The wallet address
            transactions: Raw transaction data from mempool API
            label: Label (-1 for unknown/unlabeled)
            
        Returns:
            PyG Data object
        """
        # Compute features from transactions
        features = self.compute_features_from_transactions(address, transactions)
        
        if transactions:
            return self.build_ego_graph(
                center_address=address,
                center_features=features,
                center_label=label,
                transactions=transactions
            )
        else:
            return self.build_empty_graph(
                center_address=address,
                center_features=features,
                center_label=label
            )
