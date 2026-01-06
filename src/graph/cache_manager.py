"""
Cache Manager
=============
Handles disk persistence for raw API responses, processed graphs, and progress tracking.
"""
import os
import json
import torch
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from torch_geometric.data import Data

from .config import CACHE_DIR, GRAPHS_DIR, METADATA_DIR


class CacheManager:
    """Manages caching of API responses and processed graphs."""

    def __init__(self, split: str = 'train'):
        """
        Initialize cache manager for a specific split.

        Args:
            split: 'train' or 'test'
        """
        self.split = split
        self.cache_dir = os.path.join(CACHE_DIR, split)
        self.graphs_dir = os.path.join(GRAPHS_DIR, split)
        self.metadata_dir = METADATA_DIR

        # Create directories
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.progress_file = os.path.join(self.metadata_dir, f'progress_{split}.json')
        self.failed_file = os.path.join(self.metadata_dir, f'failed_{split}.json')

    # =========================================================================
    # RAW TRANSACTION CACHE
    # =========================================================================

    def get_cache_path(self, address: str) -> str:
        """Get cache file path for an address."""
        return os.path.join(self.cache_dir, f"{address}.json")

    def get_cached_transactions(self, address: str) -> Optional[List[Dict]]:
        """Load cached raw transactions for an address."""
        cache_path = self.get_cache_path(address)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def save_transactions(self, address: str, transactions: List[Dict]):
        """Save raw transactions to cache."""
        cache_path = self.get_cache_path(address)
        with open(cache_path, 'w') as f:
            json.dump(transactions, f)

    def transaction_cached(self, address: str) -> bool:
        """Check if transactions are cached for an address."""
        return os.path.exists(self.get_cache_path(address))

    # =========================================================================
    # GRAPH CACHE
    # =========================================================================

    def get_graph_path(self, address: str) -> str:
        """Get graph file path for an address."""
        return os.path.join(self.graphs_dir, f"{address}.pt")

    def get_cached_graph(self, address: str) -> Optional[Data]:
        """Load cached PyG Data object."""
        graph_path = self.get_graph_path(address)
        if os.path.exists(graph_path):
            try:
                return torch.load(graph_path, weights_only=False)
            except Exception:
                return None
        return None

    def save_graph(self, address: str, data: Data):
        """Save PyG Data object to disk."""
        graph_path = self.get_graph_path(address)
        torch.save(data, graph_path)

    def graph_exists(self, address: str) -> bool:
        """Check if graph already exists."""
        return os.path.exists(self.get_graph_path(address))

    # =========================================================================
    # PROGRESS TRACKING
    # =========================================================================

    def load_progress(self) -> Dict[str, Any]:
        """Load progress tracking data."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return {
            'completed': [],
            'failed': [],
            'total': 0,
            'started_at': None,
            'last_updated': None
        }

    def save_progress(self, progress: Dict[str, Any]):
        """Save progress tracking data."""
        progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def get_completed_addresses(self) -> Set[str]:
        """Get set of addresses that have been processed."""
        progress = self.load_progress()
        return set(progress.get('completed', []))

    def get_failed_addresses(self) -> List[Dict]:
        """Get list of addresses that failed processing."""
        progress = self.load_progress()
        return progress.get('failed', [])

    def mark_completed(self, address: str, progress: Dict[str, Any]):
        """Mark an address as completed."""
        if address not in progress['completed']:
            progress['completed'].append(address)

    def mark_failed(self, address: str, error: str, progress: Dict[str, Any]):
        """Mark an address as failed with error message."""
        # Check if already in failed list
        for item in progress['failed']:
            if item.get('address') == address:
                item['error'] = error
                item['timestamp'] = datetime.now().isoformat()
                return

        progress['failed'].append({
            'address': address,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data."""
        tx_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
        graph_files = [f for f in os.listdir(self.graphs_dir) if f.endswith('.pt')]

        cache_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f))
            for f in tx_files
        ) / (1024 * 1024)  # MB

        graphs_size = sum(
            os.path.getsize(os.path.join(self.graphs_dir, f))
            for f in graph_files
        ) / (1024 * 1024)  # MB

        return {
            'cached_transactions': len(tx_files),
            'cached_graphs': len(graph_files),
            'cache_size_mb': round(cache_size, 2),
            'graphs_size_mb': round(graphs_size, 2)
        }

    def clear_failed(self):
        """Clear the failed addresses list (for retry)."""
        progress = self.load_progress()
        progress['failed'] = []
        self.save_progress(progress)
