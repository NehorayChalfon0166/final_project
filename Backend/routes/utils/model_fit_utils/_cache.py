"""In-memory TTL caches shared by the inference path.

Mempool.space rate-limits same-IP repeats hard (first call ~1.3s, second
~150s), so /analyze, /info, and /feature-importance share their fetched
transactions, stats, and built graph for a short window. Model weights are
cached for the lifetime of the process.
"""
import threading
import time
from typing import Any, Dict, Tuple

import torch

from src.models.optimal_gnn import OptimalBitcoinGNN

WALLET_TTL_SECONDS = 300

# Per-address: (timestamp, payload). Each cache has its own lock.
_TX_CACHE: Dict[str, Tuple[float, list]] = {}
_TX_INFLIGHT: Dict[str, threading.Event] = {}
TX_LOCK = threading.Lock()

_STATS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_STATS_INFLIGHT: Dict[str, threading.Event] = {}
STATS_LOCK = threading.Lock()

_GRAPH_CACHE: Dict[str, Tuple[float, Any]] = {}
GRAPH_LOCK = threading.Lock()

# Model cache is keyed by resolved model path; persists for the process lifetime.
MODEL_CACHE: Dict[str, Tuple[OptimalBitcoinGNN, float, torch.device]] = {}
MODEL_LOCK = threading.Lock()


def get_fresh(lock: threading.Lock, store: Dict[str, Tuple[float, Any]], key: str):
    """Return the cached payload if present and within TTL, else None."""
    with lock:
        entry = store.get(key)
        if entry and (time.time() - entry[0]) < WALLET_TTL_SECONDS:
            return entry[1]
    return None


# Direct accessors so other modules don't reach into the underlying dicts.
def tx_store() -> Dict[str, Tuple[float, list]]:
    return _TX_CACHE


def tx_inflight() -> Dict[str, threading.Event]:
    return _TX_INFLIGHT


def stats_store() -> Dict[str, Tuple[float, Dict[str, Any]]]:
    return _STATS_CACHE


def stats_inflight() -> Dict[str, threading.Event]:
    return _STATS_INFLIGHT


def graph_store() -> Dict[str, Tuple[float, Any]]:
    return _GRAPH_CACHE
