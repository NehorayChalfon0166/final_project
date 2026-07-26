"""Cached ego-graph builder for inference."""
import time
from typing import Any, Dict, List

from src.graph.graph_builder import EgoGraphBuilder

from . import _cache


def get_cached_graph(wallet_address: str, transactions: List[Dict]) -> Any:
    """Return the built ego-graph from cache, or build and cache it."""
    cached = _cache.get_fresh(_cache.GRAPH_LOCK, _cache.graph_store(), wallet_address)
    if cached is not None:
        print(f"   [graph-cache] HIT {wallet_address}")
        return cached

    graph = EgoGraphBuilder().build_graph_for_new_address(
        address=wallet_address,
        transactions=transactions,
        label=-1,
    )
    with _cache.GRAPH_LOCK:
        _cache.graph_store()[wallet_address] = (time.time(), graph)
    return graph
