"""Backend inference helpers, organized by responsibility.

- _cache: TTL caches (transactions, stats, graphs, model)
- _mempool: HTTP fetchers with cache+inflight dedup
- _graph: cached ego-graph builder
- _model: model loading
- _inference: high-level pipelines used by routes/wallet.py

The routes import the public surface from this package; internal modules are
underscored to discourage cross-route reach-ins.
"""
import os
import sys

# Ensure `src.*` imports resolve regardless of CWD/uvicorn working dir.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ._inference import analyze_wallet_pipeline, compute_feature_importance
from ._mempool import (
    fetch_transactions_mempool,
    get_cached_address_stats,
    get_cached_transactions,
)
from ._graph import get_cached_graph
from ._model import get_cached_model

__all__ = [
    "analyze_wallet_pipeline",
    "compute_feature_importance",
    "fetch_transactions_mempool",
    "get_cached_address_stats",
    "get_cached_transactions",
    "get_cached_graph",
    "get_cached_model",
]
