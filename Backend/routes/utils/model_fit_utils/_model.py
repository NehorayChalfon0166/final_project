"""GNN model loading with process-lifetime caching."""
import os
from typing import Optional, Tuple

import torch

from src.graph.config import NUM_EDGE_FEATURES, NUM_NODE_FEATURES
from src.models.optimal_gnn import OptimalBitcoinGNN

from . import _cache

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


def _resolve_model_path(model_path: Optional[str]) -> str:
    """Pick the first existing path among the explicit override and the canonical fallbacks."""
    candidates = [
        model_path,
        os.path.join(PROJECT_ROOT, 'outputs', 'gnn_model.pt'),
        os.path.join(PROJECT_ROOT, 'outputs', 'gnn_checkpoint.pt'),
        os.path.abspath(model_path) if model_path else None,
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"Model not found. Tried: {candidates}")


def get_cached_model(model_path: Optional[str] = None) -> Tuple[OptimalBitcoinGNN, float, torch.device]:
    """Load the GNN once and reuse it on subsequent calls."""
    resolved = _resolve_model_path(model_path)
    with _cache.MODEL_LOCK:
        if resolved in _cache.MODEL_CACHE:
            return _cache.MODEL_CACHE[resolved]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = OptimalBitcoinGNN(
            num_node_features=NUM_NODE_FEATURES,
            num_edge_features=NUM_EDGE_FEATURES,
        )
        model.load_state_dict(torch.load(resolved, map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        # Optional post-hoc calibration scalar produced by training.
        temperature = 1.0
        temp_path = os.path.join(os.path.dirname(resolved), 'temperature.pt')
        if os.path.exists(temp_path):
            temp_data = torch.load(temp_path, map_location=device, weights_only=True)
            temperature = float(temp_data['temperature'])

        print(f"   [model-cache] Loaded {resolved} on {device} (T={temperature:.3f})")
        _cache.MODEL_CACHE[resolved] = (model, temperature, device)
        return _cache.MODEL_CACHE[resolved]
