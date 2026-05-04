import numpy as np
import torch
import torch.nn.functional as F
import requests
import os
import sys
import threading
import time
from typing import List, Dict, Optional, Tuple, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.graph.graph_builder import EgoGraphBuilder
from src.graph.config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES
from src.models.optimal_gnn import OptimalBitcoinGNN
from src.evaluation.interpretability import ModelInterpreter


_MODEL_CACHE: Dict[str, Tuple[OptimalBitcoinGNN, float, torch.device]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


# Per-wallet TTL cache. Mempool.space rate-limits same-IP repeat calls
# (1.3s first call vs 150s second call observed), so /analyze, /info, and
# /feature-importance share fetched transactions and the built graph.
_WALLET_CACHE_TTL = 300  # seconds
_TX_CACHE: Dict[str, Tuple[float, List[Dict]]] = {}
_TX_INFLIGHT: Dict[str, threading.Event] = {}
_TX_CACHE_LOCK = threading.Lock()
_STATS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_STATS_INFLIGHT: Dict[str, threading.Event] = {}
_STATS_CACHE_LOCK = threading.Lock()
_GRAPH_CACHE: Dict[str, Tuple[float, Any]] = {}
_GRAPH_CACHE_LOCK = threading.Lock()


def _cache_get(cache_lock, cache, key):
    with cache_lock:
        entry = cache.get(key)
        if entry and (time.time() - entry[0]) < _WALLET_CACHE_TTL:
            return entry[1]
    return None


def get_cached_transactions(wallet_address: str) -> List[Dict]:
    """Fetch transactions for a wallet, deduped via a per-address TTL cache.

    If a fetch for the same address is already in flight, wait for it instead
    of issuing a second Mempool call (Mempool throttles repeats on the same IP).
    """
    cached = _cache_get(_TX_CACHE_LOCK, _TX_CACHE, wallet_address)
    if cached is not None:
        print(f"   [tx-cache] HIT {wallet_address}")
        return cached

    with _TX_CACHE_LOCK:
        cached = _TX_CACHE.get(wallet_address)
        if cached and (time.time() - cached[0]) < _WALLET_CACHE_TTL:
            return cached[1]
        event = _TX_INFLIGHT.get(wallet_address)
        if event is None:
            event = threading.Event()
            _TX_INFLIGHT[wallet_address] = event
            owner = True
        else:
            owner = False

    if not owner:
        event.wait(timeout=_WALLET_CACHE_TTL)
        cached = _cache_get(_TX_CACHE_LOCK, _TX_CACHE, wallet_address)
        return cached if cached is not None else []

    try:
        txs = fetch_transactions_mempool(wallet_address)
        with _TX_CACHE_LOCK:
            _TX_CACHE[wallet_address] = (time.time(), txs)
        return txs
    finally:
        with _TX_CACHE_LOCK:
            _TX_INFLIGHT.pop(wallet_address, None)
        event.set()


def get_cached_address_stats(wallet_address: str) -> Dict[str, Any]:
    """Fetch /address stats with per-address TTL cache and inflight dedup."""
    cached = _cache_get(_STATS_CACHE_LOCK, _STATS_CACHE, wallet_address)
    if cached is not None:
        print(f"   [stats-cache] HIT {wallet_address}")
        return cached

    with _STATS_CACHE_LOCK:
        cached = _STATS_CACHE.get(wallet_address)
        if cached and (time.time() - cached[0]) < _WALLET_CACHE_TTL:
            return cached[1]
        event = _STATS_INFLIGHT.get(wallet_address)
        if event is None:
            event = threading.Event()
            _STATS_INFLIGHT[wallet_address] = event
            owner = True
        else:
            owner = False

    if not owner:
        event.wait(timeout=_WALLET_CACHE_TTL)
        cached = _cache_get(_STATS_CACHE_LOCK, _STATS_CACHE, wallet_address)
        return cached if cached is not None else {}

    try:
        try:
            r = requests.get(
                f"https://mempool.space/api/address/{wallet_address}",
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=15,
            )
            stats = r.json() if r.status_code == 200 else {}
        except Exception as e:
            print(f"   [stats-cache] fetch error: {e}")
            stats = {}
        with _STATS_CACHE_LOCK:
            _STATS_CACHE[wallet_address] = (time.time(), stats)
        return stats
    finally:
        with _STATS_CACHE_LOCK:
            _STATS_INFLIGHT.pop(wallet_address, None)
        event.set()


def get_cached_graph(wallet_address: str, transactions: List[Dict]):
    """Return the built ego-graph from cache, or build and cache it."""
    cached = _cache_get(_GRAPH_CACHE_LOCK, _GRAPH_CACHE, wallet_address)
    if cached is not None:
        print(f"   [graph-cache] HIT {wallet_address}")
        return cached
    builder = EgoGraphBuilder()
    graph = builder.build_graph_for_new_address(
        address=wallet_address,
        transactions=transactions,
        label=-1,
    )
    with _GRAPH_CACHE_LOCK:
        _GRAPH_CACHE[wallet_address] = (time.time(), graph)
    return graph


def _resolve_model_path(model_path: Optional[str]) -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    candidates = [
        model_path,
        os.path.join(project_root, 'outputs', 'gnn_model.pt'),
        os.path.join(project_root, 'outputs', 'gnn_checkpoint.pt'),
        os.path.abspath(model_path) if model_path else None,
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"Model not found. Tried: {candidates}")


def get_cached_model(model_path: Optional[str] = None) -> Tuple[OptimalBitcoinGNN, float, torch.device]:
    """Load the GNN model once and cache it. Subsequent calls reuse the in-memory model."""
    resolved = _resolve_model_path(model_path)
    with _MODEL_CACHE_LOCK:
        if resolved in _MODEL_CACHE:
            return _MODEL_CACHE[resolved]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = OptimalBitcoinGNN(
            num_node_features=NUM_NODE_FEATURES,
            num_edge_features=NUM_EDGE_FEATURES,
        )
        state_dict = torch.load(resolved, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        temperature = 1.0
        temp_path = os.path.join(os.path.dirname(resolved), 'temperature.pt')
        if os.path.exists(temp_path):
            temp_data = torch.load(temp_path, map_location=device, weights_only=True)
            temperature = float(temp_data['temperature'])

        print(f"   [model-cache] Loaded {resolved} on {device} (T={temperature:.3f})")
        _MODEL_CACHE[resolved] = (model, temperature, device)
        return _MODEL_CACHE[resolved]


def fetch_transactions_mempool(wallet_address: str) -> List[Dict]:
    """
    Fetch confirmed transactions for a wallet from Mempool.space.

    Uses /txs/chain (max 25 confirmed per page) instead of /txs (up to 50
    confirmed+mempool) to cap worst-case payload size on heavy exchange-style
    wallets, which can otherwise return tens of MB.
    """
    print(f"   Fetching transactions from Mempool.space for {wallet_address}...")
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        url = f"https://mempool.space/api/address/{wallet_address}/txs/chain"
        r = requests.get(url, headers=headers, timeout=30)

        if r.status_code == 200:
            transactions = r.json()
            print(f"   Found {len(transactions)} transactions")
            return transactions
        else:
            print(f"   API returned status {r.status_code}")
            return []

    except Exception as e:
        print(f"   Error fetching transactions: {e}")
        return []


def analyze_wallet_pipeline(wallet_address: str, model_path: str = None):
    """
    Complete pipeline: Fetch transactions, build graph using EgoGraphBuilder, and run model inference.
    
    Args:
        wallet_address: The wallet address to analyze
        model_path: Path to saved model (optional)
    
    Returns:
        Dictionary with analysis results
    """
    print(f"\n[1/4] Starting analysis for wallet: {wallet_address}")

    # Fetch transactions from mempool (cached per-address)
    print(f"[2/4] Fetching transactions...")
    transactions = get_cached_transactions(wallet_address)

    if not transactions:
        print(f"[!] No transactions found for wallet: {wallet_address}")
        return {
            "wallet_address": wallet_address,
            "status": "no_data",
            "message": "No transaction data found for this wallet"
        }

    # Build ego-graph using EgoGraphBuilder (cached per-address)
    print(f"[3/4] Building ego-graph...")
    graph_data = get_cached_graph(wallet_address, transactions)
    
    # Prepare results
    results = {
        "wallet_address": wallet_address,
        "status": "success",
        "nodes_count": graph_data.num_nodes,
        "edges_count": graph_data.num_edges,
        "ghost_nodes": graph_data.num_ghost_nodes,
        "graph_data": {
            "x_shape": list(graph_data.x.shape),
            "y_shape": list(graph_data.y.shape),
            "edge_index_shape": list(graph_data.edge_index.shape),
            "edge_attr_shape": list(graph_data.edge_attr.shape)
        }
    }
    
    if model_path:
        print(f"[4/4] Running inference...")
        try:
            model, temperature, device = get_cached_model(model_path)

            with torch.no_grad():
                x = graph_data.x.to(device)
                edge_index = graph_data.edge_index.to(device)
                edge_attr = graph_data.edge_attr.to(device) if graph_data.edge_attr.numel() > 0 else None

                # Create batch tensor (all nodes belong to batch 0)
                batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)

                # Model expects batch parameter
                output = model(x, edge_index, edge_attr, batch)

                # Get prediction for the queried wallet (node 0)
                logits = output[0].cpu()

                # Apply temperature scaling before softmax
                logits = logits / temperature

                # Apply softmax to convert logits to probabilities
                probabilities = F.softmax(logits, dim=0).numpy()
                
                # For binary classification: [prob_benign, prob_criminal]
                if len(probabilities) >= 2:
                    prob_benign = float(probabilities[0])
                    prob_criminal = float(probabilities[1])
                    risk_score = prob_criminal
                else:
                    # Single output - treat as probability
                    prob_criminal = float(probabilities[0])
                    prob_benign = 1.0 - prob_criminal
                    risk_score = prob_criminal
                
                # Classify based on threshold (0.5)
                classification = "criminal" if risk_score > 0.5 else "benign"
                # Confidence is how far we are from the decision boundary (0.5)
                confidence = abs(risk_score - 0.5) * 2
                
                results["prediction"] = probabilities.tolist() if hasattr(probabilities, 'tolist') else [float(p) for p in probabilities]
                results["risk_score"] = risk_score
                results["classification"] = classification
                results["confidence"] = confidence
                results["message"] = f"Wallet classified as {classification.upper()} with {confidence*100:.1f}% confidence"
                
                print(f"   Classification: {classification.upper()}")
                print(f"   Risk Score: {risk_score:.4f}")
                print(f"   Confidence: {confidence*100:.1f}%")
        except Exception as e:
            print(f"[!] Inference error: {str(e)}")
            results["inference_error"] = str(e)
            results["message"] = f"Could not run inference: {str(e)}"
    else:
        print(f"[!] No model provided, skipping inference")
        results["message"] = "Analysis completed without classification (no model provided)"
    
    print(f"\n[✓] Analysis complete!")
    print(f"   Results: {results}")
    return results


def compute_feature_importance(wallet_address: str) -> Dict:
    """
    Compute feature importance for a specific wallet using gradient-based method.
    
    Args:
        wallet_address: Bitcoin wallet address to analyze
    
    Returns:
        Dictionary with status, feature importance scores, and metadata
    """
    try:
        # Build the ego graph for the wallet (cached — usually a hit if /analyze ran first)
        print(f"[Feature Importance] Building graph for {wallet_address}...")
        txs = get_cached_transactions(wallet_address)

        if not txs:
            return {
                "status": "error",
                "message": "No transactions found for this wallet address"
            }

        try:
            graph_data = get_cached_graph(wallet_address, txs)
            print(f"[Feature Importance] Graph ready: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
        except Exception as e:
            print(f"[Feature Importance] Error building graph: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Load model (cached after first call)
        print("[Feature Importance] Loading model...")
        model, _temperature, device = get_cached_model()

        # Compute gradients with respect to input features
        print("[Feature Importance] Computing feature importance...")
        x = graph_data.x.clone().requires_grad_(True).to(device)
        edge_index = graph_data.edge_index.to(device)
        edge_attr = graph_data.edge_attr.to(device)
        batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
        
        # Forward pass
        output = model(x, edge_index, edge_attr, batch)
        
        # Get prediction for the target wallet (first node)
        target_output = output[0, 1]  # Criminal class probability for first node
        
        # Compute gradients
        target_output.backward()
        
        # Get feature importance as absolute gradients for the target node
        gradients = x.grad[0].abs().cpu().numpy()
        
        # Normalize gradients so they sum to 1 (100%)
        total = gradients.sum()
        if total > 0:
            gradients = gradients / total
        
        # Feature names (must match FEATURE_COLUMNS in src/graph/config.py)
        feature_names = [
            'lifetime_seconds',
            'activity_rate',
            'in_out_balance',
            'total_txs',
            'send_receive_ratio',
            'fee_per_tx',
            'blocks_btwn_txs_mean',
            'fee_share_mean',
            'avg_tx_size',
            'tx_size_range',
            'max_sent',
            'max_received'
        ]
        
        # Create feature importance dict (values now sum to 1.0)
        feature_importance = {name: float(gradients[i]) for i, name in enumerate(feature_names)}
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"[Feature Importance] Top 3 features: {sorted_features[:3]}")
        
        return {
            "status": "success",
            "feature_importance": feature_importance,
            "num_graphs_used": 1,
            "message": f"Feature importance computed for wallet {wallet_address}"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error computing feature importance: {str(e)}"
        }
