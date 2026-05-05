"""High-level inference pipelines used by the FastAPI routes.

Two entry points:
- analyze_wallet_pipeline: classify a wallet (the /analyze route).
- compute_feature_importance: gradient saliency on the cached graph.
"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ._graph import get_cached_graph
from ._mempool import get_cached_transactions
from ._model import get_cached_model

# Order must match FEATURE_COLUMNS in src/graph/config.py.
FEATURE_NAMES = [
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
    'max_received',
]


def _graph_descriptor(graph_data) -> Dict:
    return {
        "x_shape": list(graph_data.x.shape),
        "y_shape": list(graph_data.y.shape),
        "edge_index_shape": list(graph_data.edge_index.shape),
        "edge_attr_shape": list(graph_data.edge_attr.shape),
    }


def _run_forward(model, graph_data, device) -> torch.Tensor:
    """Forward pass for a single graph; returns raw logits for the center node."""
    x = graph_data.x.to(device)
    edge_index = graph_data.edge_index.to(device)
    edge_attr = graph_data.edge_attr.to(device) if graph_data.edge_attr.numel() > 0 else None
    batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
    return model(x, edge_index, edge_attr, batch)[0].cpu()


def _classify(logits: torch.Tensor, temperature: float) -> Dict:
    """Convert logits → calibrated probabilities → verdict."""
    probabilities = F.softmax(logits / temperature, dim=0).numpy()
    if len(probabilities) >= 2:
        prob_benign = float(probabilities[0])
        prob_criminal = float(probabilities[1])
    else:
        prob_criminal = float(probabilities[0])
        prob_benign = 1.0 - prob_criminal
    risk_score = prob_criminal
    classification = "criminal" if risk_score > 0.5 else "benign"
    confidence = abs(risk_score - 0.5) * 2
    return {
        "prediction": [prob_benign, prob_criminal],
        "risk_score": risk_score,
        "classification": classification,
        "confidence": confidence,
        "message": f"Wallet classified as {classification.upper()} with {confidence*100:.1f}% confidence",
    }


def analyze_wallet_pipeline(wallet_address: str, model_path: Optional[str] = None) -> Dict:
    """Fetch transactions → build graph → run model → return verdict."""
    print(f"\n[1/4] Starting analysis for wallet: {wallet_address}")

    print("[2/4] Fetching transactions...")
    transactions = get_cached_transactions(wallet_address)
    if not transactions:
        print(f"[!] No transactions found for wallet: {wallet_address}")
        return {
            "wallet_address": wallet_address,
            "status": "no_data",
            "message": "No transaction data found for this wallet",
        }

    print("[3/4] Building ego-graph...")
    graph_data = get_cached_graph(wallet_address, transactions)
    results = {
        "wallet_address": wallet_address,
        "status": "success",
        "nodes_count": graph_data.num_nodes,
        "edges_count": graph_data.num_edges,
        "ghost_nodes": graph_data.num_ghost_nodes,
        "graph_data": _graph_descriptor(graph_data),
    }

    if not model_path:
        print("[!] No model provided, skipping inference")
        results["message"] = "Analysis completed without classification (no model provided)"
        print("\n[✓] Analysis complete!")
        return results

    print("[4/4] Running inference...")
    try:
        model, temperature, device = get_cached_model(model_path)
        with torch.no_grad():
            logits = _run_forward(model, graph_data, device)
        results.update(_classify(logits, temperature))
        print(f"   Classification: {results['classification'].upper()}")
        print(f"   Risk Score: {results['risk_score']:.4f}")
        print(f"   Confidence: {results['confidence']*100:.1f}%")
    except Exception as e:
        print(f"[!] Inference error: {e}")
        results["inference_error"] = str(e)
        results["message"] = f"Could not run inference: {e}"

    print("\n[✓] Analysis complete!")
    print(f"   Results: {results}")
    return results


def compute_feature_importance(wallet_address: str) -> Dict:
    """Gradient-based feature importance on the cached ego-graph.

    Returns the absolute gradient of the criminal-class logit w.r.t. each
    input feature on the center node, normalized to sum to 1.
    """
    try:
        print(f"[Feature Importance] Building graph for {wallet_address}...")
        txs = get_cached_transactions(wallet_address)
        if not txs:
            return {
                "status": "error",
                "message": "No transactions found for this wallet address",
            }

        try:
            graph_data = get_cached_graph(wallet_address, txs)
            print(f"[Feature Importance] Graph ready: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
        except Exception as e:
            print(f"[Feature Importance] Error building graph: {e}")
            import traceback
            traceback.print_exc()
            raise

        print("[Feature Importance] Loading model...")
        model, _temperature, device = get_cached_model()

        print("[Feature Importance] Computing feature importance...")
        x = graph_data.x.clone().requires_grad_(True).to(device)
        edge_index = graph_data.edge_index.to(device)
        edge_attr = graph_data.edge_attr.to(device)
        batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)

        output = model(x, edge_index, edge_attr, batch)
        # Saliency w.r.t. the criminal-class logit on the center node.
        output[0, 1].backward()

        gradients = x.grad[0].abs().cpu().numpy()
        total = gradients.sum()
        if total > 0:
            gradients = gradients / total

        feature_importance = {name: float(gradients[i]) for i, name in enumerate(FEATURE_NAMES)}
        sorted_features = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)
        print(f"[Feature Importance] Top 3 features: {sorted_features[:3]}")

        return {
            "status": "success",
            "feature_importance": feature_importance,
            "num_graphs_used": 1,
            "message": f"Feature importance computed for wallet {wallet_address}",
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error computing feature importance: {e}",
        }
