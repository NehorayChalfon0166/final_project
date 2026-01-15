import numpy as np
import torch
import torch.nn.functional as F
import requests
import os
import sys
from typing import List, Dict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.graph.graph_builder import EgoGraphBuilder
from src.graph.config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES
from src.models.optimal_gnn import OptimalBitcoinGNN


def fetch_transactions_mempool(wallet_address: str) -> List[Dict]:
    """
    Fetch raw transactions for a wallet address from Mempool API.
    
    Args:
        wallet_address: The Bitcoin wallet address
        
    Returns:
        List of raw transaction dictionaries
    """
    print(f"   Fetching transactions from Mempool.space for {wallet_address}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        url = f"https://mempool.space/api/address/{wallet_address}/txs"
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
    
    # Fetch transactions from mempool
    print(f"[2/4] Fetching transactions...")
    transactions = fetch_transactions_mempool(wallet_address)
    
    if not transactions:
        print(f"[!] No transactions found for wallet: {wallet_address}")
        return {
            "wallet_address": wallet_address,
            "status": "no_data",
            "message": "No transaction data found for this wallet"
        }
    
    # Build ego-graph using EgoGraphBuilder
    print(f"[3/4] Building ego-graph...")
    graph_builder = EgoGraphBuilder()
    graph_data = graph_builder.build_graph_for_new_address(
        address=wallet_address,
        transactions=transactions,
        label=-1  # Unknown label
    )
    
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
            # Resolve model path - try multiple locations
            resolved_path = None
            
            # Get project root directory (3 levels up from this file)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            
            # List of paths to try
            paths_to_try = [
                model_path,  # Original path
                os.path.join(project_root, 'outputs', 'gnn_model.pt'),  # Project root outputs
                os.path.join(project_root, 'outputs', 'gnn_checkpoint.pt'),  # Checkpoint
                os.path.abspath(model_path),  # Absolute version of original
            ]
            
            for path in paths_to_try:
                if os.path.exists(path):
                    resolved_path = path
                    break
            
            if not resolved_path:
                raise FileNotFoundError(f"Model not found. Tried: {paths_to_try}")
            
            print(f"   Loading model from: {resolved_path}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create model instance with correct architecture
            model = OptimalBitcoinGNN(
                num_node_features=NUM_NODE_FEATURES,
                num_edge_features=NUM_EDGE_FEATURES
            )
            
            # Load state dict (weights)
            state_dict = torch.load(resolved_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
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
