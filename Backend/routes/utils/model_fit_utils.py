import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import requests
import time
import os
from sklearn.preprocessing import StandardScaler


# Configuration - Update these based on your dataset
# These must match the features used during model training (18 features from REAL-CATS)
NUMERIC_FEATURES = [
    'balance', 
    'total_received_USD', 
    'total_sent_USD', 
    'transaction_fee', 
    'transaction_fee_Variance',
    'max_sent_amount', 
    'min_sent_amount',
    'lifetime', 
    'total_output_slots', 
    'total_input_slots', 
    'activity_w', 
    'activity_d', 
    'activity_time',
    'transaction_number',
    'payment_transactions',
    'receipt_transactions',
    'received_Variance_USD',
    'sent_Variance_USD'
]


# Define the GNN model architecture (must match training)
class CryptoGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=2, edge_dim=num_edge_features)
        self.conv2 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=1, edge_dim=num_edge_features)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = h.relu()
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = h.relu()
        return self.classifier(h)


def fetch_edges_mempool_directed(wallet_address):
    print("2. Fetching DIRECTED Graph Connections (Source: Mempool.space)...")
    edges_list = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    my_address = wallet_address
    print(f"   Processing: {my_address}...")
    
    # No time window filtering for single address
    start_ts, end_ts = 0, 9999999999

    # No time window filtering for single address
    start_ts, end_ts = 0, 9999999999

    try:
        url = f"https://mempool.space/api/address/{my_address}/txs"
        r = requests.get(url, headers=headers, timeout=10)
        
        if r.status_code == 200:
            txs = r.json()
            
            for tx in txs:
                # 1. Check Time
                if not tx.get('status', {}).get('confirmed'): continue
                tx_time = tx['status']['block_time']
                if tx_time < start_ts or tx_time > end_ts: continue
                
                # 2. Identify My Role (Sender or Receiver?)
                
                # --- CHECK OUTGOING (Am I an Input?) ---
                is_sender = False
                for inp in tx.get('vin', []):
                    if inp.get('prevout', {}).get('scriptpubkey_address') == my_address:
                        is_sender = True
                        break
                
                if is_sender:
                    # I am the SOURCE. I connect to all Outputs.
                    for out in tx.get('vout', []):
                        recipient = out.get('scriptpubkey_address')
                        amount = out.get('value', 0) # Amount in Satoshis
                        
                        if recipient and recipient != my_address:
                            edges_list.append({
                                'source': my_address,
                                'target': recipient,
                                'weight': amount, 
                                'timestamp': tx_time,
                                'direction': 'outgoing'
                            })

                # --- CHECK INCOMING (Am I an Output?) ---
                # Note: You can be both sender and receiver (Change address), 
                # but we excluded self-loops above (recipient != my_address).
                
                # Find out how much I received specifically
                amount_received = 0
                is_receiver = False
                for out in tx.get('vout', []):
                    if out.get('scriptpubkey_address') == my_address:
                        amount_received += out.get('value', 0)
                        is_receiver = True
                
                if is_receiver:
                    # I am the TARGET. All Inputs connect to me.
                    for inp in tx.get('vin', []):
                        sender = inp.get('prevout', {}).get('scriptpubkey_address')
                        
                        if sender and sender != my_address:
                            edges_list.append({
                                'source': sender,
                                'target': my_address,
                                'weight': amount_received, # We attribute the full receive amount to the link
                                'timestamp': tx_time,
                                'direction': 'incoming'
                            })
                            
        time.sleep(1) # Be polite
        
    except Exception as e:
        print(f"       !! Error: {e}")

    df_edges = pd.DataFrame(edges_list)
    print(f"\n   DONE. Found {len(df_edges)} Directed Edges.")
    print(df_edges.head())
    return df_edges


def process_and_save_tensors(wallet_address, df_edges):
    print("\n4. Building Tensors (with Log Scaling)...")
    
    # Calculate basic features from transaction data
    incoming = df_edges[df_edges['direction'] == 'incoming'] if not df_edges.empty else pd.DataFrame()
    outgoing = df_edges[df_edges['direction'] == 'outgoing'] if not df_edges.empty else pd.DataFrame()
    
    total_received = incoming['weight'].sum() if not incoming.empty else 0
    total_sent = outgoing['weight'].sum() if not outgoing.empty else 0
    transaction_count = len(df_edges) if not df_edges.empty else 0
    
    # Create a single-node dataframe for the wallet address with computed features
    df_nodes = pd.DataFrame({
        'address': [wallet_address],
        'label': [1],  # Default label for the query address
        'balance': [total_received - total_sent],
        'total_received_USD': [total_received / 1e8],  # Convert satoshis to BTC
        'total_sent_USD': [total_sent / 1e8],
        'transaction_fee': [0],  # Not available from API
        'transaction_fee_Variance': [0],
        'max_sent_amount': [outgoing['weight'].max() / 1e8 if not outgoing.empty else 0],
        'min_sent_amount': [outgoing['weight'].min() / 1e8 if not outgoing.empty else 0],
        'lifetime': [0],  # Would need earliest/latest timestamps
        'total_output_slots': [len(outgoing)],
        'total_input_slots': [len(incoming)],
        'activity_w': [0],  # Would need weekly data
        'activity_d': [0],  # Would need daily data
        'activity_time': [0],
        'transaction_number': [transaction_count],
        'payment_transactions': [len(outgoing)],
        'receipt_transactions': [len(incoming)],
        'received_Variance_USD': [incoming['weight'].std() / 1e8 if not incoming.empty else 0],
        'sent_Variance_USD': [outgoing['weight'].std() / 1e8 if not outgoing.empty else 0]
    })
    
    # A. Clean Edges
    if not df_edges.empty:
        df_edges = df_edges.drop_duplicates(subset=['source', 'target', 'timestamp'])
        # Log Scale Edge Weights (Handling potential negatives just in case)
        w = pd.to_numeric(df_edges['weight'], errors='coerce').fillna(0)
        df_edges['weight_log'] = np.log1p(np.maximum(0, w))
    else:
        df_edges = pd.DataFrame(columns=['source', 'target', 'weight_log'])

    # B. Map Addresses
    known_addrs = df_nodes['address'].tolist()
    ghost_addrs = list(set(df_edges['source']).union(set(df_edges['target'])) - set(known_addrs))
    all_nodes = known_addrs + ghost_addrs
    addr_map = {addr: i for i, addr in enumerate(all_nodes)}
    
    print(f"   Nodes: {len(all_nodes)} ({len(known_addrs)} Known + {len(ghost_addrs)} Ghosts)")

    # C. Build X (Features)
    print(f"   Scaling {len(NUMERIC_FEATURES)} features...")
    df_scaled = df_nodes.copy()
    
    for c in NUMERIC_FEATURES:
        if c in df_scaled.columns:
            # 1. Force Numeric & Fill NA
            series = pd.to_numeric(df_scaled[c], errors='coerce').fillna(0)
            
            # 2. Clip Negatives (Safety) & Log Scale
            # We use maximum(0, x) because log(-1) is NaN
            df_scaled[c] = np.log1p(np.maximum(0, series))
        else:
            df_scaled[c] = 0.0

    # D. Get features (skip StandardScaler - it doesn't work on single samples)
    # Just use log-scaled features directly
    known_feats = df_scaled[NUMERIC_FEATURES].values
    
    # E. Create Matrix
    x_np = np.zeros((len(all_nodes), len(NUMERIC_FEATURES)))
    x_np[0:len(known_addrs)] = known_feats
    x = torch.tensor(x_np, dtype=torch.float)
    
    # F. Labels
    y_np = np.full(len(all_nodes), -1)
    y_np[0:len(known_addrs)] = df_nodes['label'].values
    y = torch.tensor(y_np, dtype=torch.long)
    
    # G. Edges
    if not df_edges.empty:
        src = df_edges['source'].map(addr_map).values
        dst = df_edges['target'].map(addr_map).values
        
        mask = (~np.isnan(src)) & (~np.isnan(dst))
        edge_index = torch.tensor([src[mask], dst[mask]], dtype=torch.long)
        edge_attr = torch.tensor(df_edges['weight_log'].to_numpy()[mask].reshape(-1, 1), dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    # Return preprocessed tensors and metadata
    print(f"   Preprocessed Tensors. X shape: {x.shape}")
    return {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'addr_map': addr_map,
        'all_nodes': all_nodes
    }


def analyze_wallet_pipeline(wallet_address: str, model_path: str = None):
    """
    Complete pipeline: Fetch edges, preprocess, and run model inference.
    
    Args:
        wallet_address: The wallet address to analyze
        model_path: Path to saved model (optional)
    
    Returns:
        Dictionary with analysis results
    """
    print(f"\n[1/4] Starting analysis for wallet: {wallet_address}")
    
    # Fetch edges from mempool
    print(f"[2/4] Fetching transaction graph...")
    df_edges = fetch_edges_mempool_directed(wallet_address)
    
    if df_edges.empty:
        print(f"[!] No transactions found for wallet: {wallet_address}")
        return {
            "wallet_address": wallet_address,
            "status": "no_data",
            "message": "No transaction data found for this wallet"
        }
    
    # Preprocess and create tensors
    print(f"[3/4] Preprocessing graph data...")
    graph_data = process_and_save_tensors(wallet_address, df_edges)
    
    # Run inference if model provided
    results = {
        "wallet_address": wallet_address,
        "status": "success",
        "nodes_count": len(graph_data['all_nodes']),
        "edges_count": graph_data['edge_index'].shape[1],
        "graph_data": {
            "x_shape": graph_data['x'].shape,
            "y_shape": graph_data['y'].shape,
            "edge_index_shape": graph_data['edge_index'].shape,
            "edge_attr_shape": graph_data['edge_attr'].shape
        }
    }
    
    if model_path:
        print(f"[4/4] Running inference...")
        try:
            import os
            # Check if model file exists
            if not os.path.exists(model_path):
                # Try relative to Backend directory
                alt_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'crypto_gnn_model.pt')
                if os.path.exists(alt_path):
                    model_path = alt_path
                else:
                    raise FileNotFoundError(f"Model not found at {model_path} or {alt_path}")
            
            print(f"   Loading model from: {model_path}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model with weights_only=False to allow custom CryptoGNN class
            model = torch.load(model_path, map_location=device, weights_only=False)
            model.eval()
            
            with torch.no_grad():
                x = graph_data['x'].to(device)
                edge_index = graph_data['edge_index'].to(device)
                edge_attr = graph_data['edge_attr'].to(device) if graph_data['edge_attr'].numel() > 0 else None
                
                if edge_attr is not None:
                    output = model(x, edge_index, edge_attr)
                else:
                    output = model(x, edge_index)
                
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
                # Distance from 0.5, scaled to 0-1 range
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
