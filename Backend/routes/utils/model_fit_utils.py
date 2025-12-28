import pandas as pd
import numpy as np
import torch
import requests
import time
import os
from sklearn.preprocessing import StandardScaler


# Configuration - Update these based on your dataset
NUMERIC_FEATURES = [
    'in_degree', 'out_degree', 'pagerank', 'clustering_coefficient',
    'betweenness', 'closeness', 'eigenvector', 'harmonic'
]


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
    
    # Create a single-node dataframe for the wallet address
    df_nodes = pd.DataFrame({
        'address': [wallet_address],
        'label': [1]  # Default label for the query address
    })
    
    # Add default numeric features
    for feature in NUMERIC_FEATURES:
        df_nodes[feature] = 0.0
    
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

    # D. Standardization (Z-Score)
    scaler = StandardScaler()
    known_feats = scaler.fit_transform(df_scaled[NUMERIC_FEATURES])
    
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
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.load(model_path, map_location=device)
            model.eval()
            
            with torch.no_grad():
                x = graph_data['x'].to(device)
                edge_index = graph_data['edge_index'].to(device)
                edge_attr = graph_data['edge_attr'].to(device) if graph_data['edge_attr'].numel() > 0 else None
                
                if edge_attr is not None:
                    output = model(x, edge_index, edge_attr)
                else:
                    output = model(x, edge_index)
                
                prediction = output[0].cpu().numpy()
                risk_score = float(prediction[0]) if len(prediction) > 0 else 0.0
                
                results["prediction"] = prediction.tolist() if hasattr(prediction, 'tolist') else prediction
                results["risk_score"] = risk_score
        except Exception as e:
            print(f"[!] Inference error: {str(e)}")
            results["inference_error"] = str(e)
    else:
        print(f"[!] No model provided, skipping inference")
    
    print(f"\n[✓] Analysis complete!")
    print(f"   Results: {results}")
    return results
