import pandas as pd
import numpy as np
import torch
import requests
import time
import os
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.model_selection import train_test_split


# Configuration - Extended feature set from REAL-CATS dataset
NUMERIC_FEATURES = [
    # VOLUME (Size of operation)
    'balance', 
    'total_received_USD', 
    'total_sent_USD',
    
    # VELOCITY (Speed/Intensity)
    'lifetime',
    'transaction_number',
    'activity_w',
    'activity_d',
    'activity_time',
    
    # BEHAVIOR (Structure)
    'transaction_fee',
    'transaction_fee_Variance',
    'received_Variance_USD',
    'sent_Variance_USD',
    'total_input_slots',
    'total_output_slots',
    'payment_transactions',
    'receipt_transactions',
    
    # ENGINEERED FEATURES
    'flow_ratio',
    'fan_ratio'
]

# Fallback features for wallets without full REAL-CATS data
BASIC_FEATURES = [
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


def process_and_save_tensors(wallet_address, df_edges, feature_set='basic'):
    """
    Build graph tensors from wallet data.
    
    Args:
        wallet_address: The wallet address to analyze
        df_edges: DataFrame with transaction edges
        feature_set: 'basic' (for single wallet) or 'full' (for training with REAL-CATS features)
    
    Returns:
        Dictionary with tensors and metadata
    """
    print("\n4. Building Tensors (with Log Scaling)...")
    
    # Select feature set
    features_to_use = NUMERIC_FEATURES if feature_set == 'full' else BASIC_FEATURES
    
    # Create a single-node dataframe for the wallet address
    df_nodes = pd.DataFrame({
        'address': [wallet_address],
        'label': [1]  # Default label for the query address
    })
    
    # Add default numeric features
    for feature in features_to_use:
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
    print(f"   Scaling {len(features_to_use)} features...")
    df_scaled = df_nodes.copy()
    
    for c in features_to_use:
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
    known_feats = scaler.fit_transform(df_scaled[features_to_use])
    
    # E. Create Matrix
    x_np = np.zeros((len(all_nodes), len(features_to_use)))
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


# ============================================
# MODEL ARCHITECTURE (GNN with GAT layers)
# ============================================

class CryptoGNN(torch.nn.Module):
    """Graph Attention Network for cryptocurrency wallet risk classification"""
    
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


# ============================================
# TRAINING UTILITIES
# ============================================

def load_and_label_datasets(path_benign, path_criminal):
    """Load benign and criminal datasets and add labels"""
    print("1. Loading and Labeling Data...")
    
    # A. Load Benign (Label = 0)
    try:
        df_b = pd.read_csv(path_benign, sep="\t")
        df_b['label'] = 0
        print(f"   Loaded {len(df_b)} Benign records.")
    except Exception as e:
        print(f"   Error loading Benign: {e}")
        df_b = pd.DataFrame()

    # B. Load Criminal (Label = 1)
    try:
        df_c = pd.read_csv(path_criminal, sep="\t")
        df_c['label'] = 1
        print(f"   Loaded {len(df_c)} Criminal records.")
    except Exception as e:
        print(f"   Error loading Criminal: {e}")
        df_c = pd.DataFrame()

    return df_b, df_c


def merge_datasets(df_benign, df_criminal):
    """Merge benign and criminal datasets, keeping only common columns"""
    print("2. Merging Datasets...")
    
    common_cols = list(set(df_benign.columns) & set(df_criminal.columns))
    df_merged = pd.concat([df_benign[common_cols], df_criminal[common_cols]], ignore_index=True)
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   Common columns: {len(common_cols)}")
    print(f"   Total Records Merged: {len(df_merged)}")
    return df_merged


def perform_feature_engineering(df):
    """Create engineered features from wallet data"""
    print("2b. Engineering Features (Cleaning & Creating Ratios)...")
    
    # Create Ratios (Safe division to avoid /0 error)
    df['flow_ratio'] = df['total_sent_USD'] / (df['total_received_USD'] + 1e-5)
    df['fan_ratio'] = df['total_output_slots'] / (df['total_input_slots'] + 1e-5)
    
    print(f"   Created flow_ratio and fan_ratio features")
    return df


def fetch_edges_mempool_batch(df_nodes):
    """
    Fetch edges for multiple wallets (for training).
    
    Args:
        df_nodes: DataFrame with wallet addresses and metadata
    
    Returns:
        DataFrame with directed edges
    """
    print("3. Fetching DIRECTED Graph Connections (Batch Mode)...")
    edges_list = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for i, row in df_nodes.iterrows():
        my_address = row['address']
        print(f"   [{i+1}/{len(df_nodes)}] Processing: {my_address}...")
        
        # Parse Time Window
        try:
            start_ts = pd.to_datetime(row['first_time']).timestamp()
            end_ts = pd.to_datetime(row['last_time']).timestamp()
        except:
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
                            amount = out.get('value', 0)
                            
                            if recipient and recipient != my_address:
                                edges_list.append({
                                    'source': my_address,
                                    'target': recipient,
                                    'weight': amount, 
                                    'timestamp': tx_time,
                                    'direction': 'outgoing'
                                })

                    # --- CHECK INCOMING (Am I an Output?) ---
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
                                    'weight': amount_received,
                                    'timestamp': tx_time,
                                    'direction': 'incoming'
                                })
                                
            time.sleep(1)  # Be polite
            
        except Exception as e:
            print(f"       !! Error: {e}")

    df_edges = pd.DataFrame(edges_list)
    print(f"\n   DONE. Found {len(df_edges)} Directed Edges.")
    return df_edges


def process_full_dataset_tensors(df_nodes, df_edges, output_dir):
    """
    Build and save tensors for full training dataset.
    
    Args:
        df_nodes: DataFrame with wallet nodes and features
        df_edges: DataFrame with transaction edges
        output_dir: Directory to save tensor files
    
    Returns:
        Dictionary with tensors
    """
    print("\n4. Building Tensors for Full Dataset...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # A. Clean Edges
    if not df_edges.empty:
        df_edges = df_edges.drop_duplicates(subset=['source', 'target', 'timestamp'])
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
            series = pd.to_numeric(df_scaled[c], errors='coerce').fillna(0)
            df_scaled[c] = np.log1p(np.maximum(0, series))
        else:
            df_scaled[c] = 0.0

    # D. Standardization
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

    # Save tensors
    torch.save(x, os.path.join(output_dir, 'x.pt'))
    torch.save(y, os.path.join(output_dir, 'y.pt'))
    torch.save(edge_index, os.path.join(output_dir, 'edge_index.pt'))
    torch.save(edge_attr, os.path.join(output_dir, 'edge_attr.pt'))
    
    print(f"   Saved Tensors to {output_dir}. X shape: {x.shape}")
    
    return {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'edge_attr': edge_attr
    }


def train_model(data_dir, output_model_path, epochs=51):
    """
    Train the GNN model on prepared dataset.
    
    Args:
        data_dir: Directory with tensor files
        output_model_path: Path to save trained model
        epochs: Number of training epochs
    
    Returns:
        Trained model
    """
    print("\n5. Training Model...")
    
    # Load tensors
    x = torch.load(os.path.join(data_dir, 'x.pt'))
    y = torch.load(os.path.join(data_dir, 'y.pt'))
    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt'))
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr.pt'))

    # Split (Only known nodes)
    valid_idx = torch.where(y != -1)[0].numpy()
    if len(valid_idx) > 1:
        train_idx, test_idx = train_test_split(valid_idx, test_size=0.2, stratify=y[valid_idx], random_state=42)
    else:
        train_idx, test_idx = valid_idx, valid_idx

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CryptoGNN(x.shape[1], edge_attr.shape[1], 16, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Move data to device
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index, edge_attr)
        loss = criterion(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch} | Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_attr)
        pred = out.argmax(dim=1)
        correct = (pred[test_idx] == y[test_idx]).sum()
        acc = int(correct) / len(test_idx) if len(test_idx) > 0 else 0
        print(f"   Test Accuracy: {acc:.2f}")
    
    # Save model
    torch.save(model, output_model_path)
    print(f"   Model saved to {output_model_path}")
    
    return model


# ============================================
# INFERENCE PIPELINE (API Usage)
# ============================================


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
    
    # Preprocess and create tensors (using basic feature set for single wallet analysis)
    print(f"[3/4] Preprocessing graph data...")
    graph_data = process_and_save_tensors(wallet_address, df_edges, feature_set='basic')
    
    # Run inference if model provided
    results = {
        "wallet_address": wallet_address,
        "status": "success",
        "nodes_count": len(graph_data['all_nodes']),
        "edges_count": graph_data['edge_index'].shape[1],
        "graph_data": {
            "x_shape": list(graph_data['x'].shape),
            "y_shape": list(graph_data['y'].shape),
            "edge_index_shape": list(graph_data['edge_index'].shape),
            "edge_attr_shape": list(graph_data['edge_attr'].shape)
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
                risk_score = float(prediction[1]) if len(prediction) > 1 else float(prediction[0])
                
                results["prediction"] = prediction.tolist() if hasattr(prediction, 'tolist') else list(prediction)
                results["risk_score"] = risk_score
        except Exception as e:
            print(f"[!] Inference error: {str(e)}")
            results["inference_error"] = str(e)
    else:
        print(f"[!] No model provided, skipping inference")
    
    print(f"\n[✓] Analysis complete!")
    return results