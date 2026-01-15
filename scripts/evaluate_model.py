"""
Model Evaluation Script
=======================
Evaluates gnn_model.pt on 2000 wallets sampled from the TEST SET ONLY
(wallets not seen during training).

Samples:
- 500 benign from Real-CATS
- 500 criminal from Real-CATS
- 500 benign from Elliptic++
- 500 criminal from Elliptic++

Fetches live transaction data from mempool API and builds graphs.
"""
import os
import sys
import asyncio
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Batch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.optimal_gnn import OptimalBitcoinGNN
from src.graph.graph_builder import EgoGraphBuilder
from src.graph.api_fetcher import MempoolFetcher
from src.graph.cache_manager import CacheManager
from src.graph.config import FEATURE_COLUMNS, NUM_NODE_FEATURES, NUM_EDGE_FEATURES


def load_and_sample_dataset(csv_path: str, samples_per_group: int = 500) -> pd.DataFrame:
    """
    Load dataset and sample wallets from each group.

    Args:
        csv_path: Path to balanced_training_dataset.csv
        samples_per_group: Number of samples per (label, source) combination

    Returns:
        DataFrame with sampled wallets
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total wallets in dataset: {len(df)}")

    # Show distribution
    print("\nDataset distribution:")
    for source in df['source'].unique():
        for label in [0, 1]:
            count = len(df[(df['source'] == source) & (df['label'] == label)])
            label_name = 'benign' if label == 0 else 'criminal'
            print(f"  {source} - {label_name}: {count}")

    # Sample from each group
    sampled_dfs = []
    for source in ['realcats', 'elliptic']:
        for label in [0, 1]:
            group = df[(df['source'] == source) & (df['label'] == label)]
            n_available = len(group)
            n_sample = min(samples_per_group, n_available)

            if n_sample < samples_per_group:
                print(f"\nWarning: Only {n_available} samples available for {source} label={label}")

            sampled = group.sample(n=n_sample, random_state=42)
            sampled_dfs.append(sampled)
            label_name = 'benign' if label == 0 else 'criminal'
            print(f"Sampled {n_sample} {label_name} from {source}")

    result = pd.concat(sampled_dfs, ignore_index=True)
    print(f"\nTotal sampled: {len(result)} wallets")
    return result


def load_model(model_path: str, device: torch.device) -> OptimalBitcoinGNN:
    """Load the trained GNN model."""
    print(f"\nLoading model from {model_path}...")
    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model


async def fetch_all_transactions(
    addresses: list,
    cache_manager: CacheManager,
    address_info: dict = None
) -> dict:
    """
    Fetch transactions for all addresses.

    Args:
        addresses: List of wallet addresses
        cache_manager: CacheManager instance
        address_info: Dict mapping address -> (label, source) for display

    Returns:
        Dict mapping address -> transactions
    """
    fetcher = MempoolFetcher(cache_manager)
    transactions = {}

    total = len(addresses)
    print(f"\nFetching transactions for {total} addresses...")
    print("-" * 100)

    success_count = 0
    failed_count = 0
    cached_count = 0

    for i, address in enumerate(addresses, 1):
        # Get address info for display
        if address_info and address in address_info:
            label, source = address_info[address]
            label_str = "criminal" if label == 1 else "benign"
            info_str = f"[{source}/{label_str}]"
        else:
            info_str = ""

        # Fetch single address
        results = await fetcher.fetch_batch([address])
        result = results[0]

        if result.success:
            transactions[result.address] = result.transactions
            tx_count = len(result.transactions)
            success_count += 1

            if result.from_cache:
                cached_count += 1
                status = "CACHED"
            else:
                status = "OK"

            print(f"[{i:4d}/{total}] {status:6s} | {address[:16]}...{address[-8:]} | {tx_count:4d} txs | {info_str}")
        else:
            transactions[result.address] = []
            failed_count += 1
            print(f"[{i:4d}/{total}] FAILED | {address[:16]}...{address[-8:]} | Error: {result.error} | {info_str}")

    print("-" * 100)
    print(f"Fetch complete: {success_count} success, {failed_count} failed, {cached_count} from cache")

    return transactions


def build_graphs(
    df: pd.DataFrame,
    transactions: dict,
    graph_builder: EgoGraphBuilder
) -> list:
    """
    Build PyG graphs for all wallets.

    Args:
        df: DataFrame with wallet features
        transactions: Dict mapping address -> transactions
        graph_builder: EgoGraphBuilder instance

    Returns:
        List of (graph, label, source, address) tuples
    """
    print("\nBuilding graphs...")
    graphs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graphs"):
        address = row['address']
        label = int(row['label'])
        source = row['source']

        # Extract features from dataset
        features = np.array([row[col] for col in FEATURE_COLUMNS], dtype=np.float32)

        # Get transactions
        txs = transactions.get(address, [])

        # Build graph
        if txs:
            graph = graph_builder.build_ego_graph(
                center_address=address,
                center_features=features,
                center_label=label,
                transactions=txs
            )
        else:
            graph = graph_builder.build_empty_graph(
                center_address=address,
                center_features=features,
                center_label=label
            )

        graphs.append((graph, label, source, address))

    print(f"Built {len(graphs)} graphs")
    return graphs


def evaluate_model(
    model: OptimalBitcoinGNN,
    graphs: list,
    device: torch.device,
    batch_size: int = 32
) -> dict:
    """
    Evaluate model on graphs.

    Args:
        model: Trained GNN model
        graphs: List of (graph, label, source, address) tuples
        device: Torch device
        batch_size: Batch size for inference

    Returns:
        Dict with predictions and labels
    """
    print(f"\nRunning inference on {len(graphs)} graphs...")

    all_preds = []
    all_labels = []
    all_probs = []
    all_sources = []
    all_addresses = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(graphs), batch_size), desc="Inference"):
            batch_data = [g[0] for g in graphs[i:i+batch_size]]
            batch_labels = [g[1] for g in graphs[i:i+batch_size]]
            batch_sources = [g[2] for g in graphs[i:i+batch_size]]
            batch_addresses = [g[3] for g in graphs[i:i+batch_size]]

            # Create batch
            batch = Batch.from_data_list(batch_data).to(device)

            # Forward pass
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels)
            all_probs.extend(probs[:, 1].cpu().numpy())  # Criminal probability
            all_sources.extend(batch_sources)
            all_addresses.extend(batch_addresses)

    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
        'sources': all_sources,
        'addresses': all_addresses
    }


def print_statistics(results: dict):
    """Print comprehensive evaluation statistics."""
    preds = results['predictions']
    labels = results['labels']
    sources = results['sources']
    probs = results['probabilities']

    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)

    # Overall metrics
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print(f"\nAccuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Benign  Criminal")
    print(f"Actual Benign    {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"Actual Criminal  {cm[1][0]:6d}  {cm[1][1]:6d}")

    # Per-class metrics
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Benign', 'Criminal']))

    # Per-source statistics
    print("\n" + "="*70)
    print("PER-DATASET STATISTICS")
    print("="*70)

    sources_arr = np.array(sources)
    for source in ['realcats', 'elliptic']:
        mask = sources_arr == source
        if not np.any(mask):
            continue

        src_preds = preds[mask]
        src_labels = labels[mask]

        src_acc = accuracy_score(src_labels, src_preds)
        src_prec = precision_score(src_labels, src_preds, zero_division=0)
        src_rec = recall_score(src_labels, src_preds, zero_division=0)
        src_f1 = f1_score(src_labels, src_preds, zero_division=0)

        print(f"\n{source.upper()} ({np.sum(mask)} samples):")
        print(f"  Accuracy:  {src_acc*100:.2f}%")
        print(f"  Precision: {src_prec*100:.2f}%")
        print(f"  Recall:    {src_rec*100:.2f}%")
        print(f"  F1 Score:  {src_f1*100:.2f}%")

        # Breakdown by class
        for label in [0, 1]:
            class_mask = (sources_arr == source) & (labels == label)
            if not np.any(class_mask):
                continue
            class_preds = preds[class_mask]
            class_labels = labels[class_mask]
            class_acc = accuracy_score(class_labels, class_preds)
            label_name = 'Benign' if label == 0 else 'Criminal'
            print(f"    {label_name}: {class_acc*100:.2f}% ({np.sum(class_preds == label)}/{np.sum(class_mask)} correct)")

    # Probability distribution analysis
    print("\n" + "="*70)
    print("PROBABILITY DISTRIBUTION")
    print("="*70)

    print("\nCriminal probability stats for benign wallets:")
    benign_probs = probs[labels == 0]
    print(f"  Mean: {benign_probs.mean():.4f}")
    print(f"  Std:  {benign_probs.std():.4f}")
    print(f"  Min:  {benign_probs.min():.4f}")
    print(f"  Max:  {benign_probs.max():.4f}")

    print("\nCriminal probability stats for criminal wallets:")
    criminal_probs = probs[labels == 1]
    print(f"  Mean: {criminal_probs.mean():.4f}")
    print(f"  Std:  {criminal_probs.std():.4f}")
    print(f"  Min:  {criminal_probs.min():.4f}")
    print(f"  Max:  {criminal_probs.max():.4f}")

    # Edge cases - high confidence mistakes
    print("\n" + "="*70)
    print("HIGH CONFIDENCE MISCLASSIFICATIONS")
    print("="*70)

    # False positives (benign predicted as criminal with high confidence)
    fp_mask = (labels == 0) & (preds == 1)
    if np.any(fp_mask):
        fp_probs = probs[fp_mask]
        print(f"\nFalse Positives (benign -> criminal): {np.sum(fp_mask)}")
        print(f"  Mean criminal prob: {fp_probs.mean():.4f}")
        print(f"  High confidence (>0.8): {np.sum(fp_probs > 0.8)}")

    # False negatives (criminal predicted as benign with high confidence)
    fn_mask = (labels == 1) & (preds == 0)
    if np.any(fn_mask):
        fn_probs = probs[fn_mask]
        print(f"\nFalse Negatives (criminal -> benign): {np.sum(fn_mask)}")
        print(f"  Mean criminal prob: {fn_probs.mean():.4f}")
        print(f"  High confidence (<0.2): {np.sum(fn_probs < 0.2)}")


async def main():
    """Main evaluation pipeline."""
    # Paths - USE TEST SET ONLY (not seen during training)
    dataset_path = os.path.join(PROJECT_ROOT, 'src', 'features', 'output', 'test_dataset.csv')
    model_path = os.path.join(PROJECT_ROOT, 'outputs', 'gnn_model.pt')

    # Check files exist
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and sample dataset
    samples_per_group = 500
    df = load_and_sample_dataset(dataset_path, samples_per_group)

    # Load model
    model = load_model(model_path, device)

    # Initialize components
    cache_manager = CacheManager(split='eval')
    graph_builder = EgoGraphBuilder()

    # Fetch transactions
    addresses = df['address'].tolist()
    # Build address info dict for display
    address_info = {row['address']: (row['label'], row['source']) for _, row in df.iterrows()}
    transactions = await fetch_all_transactions(addresses, cache_manager, address_info)

    # Build graphs
    graphs = build_graphs(df, transactions, graph_builder)

    # Evaluate
    results = evaluate_model(model, graphs, device)

    # Print statistics
    print_statistics(results)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
