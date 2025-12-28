"""
Training script for cryptocurrency wallet risk classification model.
Uses REAL-CATS dataset with benign and criminal wallet data.
"""

import os
import sys

# Add Backend to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routes.utils.model_fit_utils import (
    load_and_label_datasets,
    merge_datasets,
    perform_feature_engineering,
    fetch_edges_mempool_batch,
    process_full_dataset_tensors,
    train_model
)

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_CATS_DATA_DIR = os.path.join(PROJECT_ROOT, 'Real_Cats_data')
FULL_DATASET_DIR = os.path.join(REAL_CATS_DATA_DIR, 'full_dataset')
PATH_BENIGN = os.path.join(REAL_CATS_DATA_DIR, 'BB.tsv')
PATH_CRIMINAL = os.path.join(REAL_CATS_DATA_DIR, 'wallets_behavioral.tsv')
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'models', 'crypto_gnn_model.pt')

# Ensure directories exist
os.makedirs(FULL_DATASET_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)


def split_criminal_wallets(input_path, output_dir):
    """
    Split criminal wallet dataset into behavioral and non-behavioral wallets.
    
    Args:
        input_path: Path to CB.tsv file
        output_dir: Directory to save split files
    """
    import pandas as pd
    
    print("0. Splitting Criminal Wallets...")
    df = pd.read_csv(input_path, sep='\t', low_memory=False)
    
    behavioral_wallets = df[
        (df["transaction_number"] > 0) |
        (df["total_received_BTC"] > 0) |
        (df["total_sent_BTC"] > 0)
    ].copy()

    non_behavioral_wallets = df[
        (df["transaction_number"] == 0) &
        (df["total_received_BTC"] == 0) &
        (df["total_sent_BTC"] == 0)
    ].copy()

    print(f"   Total wallets: {len(df)}")
    print(f"   Behavioral: {len(behavioral_wallets)}")
    print(f"   Non-behavioral: {len(non_behavioral_wallets)}")

    behavioral_path = os.path.join(output_dir, 'wallets_behavioral.tsv')
    behavioral_wallets.to_csv(behavioral_path, sep='\t', index=False)
    
    print(f"   Saved to {behavioral_path}")
    return behavioral_wallets


def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("Cryptocurrency Wallet Risk Classification - Training Pipeline")
    print("=" * 60)
    
    # Step 0: Split criminal wallets if needed
    path_cb = os.path.join(REAL_CATS_DATA_DIR, 'CB.tsv')
    if os.path.exists(path_cb) and not os.path.exists(PATH_CRIMINAL):
        split_criminal_wallets(path_cb, REAL_CATS_DATA_DIR)
    
    # Step 1: Load and label datasets
    df_benign, df_criminal = load_and_label_datasets(PATH_BENIGN, PATH_CRIMINAL)
    
    if df_benign.empty or df_criminal.empty:
        print("ERROR: Could not load datasets. Check file paths.")
        return
    
    # Step 2: Merge datasets
    df_merged = merge_datasets(df_benign, df_criminal)
    
    # Optional: Sample for faster training (remove for full dataset)
    # df_merged = df_merged.head(100)
    # print(f"   Using sample of {len(df_merged)} wallets for training")
    
    # Step 3: Feature engineering
    df_merged = perform_feature_engineering(df_merged)
    
    # Step 4: Fetch transaction graph from mempool
    # WARNING: This will make API calls. Use small sample for testing.
    print("\nWARNING: About to fetch transaction data from mempool.space API")
    print(f"This will process {len(df_merged)} wallets")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Training aborted.")
        return
    
    df_edges = fetch_edges_mempool_batch(df_merged)
    
    # Step 5: Build and save tensors
    tensors = process_full_dataset_tensors(df_merged, df_edges, FULL_DATASET_DIR)
    
    # Step 6: Train model
    model = train_model(FULL_DATASET_DIR, MODEL_OUTPUT_PATH, epochs=51)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
