"""
Pipeline Option A: Conservative (14 Features)
==============================================
Creates a unified dataset with ONLY features that are accurate for BOTH datasets.
No API calls required, no approximations.

Features (14 base + 5 engineered = 19 total):
- 11 shared features (direct mapping)
- 3 calculable features (accurate calculation from existing data)
- 5 engineered features (derived from above)

Usage:
    python pipeline_conservative.py

Output:
    output/conservative_feature_matrix.csv
"""

import pandas as pd
import numpy as np
import os
import time

from .config import (
    CONSERVATIVE_SHARED,
    CONSERVATIVE_CALCULABLE,
    CONSERVATIVE_ENGINEERED,
    COLUMN_MAPPING,
    SECONDS_PER_BLOCK,
    EPSILON,
)

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REALCATS_DIR = os.path.join(PROJECT_ROOT, 'data', 'realcats')
ELLIPTIC_DIR = os.path.join(PROJECT_ROOT, 'data', 'elliptic')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_realcats() -> pd.DataFrame:
    """Load REAL-CATS dataset (benign + behavioral criminal)."""
    print("Loading REAL-CATS data...")

    df_benign = pd.read_csv(os.path.join(REALCATS_DIR, 'BB.tsv'), sep='\t')
    df_benign['label'] = 0
    print(f"  Benign: {len(df_benign)}")

    # Load behavioral criminals only (have transaction activity)
    criminal_path = os.path.join(REALCATS_DIR, 'wallets_behavioral.tsv')
    if os.path.exists(criminal_path):
        df_criminal = pd.read_csv(criminal_path, sep='\t')
    else:
        df_criminal = pd.read_csv(os.path.join(REALCATS_DIR, 'CB.tsv'), sep='\t')
        df_criminal = df_criminal[df_criminal['transaction_number'] > 0]
    df_criminal['label'] = 1
    print(f"  Criminal: {len(df_criminal)}")

    df = pd.concat([df_benign, df_criminal], ignore_index=True)
    print(f"  Total: {len(df)}")

    return df


def load_elliptic() -> pd.DataFrame:
    """Load Elliptic++ dataset with labels."""
    print("\nLoading Elliptic++ data...")

    df = pd.read_csv(os.path.join(ELLIPTIC_DIR, 'wallets_features.csv'))

    # Take latest time step per address
    df = df.loc[df.groupby('address')['Time step'].idxmax()].reset_index(drop=True)
    print(f"  Addresses: {len(df)}")

    # Load labels
    labels_path = os.path.join(ELLIPTIC_DIR, 'wallets_classes.csv')
    if os.path.exists(labels_path):
        labels = pd.read_csv(labels_path)
        label_map = {1: 1, 2: 0, 3: -1}  # illicit=1, licit=0, unknown=-1
        labels['label'] = labels['class'].map(label_map)
        df = df.merge(labels[['address', 'label']], on='address', how='left')
        df['label'] = df['label'].fillna(-1).astype(int)
        print(f"  Illicit: {(df['label']==1).sum()}, Licit: {(df['label']==0).sum()}, Unknown: {(df['label']==-1).sum()}")

    return df


def extract_shared_features(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Extract the 11 shared features using column mapping."""
    print(f"\nExtracting shared features from {source}...")

    result = pd.DataFrame()
    result['address'] = df['address']
    result['label'] = df['label']

    for unified_name in CONSERVATIVE_SHARED:
        rc_col, el_col = COLUMN_MAPPING[unified_name]
        source_col = rc_col if source == 'realcats' else el_col

        if source_col and source_col in df.columns:
            value = pd.to_numeric(df[source_col], errors='coerce').fillna(0)

            # Convert lifetime from blocks to seconds for Elliptic++
            if unified_name == 'lifetime_seconds' and source == 'elliptic':
                value = value * SECONDS_PER_BLOCK

            result[unified_name] = value
        else:
            print(f"  Warning: {source_col} not found")
            result[unified_name] = 0

    print(f"  Extracted {len(CONSERVATIVE_SHARED)} shared features")
    return result


def calculate_additional_features(df: pd.DataFrame, source: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Calculate the 3 additional features that are accurate for both datasets."""
    print(f"\nCalculating additional features for {source}...")

    # 1. blocks_btwn_txs_mean
    if source == 'elliptic':
        # Direct from Elliptic++
        df['blocks_btwn_txs_mean'] = pd.to_numeric(
            df_raw['blocks_btwn_txs_mean'], errors='coerce'
        ).fillna(0).values
    else:
        # Calculate for REAL-CATS: lifetime / (tx_count - 1) / 600
        tx_count = df['total_txs'].values
        lifetime = df['lifetime_seconds'].values
        avg_seconds = np.where(tx_count > 1, lifetime / (tx_count - 1), 0)
        df['blocks_btwn_txs_mean'] = avg_seconds / SECONDS_PER_BLOCK

    print(f"  blocks_btwn_txs_mean - mean: {df['blocks_btwn_txs_mean'].mean():.2f}")

    # 2. fee_share_mean
    if source == 'elliptic':
        # Direct from Elliptic++
        df['fee_share_mean'] = pd.to_numeric(
            df_raw['fees_as_share_mean'], errors='coerce'
        ).fillna(0).values
    else:
        # Calculate for REAL-CATS: total_fees / total_transacted
        total_transacted = df['total_sent_btc'] + df['total_received_btc']
        df['fee_share_mean'] = np.where(
            total_transacted > 0,
            df['total_fees'] / total_transacted,
            0
        )

    df['fee_share_mean'] = df['fee_share_mean'].clip(0, 1)
    print(f"  fee_share_mean - mean: {df['fee_share_mean'].mean():.6f}")

    # 3. avg_tx_size
    if source == 'elliptic':
        # Direct from Elliptic++
        df['avg_tx_size'] = pd.to_numeric(
            df_raw['btc_transacted_mean'], errors='coerce'
        ).fillna(0).values
    else:
        # Calculate for REAL-CATS: total_transacted / tx_count
        total_transacted = df['total_sent_btc'] + df['total_received_btc']
        df['avg_tx_size'] = np.where(
            df['total_txs'] > 0,
            total_transacted / df['total_txs'],
            0
        )

    print(f"  avg_tx_size - mean: {df['avg_tx_size'].mean():.8f}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the 5 engineered features."""
    print("\nEngineering derived features...")

    # 1. send_receive_ratio
    df['send_receive_ratio'] = (
        df['total_sent_btc'] / (df['total_received_btc'] + EPSILON)
    ).clip(0, 100)
    print(f"  send_receive_ratio - mean: {df['send_receive_ratio'].mean():.4f}")

    # 2. activity_rate (transactions per day)
    seconds_per_day = 86400
    df['activity_rate'] = np.where(
        df['lifetime_seconds'] > 0,
        (df['total_txs'] / df['lifetime_seconds']) * seconds_per_day,
        0
    ).clip(0, 1000)
    print(f"  activity_rate - mean: {df['activity_rate'].mean():.4f} txs/day")

    # 3. fee_per_tx
    df['fee_per_tx'] = np.where(
        df['total_txs'] > 0,
        df['total_fees'] / df['total_txs'],
        0
    )
    print(f"  fee_per_tx - mean: {df['fee_per_tx'].mean():.8f}")

    # 4. tx_size_range
    df['tx_size_range'] = (df['max_sent'] - df['min_sent']).clip(lower=0)
    print(f"  tx_size_range - mean: {df['tx_size_range'].mean():.8f}")

    # 5. in_out_balance
    df['in_out_balance'] = (
        df['num_receive_txs'] / (df['num_send_txs'] + EPSILON)
    ).clip(0, 100)
    print(f"  in_out_balance - mean: {df['in_out_balance'].mean():.4f}")

    return df


def normalize_btc_values(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize BTC values (REAL-CATS might be in satoshis)."""
    btc_cols = ['total_sent_btc', 'total_received_btc', 'max_sent', 'min_sent',
                'max_received', 'min_received', 'total_fees']

    for col in btc_cols:
        if col in df.columns:
            median = df[col].median()
            # If median > 1000, likely satoshis
            if median > 1000:
                df[col] = df[col] / 100_000_000
                print(f"  Converted {col} from satoshis to BTC")

    return df


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed versions of numeric features."""
    numeric_cols = (
        CONSERVATIVE_SHARED +
        CONSERVATIVE_CALCULABLE +
        CONSERVATIVE_ENGINEERED
    )

    for col in numeric_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))

    return df


def main():
    """Run the conservative pipeline."""
    start_time = time.time()

    print("=" * 70)
    print("PIPELINE OPTION A: CONSERVATIVE (14 Features)")
    print("=" * 70)
    print("\nThis pipeline uses ONLY features that are accurate for BOTH datasets.")
    print("No API calls, no approximations.\n")

    # Load data
    df_realcats_raw = load_realcats()
    df_elliptic_raw = load_elliptic()

    # Process REAL-CATS
    print("\n" + "-" * 50)
    print("Processing REAL-CATS")
    print("-" * 50)

    df_rc = extract_shared_features(df_realcats_raw, 'realcats')
    df_rc = normalize_btc_values(df_rc, 'realcats')
    df_rc = calculate_additional_features(df_rc, 'realcats', df_realcats_raw)
    df_rc = engineer_features(df_rc)
    df_rc['source'] = 'realcats'

    # Process Elliptic++
    print("\n" + "-" * 50)
    print("Processing Elliptic++")
    print("-" * 50)

    df_el = extract_shared_features(df_elliptic_raw, 'elliptic')
    df_el = normalize_btc_values(df_el, 'elliptic')
    df_el = calculate_additional_features(df_el, 'elliptic', df_elliptic_raw)
    df_el = engineer_features(df_el)
    df_el['source'] = 'elliptic'

    # Combine
    print("\n" + "-" * 50)
    print("Combining datasets")
    print("-" * 50)

    df_unified = pd.concat([df_rc, df_el], ignore_index=True)

    # Add log transforms
    df_unified = add_log_transforms(df_unified)

    # Clean up
    df_unified = df_unified.replace([np.inf, -np.inf], 0).fillna(0)

    # Define final feature columns
    feature_cols = CONSERVATIVE_SHARED + CONSERVATIVE_CALCULABLE + CONSERVATIVE_ENGINEERED
    meta_cols = ['address', 'label', 'source']
    final_cols = meta_cols + feature_cols

    df_final = df_unified[final_cols].copy()

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'conservative_feature_matrix.csv')
    df_final.to_csv(output_path, index=False)

    # Also save with log transforms
    all_cols = meta_cols + feature_cols + [f'{c}_log' for c in feature_cols if f'{c}_log' in df_unified.columns]
    df_full = df_unified[[c for c in all_cols if c in df_unified.columns]]
    full_output_path = os.path.join(OUTPUT_DIR, 'conservative_feature_matrix_with_logs.csv')
    df_full.to_csv(full_output_path, index=False)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(f"\nTime: {elapsed:.2f} seconds")
    print(f"\nOutput: {output_path}")
    print(f"        {full_output_path}")

    print(f"\nDataset Summary:")
    print(f"  Total addresses: {len(df_final)}")
    print(f"  From REAL-CATS: {(df_final['source'] == 'realcats').sum()}")
    print(f"  From Elliptic++: {(df_final['source'] == 'elliptic').sum()}")

    print(f"\nLabel Distribution:")
    print(f"  Benign/Licit (0): {(df_final['label'] == 0).sum()}")
    print(f"  Criminal/Illicit (1): {(df_final['label'] == 1).sum()}")
    print(f"  Unknown (-1): {(df_final['label'] == -1).sum()}")

    print(f"\nFeatures ({len(feature_cols)}):")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {feat}")

    print("\n" + "=" * 70)

    return df_final


if __name__ == "__main__":
    df = main()
