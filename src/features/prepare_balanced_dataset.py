"""
Prepare Balanced Dataset for Training
======================================
Creates a balanced dataset using:
- ALL illicit/criminal addresses from both datasets
- SAMPLED licit/benign addresses to match

Input:  conservative_feature_matrix.csv (from pipeline_conservative.py)
Output: balanced_training_dataset.csv

Strategy:
- REAL-CATS:  40,032 criminal + 40,032 benign (sampled)
- Elliptic++: 14,266 illicit + 14,266 licit (sampled)
- Total:      54,298 criminal + 54,298 benign = 108,596 addresses
"""

import pandas as pd
import numpy as np
import os

# Configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(CURRENT_DIR, 'output')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'output')

# Random seed for reproducibility
RANDOM_SEED = 42


def load_feature_matrix(use_log: bool = True) -> pd.DataFrame:
    """Load the conservative feature matrix."""

    if use_log:
        filename = 'conservative_feature_matrix_with_logs.csv'
    else:
        filename = 'conservative_feature_matrix.csv'

    filepath = os.path.join(INPUT_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Feature matrix not found: {filepath}\n"
            "Please run pipeline_conservative.py first."
        )

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} addresses from {filename}")

    return df


def analyze_label_distribution(df: pd.DataFrame) -> dict:
    """Analyze current label distribution by source."""

    print("\n" + "=" * 60)
    print("Current Label Distribution")
    print("=" * 60)

    stats = {}

    for source in ['realcats', 'elliptic']:
        df_source = df[df['source'] == source]

        benign = (df_source['label'] == 0).sum()
        criminal = (df_source['label'] == 1).sum()
        unknown = (df_source['label'] == -1).sum()

        stats[source] = {
            'benign': benign,
            'criminal': criminal,
            'unknown': unknown,
            'total': len(df_source)
        }

        print(f"\n{source.upper()}:")
        print(f"  Benign/Licit (0):     {benign:,}")
        print(f"  Criminal/Illicit (1): {criminal:,}")
        print(f"  Unknown (-1):         {unknown:,}")
        print(f"  Total:                {len(df_source):,}")

    return stats


def deduplicate_cross_source(df: pd.DataFrame) -> pd.DataFrame:
    """Drop addresses that appear in both REAL-CATS and Elliptic++ with conflicting labels,
    and collapse same-label duplicates to a single row.

    A handful of addresses appear in both source datasets. Three cases:
      - Same label in both sources → keep one row (the REAL-CATS copy by convention).
      - Conflicting labels → drop the address entirely (we can't tell which is right).
      - Already filtered: -1/unknown was removed earlier in main().
    """
    print("\n" + "=" * 60)
    print("Deduplicating Cross-Source Addresses")
    print("=" * 60)

    # Group by address; flag conflicts and same-label dups.
    counts = df.groupby('address')['label'].nunique()
    conflicting = counts[counts > 1].index
    duplicated_addrs = df[df['address'].duplicated(keep=False)]['address'].unique()

    n_conflicts = len(conflicting)
    n_same_label_dups = len(set(duplicated_addrs) - set(conflicting))
    print(f"  Cross-source addresses with conflicting labels: {n_conflicts:,} (dropping)")
    print(f"  Cross-source addresses with consistent labels:  {n_same_label_dups:,} (collapsing to one row)")

    # Drop conflicting addresses entirely.
    df = df[~df['address'].isin(conflicting)].copy()

    # For consistent duplicates, prefer the REAL-CATS row so source distribution stays comparable.
    df = df.sort_values('source', ascending=False).drop_duplicates(subset='address', keep='first')

    print(f"  Rows after dedup: {len(df):,}")
    return df


def create_balanced_dataset(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Create balanced dataset:
    - Take ALL criminal/illicit from both sources
    - Sample benign/licit to match criminal count per source
    """

    print("\n" + "=" * 60)
    print("Creating Balanced Dataset")
    print("=" * 60)

    balanced_dfs = []

    for source in ['realcats', 'elliptic']:
        df_source = df[df['source'] == source]

        # Get all criminal/illicit (label = 1)
        df_criminal = df_source[df_source['label'] == 1].copy()
        n_criminal = len(df_criminal)

        # Sample benign/licit (label = 0) to match
        df_benign_all = df_source[df_source['label'] == 0]

        if len(df_benign_all) >= n_criminal:
            df_benign = df_benign_all.sample(n=n_criminal, random_state=RANDOM_SEED)
        else:
            # If not enough benign, take all available
            df_benign = df_benign_all.copy()
            print(f"  Warning: {source} has only {len(df_benign_all):,} benign, needed {n_criminal:,}")

        print(f"\n{source.upper()}:")
        print(f"  Criminal/Illicit: {len(df_criminal):,} (all)")
        print(f"  Benign/Licit:     {len(df_benign):,} (sampled)")

        balanced_dfs.append(df_criminal)
        balanced_dfs.append(df_benign)

    # Combine all
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)

    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\nTotal balanced dataset: {len(df_balanced):,} addresses")

    return df_balanced


def select_features(df: pd.DataFrame, use_log: bool = True) -> pd.DataFrame:
    """Select the appropriate feature columns."""

    # Base features (non-log)
    base_features = [
        'total_txs', 'total_sent_btc', 'total_received_btc', 'total_fees',
        'lifetime_seconds', 'num_send_txs', 'num_receive_txs',
        'max_sent', 'min_sent', 'max_received', 'min_received',
        'blocks_btwn_txs_mean', 'fee_share_mean', 'avg_tx_size',
        'send_receive_ratio', 'activity_rate', 'fee_per_tx',
        'tx_size_range', 'in_out_balance'
    ]

    if use_log:
        # Use log-transformed versions where available
        feature_cols = []
        for feat in base_features:
            log_feat = f'{feat}_log'
            if log_feat in df.columns:
                feature_cols.append(log_feat)
            elif feat in df.columns:
                feature_cols.append(feat)
        print(f"\nUsing LOG-SCALED features ({len(feature_cols)} features)")
    else:
        feature_cols = [f for f in base_features if f in df.columns]
        print(f"\nUsing RAW features ({len(feature_cols)} features)")

    # Metadata columns
    meta_cols = ['address', 'label', 'source']

    # Select columns
    all_cols = meta_cols + feature_cols
    df_selected = df[all_cols].copy()

    return df_selected, feature_cols


def create_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """
    Create stratified train/test split.
    Stratified by both label AND source to ensure balanced representation.
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 60)
    print(f"Creating Train/Test Split ({int((1-test_size)*100)}/{int(test_size*100)})")
    print("=" * 60)

    # Create stratification key combining label and source
    df['stratify_key'] = df['label'].astype(str) + '_' + df['source']

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=df['stratify_key']
    )

    # Remove stratify key
    train_df = train_df.drop(columns=['stratify_key'])
    test_df = test_df.drop(columns=['stratify_key'])

    print(f"\nTrain set: {len(train_df):,} addresses")
    print(f"  Benign:   {(train_df['label'] == 0).sum():,}")
    print(f"  Criminal: {(train_df['label'] == 1).sum():,}")

    print(f"\nTest set: {len(test_df):,} addresses")
    print(f"  Benign:   {(test_df['label'] == 0).sum():,}")
    print(f"  Criminal: {(test_df['label'] == 1).sum():,}")

    return train_df, test_df


def save_datasets(df_balanced: pd.DataFrame, train_df: pd.DataFrame,
                  test_df: pd.DataFrame, feature_cols: list):
    """Save all datasets to CSV."""

    print("\n" + "=" * 60)
    print("Saving Datasets")
    print("=" * 60)

    # Save balanced full dataset
    balanced_path = os.path.join(OUTPUT_DIR, 'balanced_training_dataset.csv')
    df_balanced.to_csv(balanced_path, index=False)
    print(f"\nSaved: {balanced_path}")

    # Save train/test splits
    train_path = os.path.join(OUTPUT_DIR, 'train_dataset.csv')
    test_path = os.path.join(OUTPUT_DIR, 'test_dataset.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")

    # Save feature list for reference
    features_path = os.path.join(OUTPUT_DIR, 'feature_columns.txt')
    with open(features_path, 'w') as f:
        f.write("# Feature columns used for training\n")
        f.write(f"# Total: {len(feature_cols)} features\n\n")
        for feat in feature_cols:
            f.write(f"{feat}\n")
    print(f"Saved: {features_path}")

    return balanced_path, train_path, test_path


def main(use_log: bool = True):
    """Main execution function."""

    print("=" * 60)
    print("PREPARE BALANCED DATASET FOR TRAINING")
    print("=" * 60)
    print(f"\nFeature scaling: {'LOG' if use_log else 'RAW'}")

    # Load data
    df = load_feature_matrix(use_log=use_log)

    # Analyze current distribution
    stats = analyze_label_distribution(df)

    # Remove unknown labels
    df_labeled = df[df['label'] != -1].copy()
    print(f"\nAfter removing unknown: {len(df_labeled):,} addresses")

    # Drop cross-source label conflicts; collapse consistent duplicates.
    df_labeled = deduplicate_cross_source(df_labeled)

    # Create balanced dataset
    df_balanced = create_balanced_dataset(df_labeled, stats)

    # Select features
    df_balanced, feature_cols = select_features(df_balanced, use_log=use_log)

    # Create train/test split
    train_df, test_df = create_train_test_split(df_balanced)

    # Save datasets
    save_datasets(df_balanced, train_df, test_df, feature_cols)

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"""
Dataset Statistics:
  Total addresses:    {len(df_balanced):,}
  Training set:       {len(train_df):,} (80%)
  Test set:           {len(test_df):,} (20%)

Label Distribution (balanced):
  Benign/Licit (0):   {(df_balanced['label'] == 0).sum():,} (50%)
  Criminal/Illicit:   {(df_balanced['label'] == 1).sum():,} (50%)

Source Distribution:
  REAL-CATS:          {(df_balanced['source'] == 'realcats').sum():,}
  Elliptic++:         {(df_balanced['source'] == 'elliptic').sum():,}

Features: {len(feature_cols)} ({'log-scaled' if use_log else 'raw'})

Output Files:
  - balanced_training_dataset.csv  (full balanced dataset)
  - train_dataset.csv              (80% for training)
  - test_dataset.csv               (20% for testing)
  - feature_columns.txt            (list of feature names)
    """)

    return df_balanced, train_df, test_df, feature_cols


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare balanced training dataset')
    parser.add_argument('--raw', action='store_true',
                        help='Use raw features instead of log-scaled')

    args = parser.parse_args()

    df_balanced, train_df, test_df, features = main(use_log=not args.raw)
