"""
Build graphs from existing cached JSON files.
Consolidates legacy cache directories and builds .pt graphs for any
training-set address that has cached transactions but no graph yet.
Then optionally continues the main pipeline for uncached addresses.
"""
import os
import sys
import json
import shutil
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.graph.graph_builder import EgoGraphBuilder
from src.graph.config import FEATURE_COLUMNS, TRAIN_DATASET_PATH

CACHE_DIR = os.path.join(PROJECT_ROOT, 'graph_data', 'cache')
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'graph_data', 'graphs', 'train')
METADATA_DIR = os.path.join(PROJECT_ROOT, 'graph_data', 'metadata')
PROGRESS_FILE = os.path.join(METADATA_DIR, 'progress_train.json')


def consolidate_caches(train_addrs: set) -> int:
    """Copy cached JSONs from legacy dirs into cache/train for training-set addresses."""
    legacy_dirs = [
        os.path.join(CACHE_DIR, 'transactions'),
        os.path.join(CACHE_DIR, 'eval'),
    ]
    train_cache = os.path.join(CACHE_DIR, 'train')
    copied = 0

    for d in legacy_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith('.json'):
                continue
            addr = fname.replace('.json', '')
            dest = os.path.join(train_cache, fname)
            if addr in train_addrs and not os.path.exists(dest):
                shutil.copy2(os.path.join(d, fname), dest)
                copied += 1

    return copied


def find_addresses_needing_graphs(train_addrs: set) -> list:
    """Find addresses with cached JSON in cache/train but no .pt graph."""
    train_cache = os.path.join(CACHE_DIR, 'train')
    cached = set(f.replace('.json', '') for f in os.listdir(train_cache) if f.endswith('.json'))
    existing_graphs = set(f.replace('.pt', '') for f in os.listdir(GRAPHS_DIR) if f.endswith('.pt'))

    need_build = (cached & train_addrs) - existing_graphs
    return list(need_build)


def build_graphs(addresses: list, df: pd.DataFrame) -> tuple:
    """Build .pt graph files from cached JSON transactions."""
    builder = EgoGraphBuilder()
    train_cache = os.path.join(CACHE_DIR, 'train')

    successful = 0
    failed = 0
    errors = []

    for addr in tqdm(addresses, desc="Building graphs from cache"):
        try:
            # Load cached transactions
            cache_path = os.path.join(train_cache, f"{addr}.json")
            with open(cache_path, 'r') as f:
                transactions = json.load(f)

            # Get features and label from dataset
            row = df[df['address'] == addr].iloc[0]
            features = row[FEATURE_COLUMNS].values.astype(np.float32)
            label = int(row['label'])

            # Build graph
            if transactions:
                data = builder.build_ego_graph(addr, features, label, transactions)
            else:
                data = builder.build_empty_graph(addr, features, label)

            # Save
            graph_path = os.path.join(GRAPHS_DIR, f"{addr}.pt")
            torch.save(data, graph_path)
            successful += 1

        except Exception as e:
            failed += 1
            errors.append((addr, str(e)))

    return successful, failed, errors


def update_progress(new_addresses: list):
    """Add newly built addresses to progress tracking."""
    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)

    completed_set = set(progress.get('completed', []))
    added = 0
    for addr in new_addresses:
        if addr not in completed_set:
            progress['completed'].append(addr)
            completed_set.add(addr)
            added += 1

    # Also sync: add any .pt files that exist but aren't tracked
    existing_graphs = set(f.replace('.pt', '') for f in os.listdir(GRAPHS_DIR) if f.endswith('.pt'))
    untracked = existing_graphs - completed_set
    for addr in untracked:
        progress['completed'].append(addr)
        added += 1

    from datetime import datetime
    progress['last_updated'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

    return added


def main():
    print("=" * 60)
    print("Build Graphs from Cached Transaction Data")
    print("=" * 60)

    # Load training dataset
    df = pd.read_csv(TRAIN_DATASET_PATH)
    train_addrs = set(df['address'].tolist())
    print(f"\nTraining dataset: {len(train_addrs):,} addresses")

    # Step 1: Consolidate legacy caches
    print("\n[1/4] Consolidating legacy cache directories...")
    copied = consolidate_caches(train_addrs)
    print(f"  Copied {copied} files from legacy caches to cache/train")

    # Step 2: Find addresses needing graphs
    print("\n[2/4] Finding addresses with cache but no graph...")
    need_build = find_addresses_needing_graphs(train_addrs)
    print(f"  Found {len(need_build)} addresses needing graph construction")

    if not need_build:
        print("\n  No new graphs to build from cache!")
    else:
        # Step 3: Build graphs
        print(f"\n[3/4] Building {len(need_build)} graphs...")
        successful, failed, errors = build_graphs(need_build, df)
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        if errors:
            print(f"  First 5 errors:")
            for addr, err in errors[:5]:
                print(f"    {addr}: {err}")

    # Step 4: Sync progress file
    print("\n[4/4] Syncing progress tracking...")
    # All addresses that now have .pt files
    all_graphed = set(f.replace('.pt', '') for f in os.listdir(GRAPHS_DIR) if f.endswith('.pt'))
    newly_tracked = update_progress(list(all_graphed))
    print(f"  Added {newly_tracked} addresses to progress tracking")

    # Summary
    final_graphs = len([f for f in os.listdir(GRAPHS_DIR) if f.endswith('.pt')])
    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)
    print(f"\n{'=' * 60}")
    print(f"Final state:")
    print(f"  Train graphs: {final_graphs:,}")
    print(f"  Progress completed: {len(progress['completed']):,}")
    print(f"  Remaining: {len(train_addrs) - len(set(progress['completed']) & train_addrs):,}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
