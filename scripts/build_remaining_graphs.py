#!/usr/bin/env python3
"""
Build Remaining Graphs
======================
Builds ego-graphs for addresses in train/test datasets that don't have graphs yet.
Saves graphs to the appropriate directory (train/ or test/) based on which dataset
the address belongs to.

Usage:
    python build_remaining_graphs.py [--split SPLIT] [--max N] [--batch-size N]

Examples:
    # Build all remaining graphs (both train and test)
    python build_remaining_graphs.py

    # Build only train graphs
    python build_remaining_graphs.py --split train

    # Build only test graphs
    python build_remaining_graphs.py --split test

    # Build only 100 graphs (for testing)
    python build_remaining_graphs.py --max 100
"""

import asyncio
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import aiohttp

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.graph_builder import EgoGraphBuilder
from src.graph.api_fetcher import FetchResult
from src.graph.config import FEATURE_COLUMNS, BATCH_SIZE as DEFAULT_BATCH_SIZE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RemainingGraphBuilder:
    """Builds graphs for addresses missing from existing graph data."""

    def __init__(self, split: str = 'both', batch_size: int = None):
        """
        Initialize the builder.

        Args:
            split: Which split to process ('train', 'test', or 'both')
            batch_size: Number of addresses to process per batch
        """
        self.project_root = PROJECT_ROOT
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE
        self.split = split

        # Paths
        self.train_dataset_path = self.project_root / 'src' / 'features' / 'output' / 'train_dataset.csv'
        self.test_dataset_path = self.project_root / 'src' / 'features' / 'output' / 'test_dataset.csv'

        # Output directories
        self.train_graphs_dir = self.project_root / 'graph_data' / 'train'
        self.test_graphs_dir = self.project_root / 'graph_data' / 'test'

        # Cache directory (shared)
        self.cache_dir = self.project_root / 'graph_data' / 'cache' / 'transactions'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Progress file
        self.progress_dir = self.project_root / 'graph_data' / 'metadata'
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.progress_dir / f'progress_{split}.json'

        # Create output directories
        self.train_graphs_dir.mkdir(parents=True, exist_ok=True)
        self.test_graphs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.graph_builder = EgoGraphBuilder()

    def load_datasets(self) -> tuple:
        """Load train and/or test datasets based on split setting."""
        train_df = None
        test_df = None

        if self.split in ['train', 'both']:
            if not self.train_dataset_path.exists():
                raise FileNotFoundError(f"Train dataset not found: {self.train_dataset_path}")
            train_df = pd.read_csv(self.train_dataset_path)
            train_df['_split'] = 'train'
            logger.info(f"Loaded train dataset: {len(train_df):,} addresses")

        if self.split in ['test', 'both']:
            if not self.test_dataset_path.exists():
                raise FileNotFoundError(f"Test dataset not found: {self.test_dataset_path}")
            test_df = pd.read_csv(self.test_dataset_path)
            test_df['_split'] = 'test'
            logger.info(f"Loaded test dataset: {len(test_df):,} addresses")

        # Combine if both
        if self.split == 'both':
            df = pd.concat([train_df, test_df], ignore_index=True)
        elif self.split == 'train':
            df = train_df
        else:
            df = test_df

        logger.info(f"Total addresses to consider: {len(df):,}")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        return df

    def get_existing_addresses(self) -> dict:
        """Get dict of addresses that already have graphs, mapped to their split."""
        existing = {}

        # Check train directory
        if self.train_graphs_dir.exists():
            for f in self.train_graphs_dir.glob('*.pt'):
                existing[f.stem] = 'train'

        # Check test directory
        if self.test_graphs_dir.exists():
            for f in self.test_graphs_dir.glob('*.pt'):
                existing[f.stem] = 'test'

        logger.info(f"Found {len(existing):,} existing graphs (train: {sum(1 for v in existing.values() if v == 'train')}, test: {sum(1 for v in existing.values() if v == 'test')})")
        return existing

    def get_missing_addresses(self, df: pd.DataFrame, existing: dict) -> pd.DataFrame:
        """Get dataframe of addresses missing graphs."""
        mask = ~df['address'].isin(existing.keys())
        missing_df = df[mask].copy()

        train_missing = len(missing_df[missing_df['_split'] == 'train']) if 'train' in missing_df['_split'].values else 0
        test_missing = len(missing_df[missing_df['_split'] == 'test']) if 'test' in missing_df['_split'].values else 0

        logger.info(f"Missing graphs: {len(missing_df):,} addresses (train: {train_missing}, test: {test_missing})")
        return missing_df

    def load_progress(self) -> dict:
        """Load progress from previous run."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'started_at': None,
            'completed': 0,
            'failed': 0,
            'failed_addresses': []
        }

    def save_progress(self, progress: dict):
        """Save progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def save_graph(self, address: str, split: str, data):
        """Save graph to appropriate directory based on split."""
        if split == 'train':
            path = self.train_graphs_dir / f"{address}.pt"
        else:
            path = self.test_graphs_dir / f"{address}.pt"
        torch.save(data, path)

    async def fetch_transactions(self, address: str, session) -> FetchResult:
        """Fetch transactions for a single address."""
        # Check cache first
        cache_file = self.cache_dir / f"{address}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                transactions = json.load(f)
            return FetchResult(
                address=address,
                success=True,
                transactions=transactions,
                from_cache=True
            )

        # Fetch from API
        url = f"https://mempool.emzy.de/api/address/{address}/txs"

        for attempt in range(5):
            try:
                await asyncio.sleep(0.5)  # Rate limit: 2 req/sec

                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        transactions = await response.json()
                        # Cache result
                        with open(cache_file, 'w') as f:
                            json.dump(transactions, f)
                        return FetchResult(
                            address=address,
                            success=True,
                            transactions=transactions
                        )
                    elif response.status == 429:
                        wait_time = 5 * (attempt + 1)
                        logger.warning(f"Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    elif response.status == 404:
                        # Cache empty result
                        with open(cache_file, 'w') as f:
                            json.dump([], f)
                        return FetchResult(
                            address=address,
                            success=True,
                            transactions=[]
                        )
                    else:
                        return FetchResult(
                            address=address,
                            success=False,
                            error=f"HTTP {response.status}"
                        )
            except asyncio.TimeoutError:
                if attempt < 4:
                    await asyncio.sleep(5)
                else:
                    return FetchResult(
                        address=address,
                        success=False,
                        error="Timeout"
                    )
            except Exception as e:
                return FetchResult(
                    address=address,
                    success=False,
                    error=str(e)
                )

        return FetchResult(
            address=address,
            success=False,
            error="Max retries exceeded"
        )

    def get_features_for_address(self, row: pd.Series) -> np.ndarray:
        """Get feature vector for an address from the dataset row."""
        features = []
        for col in FEATURE_COLUMNS:
            if col in row:
                features.append(row[col])
            else:
                features.append(0.0)

        return np.array(features, dtype=np.float32)

    async def process_batch(
        self,
        batch_df: pd.DataFrame,
        progress: dict,
        pbar: tqdm
    ) -> tuple:
        """Process a batch of addresses."""
        successful = 0
        failed = 0

        connector = aiohttp.TCPConnector(limit=1)
        headers = {'User-Agent': 'BitcoinWalletAnalyzer/1.0'}

        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            for _, row in batch_df.iterrows():
                address = row['address']
                label = int(row['label'])
                split = row['_split']

                try:
                    # Fetch transactions
                    result = await self.fetch_transactions(address, session)

                    if result.success:
                        # Get features from dataset row
                        features = self.get_features_for_address(row)

                        # Build graph
                        if result.transactions:
                            data = self.graph_builder.build_ego_graph(
                                center_address=address,
                                center_features=features,
                                center_label=label,
                                transactions=result.transactions
                            )
                        else:
                            data = self.graph_builder.build_empty_graph(
                                center_address=address,
                                center_features=features,
                                center_label=label
                            )

                        # Save graph to appropriate directory
                        self.save_graph(address, split, data)
                        successful += 1
                        progress['completed'] += 1
                    else:
                        failed += 1
                        progress['failed'] += 1
                        progress['failed_addresses'].append({
                            'address': address,
                            'split': split,
                            'error': result.error
                        })
                        logger.warning(f"Failed to fetch {address}: {result.error}")

                except Exception as e:
                    failed += 1
                    progress['failed'] += 1
                    progress['failed_addresses'].append({
                        'address': address,
                        'split': split,
                        'error': str(e)
                    })
                    logger.warning(f"Error processing {address}: {e}")

                pbar.update(1)

        return successful, failed

    async def run(self, max_addresses: int = None):
        """Run the graph building pipeline."""
        logger.info("=" * 60)
        logger.info(f"Building Remaining Graphs (split: {self.split})")
        logger.info("=" * 60)

        # Load data
        df = self.load_datasets()
        existing = self.get_existing_addresses()
        missing_df = self.get_missing_addresses(df, existing)

        if missing_df.empty:
            logger.info("All addresses already have graphs!")
            return

        # Limit if specified
        if max_addresses:
            missing_df = missing_df.head(max_addresses)
            logger.info(f"Limited to {max_addresses} addresses")

        # Load progress
        progress = self.load_progress()
        if progress['started_at'] is None:
            progress['started_at'] = datetime.now().isoformat()

        # Process addresses
        total_successful = 0
        total_failed = 0

        pbar = tqdm(total=len(missing_df), desc="Building graphs")

        for i in range(0, len(missing_df), self.batch_size):
            batch_df = missing_df.iloc[i:i + self.batch_size]
            successful, failed = await self.process_batch(batch_df, progress, pbar)

            total_successful += successful
            total_failed += failed

            # Save checkpoint every 500 addresses
            if (i + self.batch_size) % 500 == 0:
                self.save_progress(progress)
                logger.info(f"Checkpoint: {total_successful:,} successful, {total_failed:,} failed")

        pbar.close()

        # Final save
        self.save_progress(progress)

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Pipeline Complete")
        logger.info("=" * 60)
        logger.info(f"Split: {self.split}")
        logger.info(f"Total processed: {total_successful + total_failed:,}")
        logger.info(f"Successful: {total_successful:,}")
        logger.info(f"Failed: {total_failed:,}")
        logger.info("")

        # Count final graphs
        train_count = len(list(self.train_graphs_dir.glob('*.pt')))
        test_count = len(list(self.test_graphs_dir.glob('*.pt')))
        logger.info(f"Graphs in train/: {train_count:,}")
        logger.info(f"Graphs in test/: {test_count:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Build graphs for addresses missing from existing graph data'
    )
    parser.add_argument(
        '--split',
        choices=['train', 'test', 'both'],
        default='both',
        help='Which split to process (default: both)'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=None,
        help='Maximum number of addresses to process'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing (default: 100)'
    )

    args = parser.parse_args()

    builder = RemainingGraphBuilder(
        split=args.split,
        batch_size=args.batch_size
    )

    asyncio.run(builder.run(max_addresses=args.max))


if __name__ == "__main__":
    main()
