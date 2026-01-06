"""
Graph Construction Pipeline
===========================
Main orchestration script for building ego-graphs from wallet addresses.
"""
import asyncio
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional, List
import logging
import argparse
from datetime import datetime

from .config import (
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
    FEATURE_COLUMNS,
    BATCH_SIZE,
    CHECKPOINT_INTERVAL
)
from .api_fetcher import MempoolFetcher, FetchResult
from .graph_builder import EgoGraphBuilder
from .cache_manager import CacheManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphConstructionPipeline:
    """Main pipeline for constructing ego-graphs from wallet addresses."""

    def __init__(self, split: str = 'train'):
        """
        Initialize pipeline for a specific split.

        Args:
            split: 'train' or 'test'
        """
        self.split = split
        self.cache_manager = CacheManager(split)
        self.graph_builder = EgoGraphBuilder()
        self.fetcher = MempoolFetcher(self.cache_manager)

        # Load dataset
        dataset_path = TRAIN_DATASET_PATH if split == 'train' else TEST_DATASET_PATH
        self.df = pd.read_csv(dataset_path)
        logger.info(f"Loaded {len(self.df):,} addresses from {split} dataset")

    def get_pending_addresses(self) -> List[str]:
        """Get list of addresses that haven't been processed yet."""
        all_addresses = set(self.df['address'].tolist())
        completed = self.cache_manager.get_completed_addresses()

        # Also exclude failed addresses (can retry with --retry-failed)
        failed_list = self.cache_manager.get_failed_addresses()
        failed = {item['address'] for item in failed_list}

        pending = all_addresses - completed - failed
        return list(pending)

    def get_failed_addresses(self) -> List[str]:
        """Get list of failed addresses for retry."""
        failed_list = self.cache_manager.get_failed_addresses()
        return [item['address'] for item in failed_list]

    def get_address_data(self, address: str) -> tuple:
        """
        Get features and label for an address.

        Args:
            address: Wallet address

        Returns:
            (features, label) tuple
        """
        row = self.df[self.df['address'] == address]
        if row.empty:
            raise ValueError(f"Address not found in dataset: {address}")

        row = row.iloc[0]
        features = row[FEATURE_COLUMNS].values.astype(np.float32)
        label = int(row['label'])
        return features, label

    async def process_batch(
        self,
        addresses: List[str],
        progress: dict,
        pbar: tqdm
    ) -> tuple:
        """
        Process a batch of addresses.

        Args:
            addresses: List of addresses to process
            progress: Progress tracking dict
            pbar: Progress bar

        Returns:
            (successful_count, failed_count)
        """
        # Fetch transactions
        results = await self.fetcher.fetch_batch(addresses)

        successful = 0
        failed = 0

        for result in results:
            if result.success and result.transactions is not None:
                try:
                    # Get features and label from dataset
                    features, label = self.get_address_data(result.address)

                    # Build ego-graph
                    if result.transactions:
                        data = self.graph_builder.build_ego_graph(
                            center_address=result.address,
                            center_features=features,
                            center_label=label,
                            transactions=result.transactions
                        )
                    else:
                        # No transactions - create single-node graph
                        data = self.graph_builder.build_empty_graph(
                            center_address=result.address,
                            center_features=features,
                            center_label=label
                        )

                    # Save graph
                    self.cache_manager.save_graph(result.address, data)
                    self.cache_manager.mark_completed(result.address, progress)
                    successful += 1

                except Exception as e:
                    self.cache_manager.mark_failed(result.address, str(e), progress)
                    failed += 1
                    logger.warning(f"Error building graph for {result.address}: {e}")
            else:
                self.cache_manager.mark_failed(
                    result.address,
                    result.error or "Unknown error",
                    progress
                )
                failed += 1

            pbar.update(1)

        return successful, failed

    async def run(
        self,
        max_addresses: Optional[int] = None,
        resume: bool = True,
        retry_failed: bool = False
    ):
        """
        Run the graph construction pipeline.

        Args:
            max_addresses: Limit number of addresses to process (for testing)
            resume: Whether to resume from previous progress
            retry_failed: Whether to retry previously failed addresses
        """
        logger.info(f"Starting graph construction pipeline for {self.split} split")

        # Get addresses to process
        if retry_failed:
            pending = self.get_failed_addresses()
            # Clear failed list since we're retrying
            self.cache_manager.clear_failed()
            logger.info(f"Retrying {len(pending)} previously failed addresses")
        elif resume:
            pending = self.get_pending_addresses()
            logger.info(f"Resuming: {len(pending):,} addresses remaining")
        else:
            pending = self.df['address'].tolist()
            logger.info(f"Starting fresh: {len(pending):,} addresses to process")

        if not pending:
            logger.info("No addresses to process!")
            stats = self.cache_manager.get_cache_stats()
            logger.info(f"Cached graphs: {stats['cached_graphs']:,}")
            return

        if max_addresses:
            pending = pending[:max_addresses]
            logger.info(f"Limited to {max_addresses} addresses")

        # Load progress
        progress = self.cache_manager.load_progress()
        if progress['started_at'] is None:
            progress['started_at'] = datetime.now().isoformat()
        progress['total'] = len(self.df)

        # Process in batches
        total_successful = 0
        total_failed = 0

        pbar = tqdm(total=len(pending), desc=f"Building {self.split} graphs")

        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i:i + BATCH_SIZE]
            successful, failed = await self.process_batch(batch, progress, pbar)

            total_successful += successful
            total_failed += failed

            # Save checkpoint
            if (i + BATCH_SIZE) % CHECKPOINT_INTERVAL == 0:
                self.cache_manager.save_progress(progress)
                logger.info(f"Checkpoint: {total_successful:,} successful, {total_failed:,} failed")

        pbar.close()

        # Final save
        self.cache_manager.save_progress(progress)

        # Print summary
        stats = self.cache_manager.get_cache_stats()
        logger.info(f"""
========== Pipeline Complete ==========
Split: {self.split}
Total addresses: {len(self.df):,}
Successful: {total_successful:,}
Failed: {total_failed:,}
Cached transactions: {stats['cached_transactions']:,}
Cached graphs: {stats['cached_graphs']:,}
Cache size: {stats['cache_size_mb']:.2f} MB
Graphs size: {stats['graphs_size_mb']:.2f} MB
=======================================
        """)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Build ego-graphs for GNN training'
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
        help='Maximum addresses to process (for testing)'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh instead of resuming'
    )
    parser.add_argument(
        '--retry-failed',
        action='store_true',
        help='Retry previously failed addresses'
    )

    args = parser.parse_args()

    splits = ['train', 'test'] if args.split == 'both' else [args.split]

    for split in splits:
        pipeline = GraphConstructionPipeline(split)
        asyncio.run(pipeline.run(
            max_addresses=args.max,
            resume=not args.fresh,
            retry_failed=args.retry_failed
        ))


if __name__ == "__main__":
    main()
