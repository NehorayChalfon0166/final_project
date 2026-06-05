"""Download ego-graphs for pending REAL-CATS wallets, filtered by label.

Subclasses the standard graph pipeline and filters get_pending_addresses to
(source='realcats', label=L). Originally written to drain benign realcats only
(--label 0), now generalized so the same machinery can also fetch criminal
realcats (--label 1) — used for shaping the test split toward a 70/30 mix.

Usage:
    python scripts/download_benign_realcats.py                              # train, benign, all pending
    python scripts/download_benign_realcats.py --split test --label 0 --max 3730
    python scripts/download_benign_realcats.py --split test --label 1 --max 1642
"""
import argparse
import asyncio
import logging
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.pipeline import GraphConstructionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FilteredRealCatsPipeline(GraphConstructionPipeline):
    def __init__(self, split: str, label: int):
        super().__init__(split=split)
        self.label = label

    def get_pending_addresses(self):
        pending = set(super().get_pending_addresses())
        df = self.df
        eligible = set(df[(df["label"] == self.label) & (df["source"] == "realcats")]["address"])
        filtered = [a for a in pending if a in eligible]
        label_name = "benign" if self.label == 0 else "criminal"
        logger.info(
            f"Filtered to realcats {label_name}: {len(filtered):,} / {len(pending):,} pending"
        )
        return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--label", type=int, choices=[0, 1], default=0,
                        help="0=benign, 1=criminal")
    parser.add_argument("--max", type=int, default=None,
                        help="Max addresses to download this run (None = all pending)")
    args = parser.parse_args()

    pipeline = FilteredRealCatsPipeline(split=args.split, label=args.label)
    asyncio.run(pipeline.run(resume=True, max_addresses=args.max))
