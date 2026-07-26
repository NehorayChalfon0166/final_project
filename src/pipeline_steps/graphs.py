"""Step 2 — build ego-graphs from mempool.space for each split."""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def step_build_graphs(
    max_addresses: Optional[int] = None,
    split: str = "both",
    retry_failed: bool = False,
) -> None:
    logger.info("=" * 60)
    logger.info("STEP 2: Building Ego-Graphs")
    logger.info("=" * 60)

    from src.graph.pipeline import GraphConstructionPipeline

    splits = ["train", "test"] if split == "both" else [split]

    for s in splits:
        logger.info(f"\nProcessing {s} split...")
        pipeline = GraphConstructionPipeline(s)

        if retry_failed:
            logger.info("Retrying failed addresses only...")

        pending = pipeline.get_pending_addresses()
        logger.info(f"  Pending addresses: {len(pending):,}")

        if pending or retry_failed:
            asyncio.run(pipeline.run(
                max_addresses=max_addresses,
                resume=True,
                retry_failed=retry_failed,
            ))
