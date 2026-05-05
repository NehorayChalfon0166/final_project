"""Master pipeline CLI.

Thin dispatcher that wires command-line flags to the step functions in
``src/pipeline_steps/``. Each step is a separate module:

    --prepare   src/pipeline_steps/prepare.py    (feature engineering + balanced split)
    --graphs    src/pipeline_steps/graphs.py     (build ego-graphs from mempool.space)
    --baseline  src/pipeline_steps/baseline.py   (XGBoost tabular baseline)
    --train     src/pipeline_steps/train.py      (GNN training + temperature calibration)
    --evaluate  src/pipeline_steps/evaluate.py   (GNN vs XGBoost comparison + error CSVs)

Usage:
    python run_pipeline.py --all
    python run_pipeline.py --prepare
    python run_pipeline.py --graphs --split test --max-graphs 100
    python run_pipeline.py --train --epochs 150 --batch-size 64
    python run_pipeline.py --graphs --retry-failed
"""
import argparse
import logging
import os
from datetime import datetime

from src.pipeline_steps import (
    step_build_graphs,
    step_evaluate,
    step_prepare_data,
    step_train_baseline,
    step_train_gnn,
)
from src.pipeline_steps._paths import OUTPUTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full pipeline")

    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--prepare", action="store_true", help="Step 1: Prepare data")
    parser.add_argument("--graphs", action="store_true", help="Step 2: Build graphs")
    parser.add_argument("--baseline", action="store_true", help="Step 3: Train XGBoost")
    parser.add_argument("--train", action="store_true", help="Step 4: Train GNN")
    parser.add_argument("--evaluate", action="store_true", help="Step 5: Evaluate & compare")

    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry building previously-failed graphs")
    parser.add_argument("--compare", action="store_true",
                        help="(Deprecated alias for --evaluate)")

    parser.add_argument("--max-graphs", type=int, default=None,
                        help="Max graphs to build (testing)")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                        help="Which split to build graphs for")

    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    args = parser.parse_args()
    if args.compare:
        args.evaluate = True
    if not any([args.all, args.prepare, args.graphs, args.baseline, args.train, args.evaluate]):
        args.all = True
    return args


def main() -> None:
    args = _parse_args()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    start_time = datetime.now()
    logger.info(f"Pipeline started at {start_time}")

    try:
        if args.all or args.prepare:
            step_prepare_data()
        if args.all or args.graphs:
            step_build_graphs(
                max_addresses=args.max_graphs,
                split=args.split,
                retry_failed=args.retry_failed,
            )
        if args.all or args.baseline:
            step_train_baseline()
        if args.all or args.train:
            step_train_gnn(epochs=args.epochs, batch_size=args.batch_size)
        if args.all or args.evaluate:
            step_evaluate()
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

    duration = datetime.now() - start_time
    logger.info(f"\nPipeline completed in {duration}")


if __name__ == "__main__":
    main()
