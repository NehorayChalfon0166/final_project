"""Step 1 — feature engineering and balanced train/test split.

Reads the raw REAL-CATS + Elliptic++ datasets, produces the unified feature
matrix, and writes ``train_dataset.csv`` / ``test_dataset.csv`` to
``src/features/output/`` for use by the graph and baseline steps.
"""
import logging
import os

from ._paths import FEATURE_OUTPUT_DIR

logger = logging.getLogger(__name__)


def step_prepare_data() -> bool:
    logger.info("=" * 60)
    logger.info("STEP 1: Preparing Unified Dataset")
    logger.info("=" * 60)

    feature_matrix = os.path.join(
        FEATURE_OUTPUT_DIR, "conservative_feature_matrix_with_logs.csv"
    )
    if not os.path.exists(feature_matrix):
        logger.info("Running feature engineering pipeline...")
        from src.features.pipeline_conservative import main as run_feature_pipeline
        run_feature_pipeline()
    else:
        logger.info(f"Feature matrix exists: {feature_matrix}")

    logger.info("\nCreating balanced dataset...")
    from src.features.prepare_balanced_dataset import main as prepare_balanced
    prepare_balanced(use_log=True)

    balanced_path = os.path.join(FEATURE_OUTPUT_DIR, "balanced_training_dataset.csv")
    if os.path.exists(balanced_path):
        import pandas as pd
        df = pd.read_csv(balanced_path)
        logger.info(f"\nDataset ready: {len(df):,} addresses")
        logger.info(f"  Benign:   {(df['label'] == 0).sum():,}")
        logger.info(f"  Criminal: {(df['label'] == 1).sum():,}")
        return True
    return False
