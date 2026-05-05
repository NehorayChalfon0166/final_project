"""Step 3 — train the XGBoost tabular baseline."""
import logging
import os

from ._paths import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def step_train_baseline():
    logger.info("=" * 60)
    logger.info("STEP 3: Training XGBoost Baseline")
    logger.info("=" * 60)

    from src.baselines.xgboost_baseline import XGBoostBaseline

    baseline = XGBoostBaseline(output_dir=os.path.join(OUTPUTS_DIR, "baseline"))
    baseline.load_data()
    results = baseline.train()
    baseline.save_results()
    baseline.save_model()
    return results
