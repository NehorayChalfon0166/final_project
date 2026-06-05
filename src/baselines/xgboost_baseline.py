"""
XGBoost Baseline
================
Tabular baseline trained and evaluated on exactly the same wallets the GNN sees:
    train on the rows of src/features/output/train_dataset.csv whose address
           has a matching .pt under graph_data/graphs/train/
    test  on the rows of src/features/output/test_dataset.csv whose address
           has a matching .pt under graph_data/graphs/test/

Single fit, single eval — no cross-validation. This matches the GNN's protocol
row-for-row so the comparison printed by ``run_pipeline.py --evaluate`` is
apples-to-apples.

Usage:
    from src.baselines import XGBoostBaseline

    baseline = XGBoostBaseline()
    baseline.load_data()
    baseline.train()
    baseline.save_results()
"""
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importing `src.graph.config` would trigger `src/graph/__init__.py`, which
# eagerly loads `dataloader` and therefore PyTorch. PyTorch + sklearn +
# system libomp end up loading three OpenMP runtimes into one process on
# macOS, which deadlocks at import time. The baseline doesn't need torch at
# all, so we load `config.py` as a standalone module file to skip the
# package init.
import importlib.util as _importlib_util

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "graph", "config.py",
)
_spec = _importlib_util.spec_from_file_location("_graph_config_standalone", _CONFIG_PATH)
_config = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_config)
FEATURE_COLUMNS = _config.FEATURE_COLUMNS
FEATURE_DIR = _config.FEATURE_DIR
GRAPHS_DIR = _config.GRAPHS_DIR


@dataclass
class Results:
    """Held-out test-set metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list
    n_train_samples: int
    n_test_samples: int
    n_features: int
    training_time: float


XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    eval_metric="logloss",
    verbosity=0,
)


class XGBoostBaseline:
    """XGBoost classifier on the same train/test split the GNN uses."""

    def __init__(
        self,
        random_state: int = 42,
        output_dir: str = "outputs/baseline",
    ):
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.scaler = StandardScaler()
        self.model: Optional[XGBClassifier] = None
        self.results: Optional[Results] = None

    def load_data(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        features: Optional[list] = None,
        restrict_to_cached_graphs: bool = True,
    ) -> None:
        """Load the same train/test CSVs the GNN dataloader uses.

        When ``restrict_to_cached_graphs`` is True (the default), rows are
        filtered to addresses that have a matching ``.pt`` file under
        ``graph_data/graphs/{train,test}/`` — i.e. the exact wallets the GNN
        trains and evaluates on. The XGBoost protocol then matches the GNN
        protocol row-for-row.
        """
        train_path = train_path or os.path.join(FEATURE_DIR, "train_dataset.csv")
        test_path = test_path or os.path.join(FEATURE_DIR, "test_dataset.csv")
        features = features or FEATURE_COLUMNS

        print(f"Loading train from {train_path}")
        train_df = pd.read_csv(train_path)
        print(f"Loading test  from {test_path}")
        test_df = pd.read_csv(test_path)

        if restrict_to_cached_graphs:
            train_df = self._filter_to_cached_graphs(train_df, split="train")
            test_df = self._filter_to_cached_graphs(test_df, split="test")

        # Fit the scaler on train only — test sees the same transform.
        self.X_train = self.scaler.fit_transform(train_df[features].values)
        self.y_train = train_df["label"].values
        self.X_test = self.scaler.transform(test_df[features].values)
        self.y_test = test_df["label"].values

        print(
            f"Train: {len(self.y_train):,} samples ({np.mean(self.y_train):.1%} criminal)"
        )
        print(
            f"Test:  {len(self.y_test):,} samples ({np.mean(self.y_test):.1%} criminal)"
        )
        print(f"Features: {len(features)}")

    @staticmethod
    def _filter_to_cached_graphs(df: pd.DataFrame, split: str) -> pd.DataFrame:
        graph_dir = os.path.join(GRAPHS_DIR, split)
        cached = {f[:-3] for f in os.listdir(graph_dir) if f.endswith(".pt")}
        before = len(df)
        df = df[df["address"].astype(str).isin(cached)].reset_index(drop=True)
        print(
            f"  Filtered {split}: {before:,} -> {len(df):,} rows "
            f"(kept only addresses with a cached .pt graph)"
        )
        return df

    def train(self) -> Results:
        """Fit on train, evaluate on test."""
        if self.X_train is None:
            self.load_data()

        print("\nTraining XGBoost (single fit, no CV)...")
        start = time.time()
        self.model = XGBClassifier(random_state=self.random_state, **XGB_PARAMS)
        self.model.fit(self.X_train, self.y_train)
        training_time = time.time() - start

        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        self.results = Results(
            accuracy=accuracy_score(self.y_test, y_pred),
            precision=precision_score(self.y_test, y_pred),
            recall=recall_score(self.y_test, y_pred),
            f1=f1_score(self.y_test, y_pred),
            roc_auc=roc_auc_score(self.y_test, y_prob),
            confusion_matrix=confusion_matrix(self.y_test, y_pred).tolist(),
            n_train_samples=len(self.y_train),
            n_test_samples=len(self.y_test),
            n_features=self.X_test.shape[1],
            training_time=training_time,
        )
        self._print_results()
        return self.results

    def _print_results(self) -> None:
        r = self.results
        print(f"\n{'='*50}")
        print("XGBoost Results (held-out test set)")
        print(f"{'='*50}")
        print(f"Accuracy:  {r.accuracy*100:.2f}%")
        print(f"Precision: {r.precision*100:.2f}%")
        print(f"Recall:    {r.recall*100:.2f}%")
        print(f"F1 Score:  {r.f1*100:.2f}%")
        print(f"ROC-AUC:   {r.roc_auc:.4f}")
        print(f"{'='*50}")

    def save_results(self, filename: str = "xgboost_results.json") -> Path:
        if self.results is None:
            raise ValueError("No results. Run train() first.")
        output_path = self.output_dir / filename
        payload = {
            "model": "XGBoost",
            "protocol": "single fit on train rows that have a cached .pt graph, evaluation on test rows that have a cached .pt graph (matches GNN row-for-row)",
            "results": asdict(self.results),
            "features": FEATURE_COLUMNS,
            "timestamp": datetime.now().isoformat(),
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {output_path}")
        return output_path

    def save_model(self, filename: str = "xgboost_model.json") -> Path:
        if self.model is None:
            raise ValueError("No model. Run train() first.")
        output_path = self.output_dir / filename
        self.model.save_model(str(output_path))
        print(f"Model saved to {output_path}")
        return output_path

    def load_model(self, filename: str = "xgboost_model.json"):
        model_path = self.output_dir / filename
        self.model = XGBClassifier()
        self.model.load_model(str(model_path))
        print(f"Model loaded from {model_path}")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model. Run train() or load_model() first.")
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model. Run train() or load_model() first.")
        return self.model.predict_proba(self.scaler.transform(X))


def train_xgboost_baseline() -> Results:
    """Convenience: load data, train, save results + model."""
    baseline = XGBoostBaseline()
    baseline.load_data()
    results = baseline.train()
    baseline.save_results()
    baseline.save_model()
    return results


if __name__ == "__main__":
    train_xgboost_baseline()
