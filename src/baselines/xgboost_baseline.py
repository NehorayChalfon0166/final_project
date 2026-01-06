"""
XGBoost Baseline
================
Train XGBoost classifier for comparison with GNN.

Usage:
    from src.baselines import XGBoostBaseline

    baseline = XGBoostBaseline()
    baseline.load_data()
    results = baseline.train()
    baseline.save_results()
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.graph.config import FEATURE_COLUMNS, FEATURE_DIR


@dataclass
class Results:
    """Results container."""
    accuracy: float
    accuracy_std: float
    precision: float
    precision_std: float
    recall: float
    recall_std: float
    f1: float
    f1_std: float
    roc_auc: float
    roc_auc_std: float
    confusion_matrix: list
    n_samples: int
    n_features: int
    training_time: float


class XGBoostBaseline:
    """
    XGBoost baseline model.

    Simple interface:
        baseline = XGBoostBaseline()
        baseline.load_data()
        results = baseline.train()
        baseline.save_results()
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        output_dir: str = "baselines/results"
    ):
        self.n_folds = n_folds
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.results: Optional[Results] = None
        self.model = None

    def load_data(
        self,
        data_path: Optional[str] = None,
        features: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for training.

        Args:
            data_path: Path to CSV (default: balanced_training_dataset.csv)
            features: Feature columns (default: 12 selected features)

        Returns:
            X, y arrays
        """
        data_path = data_path or os.path.join(FEATURE_DIR, 'balanced_training_dataset.csv')
        features = features or FEATURE_COLUMNS

        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        self.X = df[features].values
        self.y = df['label'].values

        # Standardize
        self.X = self.scaler.fit_transform(self.X)

        print(f"Loaded {len(self.y):,} samples, {len(features)} features")
        print(f"Class balance: {np.mean(self.y):.1%} positive")

        return self.X, self.y

    def train(self) -> Results:
        """
        Train with 5-fold cross-validation.

        Returns:
            Results dataclass with all metrics
        """
        import time

        if self.X is None:
            self.load_data()

        print(f"\nTraining XGBoost ({self.n_folds}-fold CV)...")
        start_time = time.time()

        cv = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        # Collect metrics per fold
        metrics = {k: [] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
        all_preds, all_probs, all_labels = [], [], []

        for fold, (train_idx, val_idx) in enumerate(cv.split(self.X, self.y)):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # Train
            model = XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0
            )
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            # Store
            all_preds.extend(y_pred)
            all_probs.extend(y_prob)
            all_labels.extend(y_val)

            # Metrics
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred))
            metrics['recall'].append(recall_score(y_val, y_pred))
            metrics['f1'].append(f1_score(y_val, y_pred))
            metrics['roc_auc'].append(roc_auc_score(y_val, y_prob))

            print(f"  Fold {fold+1}: Acc={metrics['accuracy'][-1]:.4f}, F1={metrics['f1'][-1]:.4f}")

        training_time = time.time() - start_time

        # Final model on all data
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0
        )
        self.model.fit(self.X, self.y)

        # Results
        self.results = Results(
            accuracy=np.mean(metrics['accuracy']),
            accuracy_std=np.std(metrics['accuracy']),
            precision=np.mean(metrics['precision']),
            precision_std=np.std(metrics['precision']),
            recall=np.mean(metrics['recall']),
            recall_std=np.std(metrics['recall']),
            f1=np.mean(metrics['f1']),
            f1_std=np.std(metrics['f1']),
            roc_auc=np.mean(metrics['roc_auc']),
            roc_auc_std=np.std(metrics['roc_auc']),
            confusion_matrix=confusion_matrix(all_labels, all_preds).tolist(),
            n_samples=len(self.y),
            n_features=self.X.shape[1],
            training_time=training_time
        )

        self._print_results()
        return self.results

    def _print_results(self):
        """Print formatted results."""
        r = self.results
        print(f"\n{'='*50}")
        print("XGBoost Results")
        print(f"{'='*50}")
        print(f"Accuracy:  {r.accuracy*100:.2f}% ± {r.accuracy_std*100:.2f}%")
        print(f"Precision: {r.precision*100:.2f}% ± {r.precision_std*100:.2f}%")
        print(f"Recall:    {r.recall*100:.2f}% ± {r.recall_std*100:.2f}%")
        print(f"F1 Score:  {r.f1*100:.2f}% ± {r.f1_std*100:.2f}%")
        print(f"ROC-AUC:   {r.roc_auc:.4f} ± {r.roc_auc_std:.4f}")
        print(f"{'='*50}")

    def save_results(self, filename: str = "xgboost_results.json") -> Path:
        """Save results to JSON."""
        if self.results is None:
            raise ValueError("No results. Run train() first.")

        output_path = self.output_dir / filename

        data = {
            'model': 'XGBoost',
            'results': asdict(self.results),
            'features': FEATURE_COLUMNS,
            'timestamp': datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {output_path}")
        return output_path

    def save_model(self, filename: str = "xgboost_model.json") -> Path:
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model. Run train() first.")

        output_path = self.output_dir / filename
        self.model.save_model(str(output_path))
        print(f"Model saved to {output_path}")
        return output_path

    def load_model(self, filename: str = "xgboost_model.json"):
        """Load a saved model."""
        model_path = self.output_dir / filename
        self.model = XGBClassifier()
        self.model.load_model(str(model_path))
        print(f"Model loaded from {model_path}")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data."""
        if self.model is None:
            raise ValueError("No model. Run train() or load_model() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities on new data."""
        if self.model is None:
            raise ValueError("No model. Run train() or load_model() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


# Convenience function
def train_xgboost_baseline() -> Results:
    """Quick function to train XGBoost baseline."""
    baseline = XGBoostBaseline()
    baseline.load_data()
    results = baseline.train()
    baseline.save_results()
    baseline.save_model()
    return results


if __name__ == "__main__":
    train_xgboost_baseline()
