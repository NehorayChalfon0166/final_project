"""
Metrics Module
==============
Comprehensive metric computation for binary classification.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


@dataclass
class MetricsReport:
    """Container for all evaluation metrics."""

    # Basic metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # AUC metrics
    roc_auc: float = 0.0
    pr_auc: float = 0.0

    # Confusion matrix
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    # Per-class metrics
    precision_per_class: Dict[int, float] = field(default_factory=dict)
    recall_per_class: Dict[int, float] = field(default_factory=dict)
    f1_per_class: Dict[int, float] = field(default_factory=dict)

    # Curves (for plotting)
    roc_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    pr_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    # Sample counts
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding curves)."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'roc_auc': self.roc_auc,
            'pr_auc': self.pr_auc,
            'tp': self.tp,
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn,
            'precision_per_class': self.precision_per_class,
            'recall_per_class': self.recall_per_class,
            'f1_per_class': self.f1_per_class,
            'n_samples': self.n_samples,
            'n_positive': self.n_positive,
            'n_negative': self.n_negative
        }

    def __str__(self) -> str:
        """Pretty print metrics."""
        return f"""
Evaluation Metrics
==================
Accuracy:   {self.accuracy:.4f} ({self.accuracy*100:.2f}%)
Precision:  {self.precision:.4f}
Recall:     {self.recall:.4f}
F1 Score:   {self.f1:.4f}
ROC-AUC:    {self.roc_auc:.4f}
PR-AUC:     {self.pr_auc:.4f}

Confusion Matrix:
                 Predicted
              Neg      Pos
Actual Neg    {self.tn:<8} {self.fp:<8} (TN, FP)
Actual Pos    {self.fn:<8} {self.tp:<8} (FN, TP)

Sample Distribution:
  Total:    {self.n_samples}
  Positive: {self.n_positive} ({self.n_positive/self.n_samples*100:.1f}%)
  Negative: {self.n_negative} ({self.n_negative/self.n_samples*100:.1f}%)
"""


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> MetricsReport:
    """
    Compute all classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probabilities for positive class (optional)

    Returns:
        MetricsReport with all metrics
    """
    report = MetricsReport()

    # Sample counts
    report.n_samples = len(y_true)
    report.n_positive = int(np.sum(y_true == 1))
    report.n_negative = int(np.sum(y_true == 0))

    # Basic metrics
    report.accuracy = accuracy_score(y_true, y_pred)
    report.precision = precision_score(y_true, y_pred, zero_division=0)
    report.recall = recall_score(y_true, y_pred, zero_division=0)
    report.f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        report.tn, report.fp, report.fn, report.tp = cm.ravel()
    else:
        # Handle edge cases (single class)
        report.tn = report.fp = report.fn = report.tp = 0

    # Per-class metrics
    for cls in [0, 1]:
        mask = y_true == cls
        if mask.sum() > 0:
            report.precision_per_class[cls] = precision_score(
                y_true, y_pred, pos_label=cls, zero_division=0
            )
            report.recall_per_class[cls] = recall_score(
                y_true, y_pred, pos_label=cls, zero_division=0
            )
            report.f1_per_class[cls] = f1_score(
                y_true, y_pred, pos_label=cls, zero_division=0
            )

    # AUC metrics (require probabilities)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            report.roc_auc = roc_auc_score(y_true, y_prob)
            report.pr_auc = average_precision_score(y_true, y_prob)

            # Compute curves for plotting
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            report.roc_curve = (fpr, tpr, thresholds)

            precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_prob)
            report.pr_curve = (precision_curve, recall_curve, thresholds_pr)
        except Exception:
            pass

    return report


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    sources: Optional[np.ndarray] = None
) -> Dict[str, MetricsReport]:
    """
    Compute metrics overall and per-source.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        sources: Source labels for per-source analysis (optional)

    Returns:
        Dictionary with 'overall' and per-source metrics
    """
    results = {}

    # Overall metrics
    results['overall'] = compute_metrics(y_true, y_pred, y_prob)

    # Per-source metrics
    if sources is not None:
        unique_sources = np.unique(sources)
        for source in unique_sources:
            mask = sources == source
            if mask.sum() > 0:
                source_prob = y_prob[mask] if y_prob is not None else None
                results[f'source_{source}'] = compute_metrics(
                    y_true[mask],
                    y_pred[mask],
                    source_prob
                )

    return results


def aggregate_cv_metrics(fold_metrics: List[MetricsReport]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across cross-validation folds.

    Args:
        fold_metrics: List of MetricsReport from each fold

    Returns:
        Dictionary with mean and std for each metric
    """
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

    aggregated = {}
    for name in metric_names:
        values = [getattr(m, name) for m in fold_metrics]
        aggregated[name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }

    return aggregated


def print_cv_summary(aggregated: Dict[str, Dict[str, float]]) -> str:
    """
    Generate summary string for cross-validation results.

    Args:
        aggregated: Output from aggregate_cv_metrics

    Returns:
        Formatted summary string
    """
    lines = [
        "Cross-Validation Results (5-Fold)",
        "=" * 40,
        f"{'Metric':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}",
        "-" * 40
    ]

    for name, stats in aggregated.items():
        lines.append(
            f"{name:<12} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
            f"{stats['min']:>10.4f} {stats['max']:>10.4f}"
        )

    return '\n'.join(lines)
