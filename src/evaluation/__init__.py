"""
Evaluation Framework for Bitcoin Wallet GNN Classifier
=======================================================
Comprehensive evaluation including:
- Standard metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
- K-fold cross-validation
- Per-source analysis (REAL-CATS vs Elliptic++)
- Error analysis (FP/FN)
- Model interpretability (attention, feature importance, embeddings)
"""

from .metrics import (
    compute_metrics,
    compute_all_metrics,
    aggregate_cv_metrics,
    print_cv_summary,
    MetricsReport
)
from .cross_validation import KFoldCV, CVConfig
from .error_analysis import ErrorAnalyzer
from .interpretability import ModelInterpreter

__all__ = [
    'compute_metrics',
    'compute_all_metrics',
    'aggregate_cv_metrics',
    'print_cv_summary',
    'MetricsReport',
    'KFoldCV',
    'CVConfig',
    'ErrorAnalyzer',
    'ModelInterpreter'
]
