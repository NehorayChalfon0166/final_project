"""
Baseline Models
===============
Tabular (XGBoost) and graph-aware (GCN) baselines for comparison with the
OptimalBitcoinGNN model.
"""
from .xgboost_baseline import XGBoostBaseline
from .gcn_baseline import BasicGCN

__all__ = ['XGBoostBaseline', 'BasicGCN']
