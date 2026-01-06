"""
Feature Engineering Module
===========================
Scripts for extracting and merging features from REAL-CATS and Elliptic++ datasets.

Usage:
    from src.features.pipeline_conservative import main as run_feature_pipeline
    run_feature_pipeline()
"""

from .pipeline_conservative import main as run_feature_pipeline
from .prepare_balanced_dataset import main as prepare_balanced

__all__ = ['run_feature_pipeline', 'prepare_balanced']
