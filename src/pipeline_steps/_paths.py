"""Shared path constants for the offline pipeline steps."""
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FEATURE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "features", "output")
