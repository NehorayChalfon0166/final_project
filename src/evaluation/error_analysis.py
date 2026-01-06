"""
Error Analysis Module
=====================
Analyze false positives and false negatives for model debugging.
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ErrorCase:
    """Container for a misclassified case."""
    address: str
    true_label: int
    pred_label: int
    confidence: float
    source: str
    features: Dict[str, float]
    graph_stats: Dict[str, float]


class ErrorAnalyzer:
    """
    Analyze classification errors (FP/FN).

    Provides insights into:
    - Feature patterns of misclassified samples
    - Common characteristics of errors
    - Per-source error breakdown
    - Export to CSV for manual review
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        output_dir: str = "evaluation/results"
    ):
        """
        Initialize error analyzer.

        Args:
            feature_names: List of feature column names
            output_dir: Directory for output files
        """
        self.feature_names = feature_names or [
            'lifetime_seconds_log', 'activity_rate_log', 'in_out_balance_log',
            'total_txs_log', 'send_receive_ratio_log', 'fee_per_tx_log',
            'blocks_btwn_txs_mean_log', 'fee_share_mean_log', 'avg_tx_size_log',
            'tx_size_range_log', 'max_sent_log', 'max_received_log'
        ]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.false_positives: List[ErrorCase] = []
        self.false_negatives: List[ErrorCase] = []
        self.true_positives: List[ErrorCase] = []
        self.true_negatives: List[ErrorCase] = []

    def analyze(
        self,
        graphs: List[Data],
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Analyze predictions and categorize errors.

        Args:
            graphs: List of graph data objects
            predictions: Model predictions (0 or 1)
            probabilities: Prediction probabilities for positive class
            labels: Ground truth labels

        Returns:
            Analysis summary dictionary
        """
        self._clear()

        for i, graph in enumerate(graphs):
            pred = int(predictions[i])
            label = int(labels[i])
            prob = float(probabilities[i])

            # Extract info from graph
            address = getattr(graph, 'address', f'unknown_{i}')
            source = getattr(graph, 'source', 'unknown')

            # Get center node features
            features = self._extract_features(graph)
            graph_stats = self._compute_graph_stats(graph)

            case = ErrorCase(
                address=address,
                true_label=label,
                pred_label=pred,
                confidence=prob if pred == 1 else (1 - prob),
                source=source,
                features=features,
                graph_stats=graph_stats
            )

            # Categorize
            if pred == 1 and label == 0:
                self.false_positives.append(case)
            elif pred == 0 and label == 1:
                self.false_negatives.append(case)
            elif pred == 1 and label == 1:
                self.true_positives.append(case)
            else:
                self.true_negatives.append(case)

        return self._generate_summary()

    def _clear(self):
        """Clear previous analysis."""
        self.false_positives = []
        self.false_negatives = []
        self.true_positives = []
        self.true_negatives = []

    def _extract_features(self, graph: Data) -> Dict[str, float]:
        """Extract center node features."""
        features = {}
        if graph.x is not None and len(graph.x) > 0:
            center_features = graph.x[0].cpu().numpy()
            for i, name in enumerate(self.feature_names):
                if i < len(center_features):
                    features[name] = float(center_features[i])
        return features

    def _compute_graph_stats(self, graph: Data) -> Dict[str, float]:
        """Compute graph-level statistics."""
        stats = {}

        # Number of nodes and edges
        stats['num_nodes'] = graph.x.size(0) if graph.x is not None else 0
        stats['num_edges'] = graph.edge_index.size(1) if graph.edge_index is not None else 0

        # Edge density
        if stats['num_nodes'] > 1:
            max_edges = stats['num_nodes'] * (stats['num_nodes'] - 1)
            stats['edge_density'] = stats['num_edges'] / max_edges if max_edges > 0 else 0
        else:
            stats['edge_density'] = 0

        # Average edge weight (if edge_attr exists)
        if graph.edge_attr is not None and len(graph.edge_attr) > 0:
            stats['avg_edge_weight'] = float(graph.edge_attr[:, 0].mean())
        else:
            stats['avg_edge_weight'] = 0

        return stats

    def _generate_summary(self) -> Dict:
        """Generate analysis summary."""
        total = len(self.false_positives) + len(self.false_negatives) + \
                len(self.true_positives) + len(self.true_negatives)

        summary = {
            'total_samples': total,
            'false_positives': len(self.false_positives),
            'false_negatives': len(self.false_negatives),
            'true_positives': len(self.true_positives),
            'true_negatives': len(self.true_negatives),
            'fp_rate': len(self.false_positives) / (len(self.false_positives) + len(self.true_negatives)) if (len(self.false_positives) + len(self.true_negatives)) > 0 else 0,
            'fn_rate': len(self.false_negatives) / (len(self.false_negatives) + len(self.true_positives)) if (len(self.false_negatives) + len(self.true_positives)) > 0 else 0,
        }

        # Per-source breakdown
        summary['per_source'] = self._per_source_breakdown()

        # Feature statistics for errors
        summary['fp_feature_stats'] = self._compute_feature_stats(self.false_positives)
        summary['fn_feature_stats'] = self._compute_feature_stats(self.false_negatives)

        # Comparison with correct predictions
        summary['tp_feature_stats'] = self._compute_feature_stats(self.true_positives)
        summary['tn_feature_stats'] = self._compute_feature_stats(self.true_negatives)

        return summary

    def _per_source_breakdown(self) -> Dict[str, Dict[str, int]]:
        """Breakdown errors by source."""
        breakdown = defaultdict(lambda: {'fp': 0, 'fn': 0, 'tp': 0, 'tn': 0})

        for case in self.false_positives:
            breakdown[case.source]['fp'] += 1
        for case in self.false_negatives:
            breakdown[case.source]['fn'] += 1
        for case in self.true_positives:
            breakdown[case.source]['tp'] += 1
        for case in self.true_negatives:
            breakdown[case.source]['tn'] += 1

        return dict(breakdown)

    def _compute_feature_stats(self, cases: List[ErrorCase]) -> Dict[str, Dict[str, float]]:
        """Compute feature statistics for a set of cases."""
        if not cases:
            return {}

        feature_values = defaultdict(list)
        for case in cases:
            for name, value in case.features.items():
                feature_values[name].append(value)

        stats = {}
        for name, values in feature_values.items():
            arr = np.array(values)
            stats[name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr))
            }

        return stats

    def get_discriminative_features(self) -> Dict[str, float]:
        """
        Find features that discriminate between FP/FN and correct predictions.

        Returns:
            Dictionary of feature names to discrimination scores
        """
        scores = {}

        for feature in self.feature_names:
            # FP vs TN (what makes benign look criminal?)
            fp_vals = [c.features.get(feature, 0) for c in self.false_positives]
            tn_vals = [c.features.get(feature, 0) for c in self.true_negatives]

            if fp_vals and tn_vals:
                fp_mean = np.mean(fp_vals)
                tn_mean = np.mean(tn_vals)
                pooled_std = np.sqrt((np.var(fp_vals) + np.var(tn_vals)) / 2)

                if pooled_std > 0:
                    scores[f'{feature}_fp_vs_tn'] = abs(fp_mean - tn_mean) / pooled_std

            # FN vs TP (what makes criminal look benign?)
            fn_vals = [c.features.get(feature, 0) for c in self.false_negatives]
            tp_vals = [c.features.get(feature, 0) for c in self.true_positives]

            if fn_vals and tp_vals:
                fn_mean = np.mean(fn_vals)
                tp_mean = np.mean(tp_vals)
                pooled_std = np.sqrt((np.var(fn_vals) + np.var(tp_vals)) / 2)

                if pooled_std > 0:
                    scores[f'{feature}_fn_vs_tp'] = abs(fn_mean - tp_mean) / pooled_std

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def export_errors(self, prefix: str = "errors") -> Tuple[str, str]:
        """
        Export FP and FN cases to CSV files.

        Args:
            prefix: Filename prefix

        Returns:
            Tuple of (fp_path, fn_path)
        """
        fp_path = self.output_dir / f"{prefix}_false_positives.csv"
        fn_path = self.output_dir / f"{prefix}_false_negatives.csv"

        # Export FP
        if self.false_positives:
            fp_data = self._cases_to_dataframe(self.false_positives)
            fp_data.to_csv(fp_path, index=False)
            logger.info(f"Exported {len(self.false_positives)} false positives to {fp_path}")

        # Export FN
        if self.false_negatives:
            fn_data = self._cases_to_dataframe(self.false_negatives)
            fn_data.to_csv(fn_path, index=False)
            logger.info(f"Exported {len(self.false_negatives)} false negatives to {fn_path}")

        return str(fp_path), str(fn_path)

    def _cases_to_dataframe(self, cases: List[ErrorCase]) -> pd.DataFrame:
        """Convert error cases to DataFrame."""
        records = []
        for case in cases:
            record = {
                'address': case.address,
                'true_label': case.true_label,
                'pred_label': case.pred_label,
                'confidence': case.confidence,
                'source': case.source,
                **case.features,
                **{f'graph_{k}': v for k, v in case.graph_stats.items()}
            }
            records.append(record)

        return pd.DataFrame(records)

    def get_high_confidence_errors(self, threshold: float = 0.8) -> Dict[str, List[ErrorCase]]:
        """
        Get errors where model was highly confident (but wrong).

        These are the most concerning cases - model is confidently wrong.

        Args:
            threshold: Confidence threshold

        Returns:
            Dictionary with 'fp' and 'fn' lists
        """
        high_conf_fp = [c for c in self.false_positives if c.confidence >= threshold]
        high_conf_fn = [c for c in self.false_negatives if c.confidence >= threshold]

        return {
            'fp': high_conf_fp,
            'fn': high_conf_fn
        }

    def print_summary(self) -> str:
        """Generate printable summary."""
        lines = [
            "Error Analysis Summary",
            "=" * 50,
            f"Total False Positives: {len(self.false_positives)}",
            f"Total False Negatives: {len(self.false_negatives)}",
            "",
            "Per-Source Breakdown:",
            "-" * 30
        ]

        breakdown = self._per_source_breakdown()
        for source, counts in breakdown.items():
            total = sum(counts.values())
            error_rate = (counts['fp'] + counts['fn']) / total if total > 0 else 0
            lines.append(f"  {source}:")
            lines.append(f"    TP={counts['tp']}, TN={counts['tn']}, FP={counts['fp']}, FN={counts['fn']}")
            lines.append(f"    Error Rate: {error_rate:.2%}")

        lines.extend([
            "",
            "Top Discriminative Features (FP vs TN):",
            "-" * 30
        ])

        disc_features = self.get_discriminative_features()
        fp_features = [(k, v) for k, v in disc_features.items() if '_fp_vs_tn' in k][:5]
        for name, score in fp_features:
            lines.append(f"  {name.replace('_fp_vs_tn', '')}: {score:.3f}")

        lines.extend([
            "",
            "Top Discriminative Features (FN vs TP):",
            "-" * 30
        ])

        fn_features = [(k, v) for k, v in disc_features.items() if '_fn_vs_tp' in k][:5]
        for name, score in fn_features:
            lines.append(f"  {name.replace('_fn_vs_tp', '')}: {score:.3f}")

        return '\n'.join(lines)
