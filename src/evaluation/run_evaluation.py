"""
Evaluation Runner
=================
Main script for running full model evaluation:
- Standard metrics
- K-fold cross-validation
- Error analysis
- Model interpretability

Usage:
    python -m evaluation.run_evaluation --model-path models/optimal_gnn_model.pt
    python -m evaluation.run_evaluation --cross-validation --n-folds 5
"""
import os
import sys
import argparse
import logging
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.graph.dataloader import EgoGraphDataset, get_test_loader
from src.graph.config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES, FEATURE_COLUMNS
from src.models.optimal_gnn import OptimalBitcoinGNN
from src.models.utils import get_center_labels

from .metrics import compute_metrics, compute_all_metrics, MetricsReport
from .cross_validation import KFoldCV, CVConfig
from .error_analysis import ErrorAnalyzer
from .interpretability import ModelInterpreter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> OptimalBitcoinGNN:
    """Load trained model from checkpoint."""
    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        hidden_dim=64
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning(f"Model path not found: {model_path}, using random weights")

    return model


@torch.no_grad()
def evaluate_model(
    model: OptimalBitcoinGNN,
    dataset: EgoGraphDataset,
    device: torch.device,
    batch_size: int = 64
) -> dict:
    """
    Run full evaluation on a dataset.

    Returns:
        Dictionary with predictions, probabilities, labels, sources
    """
    model.eval()

    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []
    all_sources = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        probs = torch.softmax(out, dim=1)
        preds = out.argmax(dim=1)
        labels = get_center_labels(batch)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Get sources from dataset
    for i in range(len(dataset)):
        graph = dataset[i]
        source = getattr(graph, 'source', 'unknown')
        all_sources.append(source)

    return {
        'predictions': np.array(all_preds),
        'probabilities': np.array(all_probs),
        'labels': np.array(all_labels),
        'sources': np.array(all_sources)
    }


def run_standard_evaluation(args, device):
    """Run standard evaluation (metrics only)."""
    logger.info("=" * 60)
    logger.info("Running Standard Evaluation")
    logger.info("=" * 60)

    # Load model
    model = load_model(args.model_path, device)

    # Load test dataset
    test_dataset = EgoGraphDataset(split='test')
    if len(test_dataset) == 0:
        logger.error("No test graphs found. Run graph pipeline first.")
        return None

    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Evaluate
    results = evaluate_model(model, test_dataset, device, args.batch_size)

    # Compute metrics
    metrics = compute_all_metrics(
        results['labels'],
        results['predictions'],
        results['probabilities'],
        results['sources']
    )

    # Print results
    logger.info("\n" + str(metrics['overall']))

    # Print per-source results
    for key, report in metrics.items():
        if key.startswith('source_'):
            logger.info(f"\n{key}:")
            logger.info(f"  Accuracy:  {report.accuracy:.4f}")
            logger.info(f"  Precision: {report.precision:.4f}")
            logger.info(f"  Recall:    {report.recall:.4f}")
            logger.info(f"  F1:        {report.f1:.4f}")

    return metrics


def run_cross_validation(args, device):
    """Run K-fold cross-validation."""
    logger.info("=" * 60)
    logger.info(f"Running {args.n_folds}-Fold Cross-Validation")
    logger.info("=" * 60)

    # Load full dataset (combine train and test)
    train_dataset = EgoGraphDataset(split='train')
    test_dataset = EgoGraphDataset(split='test')

    all_graphs = [train_dataset[i] for i in range(len(train_dataset))]
    all_graphs.extend([test_dataset[i] for i in range(len(test_dataset))])

    if len(all_graphs) == 0:
        logger.error("No graphs found. Run graph pipeline first.")
        return None

    logger.info(f"Total graphs for CV: {len(all_graphs)}")

    # Configure CV
    config = CVConfig(
        n_folds=args.n_folds,
        epochs=args.cv_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience
    )

    # Run CV
    cv = KFoldCV(
        model_class=OptimalBitcoinGNN,
        model_kwargs={
            'num_node_features': NUM_NODE_FEATURES,
            'num_edge_features': NUM_EDGE_FEATURES,
            'hidden_dim': args.hidden_dim
        },
        config=config,
        device=device
    )

    results = cv.run(all_graphs)

    # Save results
    output_path = Path(args.output_dir) / 'cv_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'aggregated': results['aggregated'],
            'config': results['config'],
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"CV results saved to {output_path}")

    return results


def run_error_analysis(args, device):
    """Run error analysis on test set."""
    logger.info("=" * 60)
    logger.info("Running Error Analysis")
    logger.info("=" * 60)

    # Load model and data
    model = load_model(args.model_path, device)
    test_dataset = EgoGraphDataset(split='test')

    if len(test_dataset) == 0:
        logger.error("No test graphs found.")
        return None

    # Get predictions
    results = evaluate_model(model, test_dataset, device, args.batch_size)

    # Get graph list
    graphs = [test_dataset[i] for i in range(len(test_dataset))]

    # Run analysis
    analyzer = ErrorAnalyzer(
        feature_names=FEATURE_COLUMNS,
        output_dir=args.output_dir
    )

    summary = analyzer.analyze(
        graphs=graphs,
        predictions=results['predictions'],
        probabilities=results['probabilities'],
        labels=results['labels']
    )

    # Print summary
    logger.info("\n" + analyzer.print_summary())

    # Export errors
    fp_path, fn_path = analyzer.export_errors(prefix='eval')
    logger.info(f"\nExported FP to: {fp_path}")
    logger.info(f"Exported FN to: {fn_path}")

    # High confidence errors
    high_conf = analyzer.get_high_confidence_errors(threshold=0.9)
    logger.info(f"\nHigh-confidence errors (>90%):")
    logger.info(f"  False Positives: {len(high_conf['fp'])}")
    logger.info(f"  False Negatives: {len(high_conf['fn'])}")

    return summary


def run_interpretability(args, device):
    """Run model interpretability analysis."""
    logger.info("=" * 60)
    logger.info("Running Interpretability Analysis")
    logger.info("=" * 60)

    # Load model and data
    model = load_model(args.model_path, device)
    test_dataset = EgoGraphDataset(split='test')

    if len(test_dataset) == 0:
        logger.error("No test graphs found.")
        return None

    graphs = [test_dataset[i] for i in range(min(len(test_dataset), 500))]  # Limit for speed
    labels = np.array([g.y[0].item() for g in graphs])

    # Initialize interpreter
    interpreter = ModelInterpreter(
        model=model,
        feature_names=FEATURE_COLUMNS,
        device=device
    )

    # Feature importance
    logger.info("\nComputing feature importance...")
    importance = interpreter.compute_feature_importance(graphs[:100])
    logger.info("\n" + interpreter.print_feature_importance(importance))

    # Per-class feature importance
    logger.info("\nComputing per-class feature importance...")
    class_importance = interpreter.compute_class_feature_importance(graphs, labels)

    logger.info("\nBenign wallets - top features:")
    for name, score in list(class_importance['benign'].items())[:5]:
        logger.info(f"  {name}: {score:.4f}")

    logger.info("\nCriminal wallets - top features:")
    for name, score in list(class_importance['criminal'].items())[:5]:
        logger.info(f"  {name}: {score:.4f}")

    # Embedding analysis
    logger.info("\nExtracting embeddings...")
    embeddings, emb_labels, sources = interpreter.extract_embeddings(graphs)

    emb_stats = interpreter.get_embedding_stats(embeddings, emb_labels)
    logger.info("\nEmbedding space statistics:")
    for key, value in emb_stats.items():
        logger.info(f"  {key}: {value:.4f}")

    # Save results
    output_path = Path(args.output_dir) / 'interpretability_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'feature_importance': importance,
            'class_importance': class_importance,
            'embedding_stats': emb_stats,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    return {
        'importance': importance,
        'class_importance': class_importance,
        'embedding_stats': emb_stats
    }


def main():
    parser = argparse.ArgumentParser(description='Run GNN model evaluation')

    # Mode selection
    parser.add_argument('--standard', action='store_true', help='Run standard evaluation')
    parser.add_argument('--cross-validation', action='store_true', help='Run cross-validation')
    parser.add_argument('--error-analysis', action='store_true', help='Run error analysis')
    parser.add_argument('--interpretability', action='store_true', help='Run interpretability analysis')
    parser.add_argument('--all', action='store_true', help='Run all evaluations')

    # Model settings
    parser.add_argument('--model-path', type=str, default='models/optimal_gnn_model.pt')
    parser.add_argument('--hidden-dim', type=int, default=64)

    # CV settings
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--cv-epochs', type=int, default=100)

    # Training settings
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=20)

    # Output
    parser.add_argument('--output-dir', type=str, default='evaluation/results')

    args = parser.parse_args()

    # If no mode specified, run all
    if not any([args.standard, args.cross_validation, args.error_analysis, args.interpretability, args.all]):
        args.all = True

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    if args.standard or args.all:
        results['standard'] = run_standard_evaluation(args, device)

    if args.cross_validation or args.all:
        results['cross_validation'] = run_cross_validation(args, device)

    if args.error_analysis or args.all:
        results['error_analysis'] = run_error_analysis(args, device)

    if args.interpretability or args.all:
        results['interpretability'] = run_interpretability(args, device)

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
