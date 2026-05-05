"""Step 5 — evaluate the trained GNN, compare against the XGBoost baseline,
write per-error CSVs and gradient feature-importance.
"""
import json
import logging
import os
from datetime import datetime

import numpy as np

from ._json import NpEncoder
from ._paths import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def step_evaluate():
    logger.info("=" * 60)
    logger.info("STEP 5: Model Evaluation")
    logger.info("=" * 60)

    import torch
    from torch_geometric.loader import DataLoader

    from src.evaluation.metrics import compute_metrics
    from src.graph.config import FEATURE_COLUMNS, NUM_EDGE_FEATURES, NUM_NODE_FEATURES
    from src.graph.dataloader import EgoGraphDataset
    from src.models.optimal_gnn import OptimalBitcoinGNN
    from src.models.utils import get_center_labels

    eval_dir = os.path.join(OUTPUTS_DIR, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(OUTPUTS_DIR, "gnn_model.pt")
    if not os.path.exists(model_path):
        logger.error(f"GNN model not found at {model_path}. Run --train first.")
        return None

    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        hidden_dim=64,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"Loaded model from {model_path}")

    test_dataset = EgoGraphDataset(split="test")
    if len(test_dataset) == 0:
        logger.error("No test graphs found. Run --graphs first.")
        return None
    logger.info(f"Test dataset: {len(test_dataset)} graphs")

    logger.info("\nEvaluating GNN model...")
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            labels = get_center_labels(batch)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    gnn_metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    logger.info("\n" + str(gnn_metrics))

    xgb_results = _load_xgb_results()
    _print_comparison(gnn_metrics, xgb_results)

    eval_results = {
        "gnn": gnn_metrics.to_dict(),
        "xgboost": xgb_results,
        "comparison": _build_comparison(gnn_metrics, xgb_results),
        "test_samples": len(test_dataset),
        "timestamp": datetime.now().isoformat(),
    }
    eval_path = os.path.join(eval_dir, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, cls=NpEncoder)
    logger.info(f"\nEvaluation results saved to: {eval_path}")

    _run_error_analysis(test_dataset, all_preds, all_probs, all_labels, FEATURE_COLUMNS, eval_dir)
    _run_interpretability(test_dataset, model, FEATURE_COLUMNS, device, eval_dir)

    return eval_results


def _load_xgb_results():
    xgb_path = os.path.join(OUTPUTS_DIR, "baseline", "xgboost_results.json")
    if not os.path.exists(xgb_path):
        return None
    with open(xgb_path) as f:
        return json.load(f)["results"]


def _print_comparison(gnn_metrics, xgb_results) -> None:
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<15} {'XGBoost':>15} {'GNN':>15} {'Difference':>15}")
    print("-" * 70)

    metrics_to_compare = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for metric in metrics_to_compare:
        gnn_val = getattr(gnn_metrics, metric, 0)
        xgb_val = xgb_results.get(metric, 0) if xgb_results else 0
        diff = gnn_val - xgb_val
        sign = "+" if diff >= 0 else ""
        xgb_str = f"{xgb_val*100:.2f}%" if xgb_results else "N/A"
        print(f"{metric:<15} {xgb_str:>15} {gnn_val*100:>14.2f}% {sign}{diff*100:>14.2f}%")
    print("=" * 70)

    if xgb_results:
        gnn_f1 = gnn_metrics.f1
        xgb_f1 = xgb_results.get("f1", 0)
        if gnn_f1 > xgb_f1:
            print(f"\n>>> GNN outperforms XGBoost by {(gnn_f1 - xgb_f1) * 100:.2f}% F1")
            print("    Graph structure provides value for this task!")
        else:
            print(f"\n>>> XGBoost outperforms GNN by {(xgb_f1 - gnn_f1) * 100:.2f}% F1")
            print("    Tabular features may be sufficient for this task.")


def _build_comparison(gnn_metrics, xgb_results) -> dict:
    comparison = {}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        gnn_val = getattr(gnn_metrics, metric, 0)
        xgb_val = xgb_results.get(metric, 0) if xgb_results else 0
        comparison[metric] = {
            "xgboost": xgb_val if xgb_results else None,
            "gnn": gnn_val,
            "diff": gnn_val - xgb_val,
        }
    return comparison


def _run_error_analysis(test_dataset, all_preds, all_probs, all_labels, feature_columns, eval_dir):
    try:
        from src.evaluation.error_analysis import ErrorAnalyzer

        logger.info("\nRunning error analysis...")
        graphs = [test_dataset[i] for i in range(len(test_dataset))]
        analyzer = ErrorAnalyzer(feature_names=feature_columns, output_dir=eval_dir)
        analyzer.analyze(
            graphs=graphs,
            predictions=np.array(all_preds),
            probabilities=np.array(all_probs),
            labels=np.array(all_labels),
        )
        logger.info(analyzer.print_summary())
        fp_path, fn_path = analyzer.export_errors(prefix="gnn")
        logger.info(f"False positives exported to: {fp_path}")
        logger.info(f"False negatives exported to: {fn_path}")
    except Exception as e:
        logger.warning(f"Error analysis skipped: {e}")


def _run_interpretability(test_dataset, model, feature_columns, device, eval_dir):
    try:
        from src.evaluation.interpretability import ModelInterpreter

        logger.info("\nRunning interpretability analysis...")
        interpreter = ModelInterpreter(
            model=model, feature_names=feature_columns, device=device
        )
        sample_graphs = [test_dataset[i] for i in range(min(100, len(test_dataset)))]
        importance = interpreter.compute_feature_importance(sample_graphs)

        logger.info("\nTop 5 Important Features:")
        for i, (name, score) in enumerate(list(importance.items())[:5]):
            logger.info(f"  {i + 1}. {name}: {score:.4f}")

        interp_path = os.path.join(eval_dir, "feature_importance.json")
        with open(interp_path, "w") as f:
            json.dump(
                {
                    "feature_importance": importance,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
                cls=NpEncoder,
            )
        logger.info(f"Feature importance saved to: {interp_path}")
    except Exception as e:
        logger.warning(f"Interpretability analysis skipped: {e}")
