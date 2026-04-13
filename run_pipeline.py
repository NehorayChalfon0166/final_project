"""
Master Pipeline
===============
Runs the full pipeline from data preparation to model evaluation.

Steps:
1. Prepare unified dataset (REAL-CATS behavioral + Elliptic++)
2. Build ego-graphs for all addresses
3. Train XGBoost baseline
4. Train GNN model
5. Evaluate models and compare results

Usage:
    python run_pipeline.py --all              # Run everything
    python run_pipeline.py --prepare          # Only data preparation
    python run_pipeline.py --graphs           # Only graph building
    python run_pipeline.py --baseline         # Only XGBoost baseline
    python run_pipeline.py --train            # Only GNN training
    python run_pipeline.py --evaluate         # Only evaluation & comparison
    python run_pipeline.py --graphs --retry-failed  # Retry failed graph builds
"""
import os
import json
import argparse
import logging
import asyncio
from datetime import datetime
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')


# class to handle the conversion of numpy types for JSON
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def step_prepare_data():
    """Step 1: Prepare unified dataset from REAL-CATS and Elliptic++."""
    logger.info("=" * 60)
    logger.info("STEP 1: Preparing Unified Dataset")
    logger.info("=" * 60)

    feature_matrix = os.path.join(PROJECT_ROOT, 'src', 'features', 'output',
                                   'conservative_feature_matrix_with_logs.csv')

    if not os.path.exists(feature_matrix):
        logger.info("Running feature engineering pipeline...")
        from src.features.pipeline_conservative import main as run_feature_pipeline
        run_feature_pipeline()
    else:
        logger.info(f"Feature matrix exists: {feature_matrix}")

    logger.info("\nCreating balanced dataset...")
    from src.features.prepare_balanced_dataset import main as prepare_balanced
    prepare_balanced(use_log=True)

    balanced_path = os.path.join(PROJECT_ROOT, 'src', 'features', 'output',
                                  'balanced_training_dataset.csv')
    if os.path.exists(balanced_path):
        import pandas as pd
        df = pd.read_csv(balanced_path)
        logger.info(f"\nDataset ready: {len(df):,} addresses")
        logger.info(f"  Benign: {(df['label'] == 0).sum():,}")
        logger.info(f"  Criminal: {(df['label'] == 1).sum():,}")
        return True
    return False


def step_build_graphs(max_addresses: int = None, split: str = 'both', retry_failed: bool = False):
    """Step 2: Build ego-graphs using mempool.space API."""
    logger.info("=" * 60)
    logger.info("STEP 2: Building Ego-Graphs")
    logger.info("=" * 60)

    from src.graph.pipeline import GraphConstructionPipeline

    splits = ['train', 'test'] if split == 'both' else [split]

    for s in splits:
        logger.info(f"\nProcessing {s} split...")
        pipeline = GraphConstructionPipeline(s)

        # Pass the retry_failed flag to the pipeline
        if retry_failed:
            logger.info("Retrying failed addresses only...")
            
        pending = pipeline.get_pending_addresses()
        logger.info(f"  Pending addresses: {len(pending):,}")

        # If we have pending addresses OR we want to retry failed ones, run it
        if pending or retry_failed:
            asyncio.run(pipeline.run(
                max_addresses=max_addresses,
                resume=True,
                retry_failed=retry_failed  # <--- Passed down here
            ))


def step_train_baseline():
    """Step 3: Train XGBoost baseline."""
    logger.info("=" * 60)
    logger.info("STEP 3: Training XGBoost Baseline")
    logger.info("=" * 60)

    from src.baselines.xgboost_baseline import XGBoostBaseline

    baseline = XGBoostBaseline(output_dir=os.path.join(OUTPUTS_DIR, 'baseline'))
    baseline.load_data()
    results = baseline.train()
    baseline.save_results()
    baseline.save_model()

    return results


def step_train_gnn(epochs: int = 100, batch_size: int = 64):
    """Step 4: Train GNN model."""
    logger.info("=" * 60)
    logger.info("STEP 4: Training GNN Model")
    logger.info("=" * 60)

    import torch
    from src.graph.dataloader import EgoGraphDataset, get_train_loader, get_test_loader
    from src.graph.config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES
    from src.models.optimal_gnn import OptimalBitcoinGNN, EarlyStopping
    from src.models.train_optimal import train_epoch, evaluate, save_checkpoint

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    train_dataset = EgoGraphDataset(split='train')
    test_dataset = EgoGraphDataset(split='test')

    if len(train_dataset) == 0:
        logger.error("No training graphs found! Run step 2 (graph building) first.")
        return None

    logger.info(f"Training graphs: {len(train_dataset):,}")
    logger.info(f"Test graphs: {len(test_dataset):,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        hidden_dim=64
    ).to(device)

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    import torch.nn as nn

    train_loader = get_train_loader(batch_size=batch_size, shuffle=True)
    test_loader = get_test_loader(batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer, max_lr=0.005, epochs=epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    early_stopping = EarlyStopping(patience=20, mode='max')

    best_f1 = 0
    best_epoch = 0
    history = {'train_loss': [], 'test_f1': []}
    model_path = os.path.join(OUTPUTS_DIR, 'gnn_model.pt')

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        test_metrics = evaluate(model, test_loader, device)

        history['train_loss'].append(train_loss)
        history['test_f1'].append(test_metrics['f1'])

        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss={train_loss:.4f}, F1={test_metrics['f1']:.4f}")

        if (epoch + 1) % 20 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, history,
                          best_f1, best_epoch, argparse.Namespace(
                              epochs=epochs, batch_size=batch_size, lr=0.001,
                              max_lr=0.005, weight_decay=0.01, patience=20,
                              hidden_dim=64, use_focal_loss=False, label_smoothing=0.0,
                              save_path=model_path, log_interval=10,
                              checkpoint_interval=20, resume=None
                          ), os.path.join(OUTPUTS_DIR, 'gnn_checkpoint.pt'))

        if early_stopping(test_metrics['f1'], model):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(model_path, weights_only=True))
    final_metrics = evaluate(model, test_loader, device)

    # Temperature scaling calibration
    from src.models.optimal_gnn import TemperatureScaler
    from src.models.train_optimal import get_center_labels

    logger.info("\nCalibrating temperature scaling on test set...")
    temp_scaler = TemperatureScaler.calibrate(
        model, test_loader, device, get_center_labels
    )
    temp_value = temp_scaler.temperature.item()
    logger.info(f"  Optimal temperature: {temp_value:.3f}")

    # Save temperature alongside model
    temp_path = os.path.join(OUTPUTS_DIR, 'temperature.pt')
    torch.save({'temperature': temp_value}, temp_path)
    logger.info(f"  Temperature saved to: {temp_path}")

    # Save training history
    history_path = os.path.join(OUTPUTS_DIR, 'gnn_training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'history': history,
            'final_metrics': final_metrics,
            'best_epoch': best_epoch,
            'epochs_trained': epoch + 1,
            'temperature': temp_value,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, cls=NpEncoder)

    logger.info(f"\nGNN Final Results:")
    logger.info(f"  Best Epoch: {best_epoch}")
    logger.info(f"  Accuracy:   {final_metrics['accuracy']:.4f}")
    logger.info(f"  Precision:  {final_metrics['precision']:.4f}")
    logger.info(f"  Recall:     {final_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:   {final_metrics['f1']:.4f}")

    return final_metrics


def step_evaluate():
    """Step 5: Evaluate GNN model and compare with baseline."""
    logger.info("=" * 60)
    logger.info("STEP 5: Model Evaluation")
    logger.info("=" * 60)

    import torch
    import numpy as np
    from src.graph.dataloader import EgoGraphDataset
    from src.graph.config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES, FEATURE_COLUMNS
    from src.models.optimal_gnn import OptimalBitcoinGNN
    from src.evaluation.metrics import compute_metrics

    eval_dir = os.path.join(OUTPUTS_DIR, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(OUTPUTS_DIR, 'gnn_model.pt')

    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"GNN model not found at {model_path}. Run --train first.")
        return None

    # Load model
    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        hidden_dim=64
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"Loaded model from {model_path}")

    # Load test dataset
    test_dataset = EgoGraphDataset(split='test')
    if len(test_dataset) == 0:
        logger.error("No test graphs found. Run --graphs first.")
        return None

    logger.info(f"Test dataset: {len(test_dataset)} graphs")

    # Evaluate GNN
    logger.info("\nEvaluating GNN model...")
    from torch_geometric.loader import DataLoader
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)

            # Get center node labels
            if hasattr(batch, 'ptr'):
                center_indices = batch.ptr[:-1]
            else:
                batch_size = batch.batch.max().item() + 1
                center_indices = []
                for i in range(batch_size):
                    mask = (batch.batch == i)
                    first_idx = mask.nonzero(as_tuple=True)[0][0]
                    center_indices.append(first_idx)
                center_indices = torch.tensor(center_indices, device=device)

            labels = batch.y[center_indices]

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute GNN metrics
    gnn_metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )

    logger.info("\n" + str(gnn_metrics))

    # Load XGBoost results for comparison
    xgb_path = os.path.join(OUTPUTS_DIR, 'baseline', 'xgboost_results.json')
    xgb_results = None
    if os.path.exists(xgb_path):
        with open(xgb_path) as f:
            xgb_data = json.load(f)
            xgb_results = xgb_data['results']
    else:
        # Try old path
        old_xgb_path = 'src/baselines/results/xgboost_results.json'
        if os.path.exists(old_xgb_path):
            with open(old_xgb_path) as f:
                xgb_data = json.load(f)
                xgb_results = xgb_data['results']

    # Comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<15} {'XGBoost':>15} {'GNN':>15} {'Difference':>15}")
    print("-" * 70)

    comparison = {}
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    for metric in metrics_to_compare:
        gnn_val = getattr(gnn_metrics, metric, 0)
        xgb_val = xgb_results.get(metric, 0) if xgb_results else 0

        diff = gnn_val - xgb_val
        sign = '+' if diff >= 0 else ''

        xgb_str = f"{xgb_val*100:.2f}%" if xgb_results else "N/A"
        print(f"{metric:<15} {xgb_str:>15} {gnn_val*100:>14.2f}% {sign}{diff*100:>14.2f}%")

        comparison[metric] = {
            'xgboost': xgb_val if xgb_results else None,
            'gnn': gnn_val,
            'diff': diff
        }

    print("=" * 70)

    # Verdict
    if xgb_results:
        gnn_f1 = gnn_metrics.f1
        xgb_f1 = xgb_results.get('f1', 0)

        if gnn_f1 > xgb_f1:
            print(f"\n>>> GNN outperforms XGBoost by {(gnn_f1-xgb_f1)*100:.2f}% F1")
            print("    Graph structure provides value for this task!")
        else:
            print(f"\n>>> XGBoost outperforms GNN by {(xgb_f1-gnn_f1)*100:.2f}% F1")
            print("    Tabular features may be sufficient for this task.")

    # Save evaluation results
    eval_results = {
        'gnn': gnn_metrics.to_dict(),
        'xgboost': xgb_results,
        'comparison': comparison,
        'test_samples': len(test_dataset),
        'timestamp': datetime.now().isoformat()
    }

    eval_path = os.path.join(eval_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        # Also use NpEncoder here to be safe
        json.dump(eval_results, f, indent=2, cls=NpEncoder)

    logger.info(f"\nEvaluation results saved to: {eval_path}")

    # Run error analysis if available
    try:
        from src.evaluation.error_analysis import ErrorAnalyzer

        logger.info("\nRunning error analysis...")
        graphs = [test_dataset[i] for i in range(len(test_dataset))]

        analyzer = ErrorAnalyzer(
            feature_names=FEATURE_COLUMNS,
            output_dir=eval_dir
        )

        analyzer.analyze(
            graphs=graphs,
            predictions=np.array(all_preds),
            probabilities=np.array(all_probs),
            labels=np.array(all_labels)
        )

        logger.info(analyzer.print_summary())

        fp_path, fn_path = analyzer.export_errors(prefix='gnn')
        logger.info(f"False positives exported to: {fp_path}")
        logger.info(f"False negatives exported to: {fn_path}")

    except Exception as e:
        logger.warning(f"Error analysis skipped: {e}")

    # Run interpretability analysis if available
    try:
        from src.evaluation.interpretability import ModelInterpreter

        logger.info("\nRunning interpretability analysis...")
        interpreter = ModelInterpreter(
            model=model,
            feature_names=FEATURE_COLUMNS,
            device=device
        )

        # Feature importance on subset
        sample_graphs = [test_dataset[i] for i in range(min(100, len(test_dataset)))]
        importance = interpreter.compute_feature_importance(sample_graphs)

        logger.info("\nTop 5 Important Features:")
        for i, (name, score) in enumerate(list(importance.items())[:5]):
            logger.info(f"  {i+1}. {name}: {score:.4f}")

        # Save interpretability results
        interp_path = os.path.join(eval_dir, 'feature_importance.json')
        with open(interp_path, 'w') as f:
            json.dump({
                'feature_importance': importance,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, cls=NpEncoder)

        logger.info(f"Feature importance saved to: {interp_path}")

    except Exception as e:
        logger.warning(f"Interpretability analysis skipped: {e}")

    return eval_results


def main():
    parser = argparse.ArgumentParser(description='Run the full pipeline')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--prepare', action='store_true', help='Step 1: Prepare data')
    parser.add_argument('--graphs', action='store_true', help='Step 2: Build graphs')
    parser.add_argument('--baseline', action='store_true', help='Step 3: Train XGBoost')
    parser.add_argument('--train', action='store_true', help='Step 4: Train GNN')
    parser.add_argument('--evaluate', action='store_true', help='Step 5: Evaluate & compare')
    
    # New flag for retrying failed graphs
    parser.add_argument('--retry-failed', action='store_true', help='Retry building failed graphs')

    # Deprecated alias
    parser.add_argument('--compare', action='store_true', help='(Deprecated: use --evaluate)')

    # Graph building options
    parser.add_argument('--max-graphs', type=int, default=None,
                        help='Max graphs to build (for testing)')
    parser.add_argument('--split', choices=['train', 'test', 'both'], default='both',
                        help='Which split to build graphs for')

    # Training options
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')

    args = parser.parse_args()

    # Handle deprecated --compare
    if args.compare:
        args.evaluate = True

    # If no specific step, default to --all
    if not any([args.all, args.prepare, args.graphs, args.baseline, args.train, args.evaluate]):
        args.all = True

    # Create outputs directory
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    start_time = datetime.now()
    logger.info(f"Pipeline started at {start_time}")

    try:
        if args.all or args.prepare:
            step_prepare_data()

        if args.all or args.graphs:
            step_build_graphs(
                max_addresses=args.max_graphs, 
                split=args.split, 
                retry_failed=args.retry_failed # <--- Passed here
            )

        if args.all or args.baseline:
            step_train_baseline()

        if args.all or args.train:
            step_train_gnn(epochs=args.epochs, batch_size=args.batch_size)

        if args.all or args.evaluate:
            step_evaluate()

    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"\nPipeline completed in {duration}")


if __name__ == "__main__":
    main()