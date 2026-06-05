"""Train the BasicGCN baseline on cached ego-graphs.

Mirrors the protocol used by ``src/pipeline_steps/train.py``:
- Train on graphs from the train split (85/15 train/val by deterministic seed)
- Track best epoch by validation F1, save weights at that epoch
- Final evaluation on the held-out test split

Outputs:
- outputs/baseline/gcn_model.pt
- outputs/baseline/gcn_results.json   (test-set metrics)
- outputs/baseline/gcn_history.json   (per-epoch loss / val F1)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baselines.gcn_baseline import BasicGCN, count_parameters
from src.graph.config import NUM_NODE_FEATURES
from src.graph.dataloader import get_test_loader, get_train_val_loaders
from src.models.utils import get_center_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total, batches = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        labels = get_center_labels(batch)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
        batches += 1
    return total / max(batches, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, probs, labels = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        p = torch.softmax(out, dim=1)
        preds.extend(out.argmax(dim=1).cpu().numpy())
        probs.extend(p[:, 1].cpu().numpy())
        labels.extend(get_center_labels(batch).cpu().numpy())

    y_true = np.array(labels)
    y_pred = np.array(preds)
    y_prob = np.array(probs)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n_samples": int(len(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "baseline"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_loader, val_loader = get_train_val_loaders(
        batch_size=args.batch_size, val_ratio=0.15, num_workers=args.num_workers
    )
    test_loader = get_test_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    logger.info(
        f"Train graphs: {len(train_loader.dataset):,} | "
        f"Val: {len(val_loader.dataset):,} | "
        f"Test: {len(test_loader.dataset):,}"
    )

    model = BasicGCN(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)
    logger.info(f"BasicGCN parameters: {count_parameters(model):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    model_path = os.path.join(args.output_dir, "gcn_model.pt")
    history = {"train_loss": [], "val_f1": [], "val_accuracy": []}
    best_f1 = 0.0
    best_epoch = -1
    epochs_since_improve = 0

    train_start = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_metrics["f1"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        improved = val_metrics["f1"] > best_f1 + 1e-4
        if improved:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        logger.info(
            f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
            f"val_acc {val_metrics['accuracy']:.4f} | "
            f"val_f1 {val_metrics['f1']:.4f} | "
            f"best {best_f1:.4f}@{best_epoch} | "
            f"{epoch_time:.1f}s"
        )

        if epochs_since_improve >= args.patience:
            logger.info(f"Early stopping at epoch {epoch} (no val_f1 improvement for {args.patience} epochs)")
            break

    training_time = time.time() - train_start

    # Load best checkpoint and evaluate once on test set.
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    logger.info("\nTest metrics:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v}")

    payload = {
        "model": "BasicGCN",
        "protocol": "train on cached train-split ego-graphs, evaluate on cached test-split ego-graphs",
        "architecture": {
            "layers": 3,
            "type": "GCNConv",
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "edge_attr_used": False,
            "readout": "global_mean_pool",
            "num_node_features": NUM_NODE_FEATURES,
            "num_parameters": int(count_parameters(model)),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "epochs_trained": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "best_val_f1": best_f1,
        },
        "results": {**test_metrics, "training_time": training_time},
        "timestamp": datetime.now().isoformat(),
    }
    results_path = os.path.join(args.output_dir, "gcn_results.json")
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    history_path = os.path.join(args.output_dir, "gcn_history.json")
    with open(history_path, "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "best_val_f1": best_f1}, f, indent=2)
    logger.info(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
