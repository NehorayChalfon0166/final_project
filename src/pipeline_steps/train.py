"""Step 4 — train OptimalBitcoinGNN, calibrate temperature, save artifacts."""
import argparse
import json
import logging
import os
from datetime import datetime

from ._json import NpEncoder
from ._paths import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def step_train_gnn(epochs: int = 100, batch_size: int = 64):
    logger.info("=" * 60)
    logger.info("STEP 4: Training GNN Model")
    logger.info("=" * 60)

    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    from src.graph.config import NUM_EDGE_FEATURES, NUM_NODE_FEATURES
    from src.graph.dataloader import EgoGraphDataset, get_test_loader, get_train_val_loaders
    from src.models.optimal_gnn import EarlyStopping, OptimalBitcoinGNN, TemperatureScaler
    from src.models.train_optimal import evaluate, save_checkpoint, train_epoch
    from src.models.utils import get_center_labels

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    full_train_dataset = EgoGraphDataset(split="train")
    test_dataset = EgoGraphDataset(split="test")
    if len(full_train_dataset) == 0:
        logger.error("No training graphs found! Run step 2 (graph building) first.")
        return None

    logger.info(f"Total training graphs: {len(full_train_dataset):,}")
    logger.info(f"Test graphs: {len(test_dataset):,}")

    # MPS attempted earlier — first epoch hung past 7 min on this graph workload
    # (likely silent CPU fallback for unsupported scatter ops in PyG). Stick to
    # CPU until we have a clean MPS path verified.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        hidden_dim=64,
    ).to(device)

    train_loader, val_loader = get_train_val_loaders(batch_size=batch_size, val_ratio=0.15)
    test_loader = get_test_loader(batch_size=batch_size)
    logger.info(
        f"Train/Val split: {len(train_loader.dataset):,} train, "
        f"{len(val_loader.dataset):,} val"
    )

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.005,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    early_stopping = EarlyStopping(patience=20, mode="max")

    history = {"train_loss": [], "val_f1": []}
    best_f1 = 0.0
    best_epoch = 0
    model_path = os.path.join(OUTPUTS_DIR, "gnn_model.pt")
    checkpoint_path = os.path.join(OUTPUTS_DIR, "gnn_checkpoint.pt")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_metrics["f1"])

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss={train_loss:.4f}, Val F1={val_metrics['f1']:.4f}")

        if (epoch + 1) % 20 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, history,
                best_f1, best_epoch,
                argparse.Namespace(
                    epochs=epochs, batch_size=batch_size, lr=0.001,
                    max_lr=0.005, weight_decay=0.01, patience=20,
                    hidden_dim=64, use_focal_loss=False, label_smoothing=0.0,
                    save_path=model_path, log_interval=10,
                    checkpoint_interval=20, resume=None,
                ),
                checkpoint_path,
            )

        if early_stopping(val_metrics["f1"], model):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Final evaluation on held-out test set with the best checkpoint.
    model.load_state_dict(torch.load(model_path, weights_only=True))
    final_metrics = evaluate(model, test_loader, device)

    logger.info("\nCalibrating temperature scaling on validation set...")
    temp_scaler = TemperatureScaler.calibrate(model, val_loader, device, get_center_labels)
    temp_value = temp_scaler.temperature.item()
    logger.info(f"  Optimal temperature: {temp_value:.3f}")

    temp_path = os.path.join(OUTPUTS_DIR, "temperature.pt")
    torch.save({"temperature": temp_value}, temp_path)
    logger.info(f"  Temperature saved to: {temp_path}")

    history_path = os.path.join(OUTPUTS_DIR, "gnn_training_history.json")
    with open(history_path, "w") as f:
        json.dump(
            {
                "history": history,
                "final_metrics": final_metrics,
                "best_epoch": best_epoch,
                "best_val_f1": best_f1,
                "epochs_trained": epoch + 1,
                "temperature": temp_value,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            cls=NpEncoder,
        )

    logger.info("\nGNN Final Results (held-out test set):")
    logger.info(f"  Best Epoch: {best_epoch} (selected by Val F1: {best_f1:.4f})")
    logger.info(f"  Accuracy:   {final_metrics['accuracy']:.4f}")
    logger.info(f"  Precision:  {final_metrics['precision']:.4f}")
    logger.info(f"  Recall:     {final_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:   {final_metrics['f1']:.4f}")

    return final_metrics
