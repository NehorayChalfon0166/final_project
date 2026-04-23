"""
Training Script for OptimalBitcoinGNN
=====================================
Full training pipeline with:
- AdamW optimizer
- OneCycleLR scheduler
- Focal Loss
- Early stopping
- Gradient clipping
- Full metrics logging

Usage:
    python train_optimal.py --epochs 150 --batch-size 64
"""
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.graph.dataloader import get_train_val_loaders, get_test_loader, EgoGraphDataset
from src.graph.config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES
from src.models.optimal_gnn import OptimalBitcoinGNN, FocalLoss, EarlyStopping
from src.models.utils import get_center_labels

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch(model, loader, optimizer, scheduler, criterion, device, max_grad_norm=1.0):
    """
    Train for one epoch.

    Args:
        model: The GNN model
        loader: Training DataLoader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to use
        max_grad_norm: Max gradient norm for clipping

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Get labels for center nodes
        labels = get_center_labels(batch)

        # Compute loss
        loss = criterion(out, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate model on a dataset.

    Args:
        model: The GNN model
        loader: DataLoader
        device: Device to use

    Returns:
        Dictionary with metrics
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Get predictions and probabilities
        probs = torch.softmax(out, dim=1)
        preds = out.argmax(dim=1)

        # Get labels for center nodes
        labels = get_center_labels(batch)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of criminal class

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def save_checkpoint(model, optimizer, scheduler, epoch, history, best_f1, best_epoch, args, path):
    """Save training checkpoint for resume."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'args': vars(args)
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(path, model, optimizer, scheduler):
    """Load training checkpoint."""
    # weights_only=False required for checkpoint containing optimizer/scheduler state
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return (
        checkpoint['epoch'],
        checkpoint['history'],
        checkpoint['best_f1'],
        checkpoint['best_epoch']
    )


def main():
    parser = argparse.ArgumentParser(description='Train OptimalBitcoinGNN')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--max-lr', type=float, default=0.005, help='Max learning rate for OneCycleLR')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--use-focal-loss', action='store_true', help='Use Focal Loss instead of CrossEntropy')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing for CrossEntropy')
    parser.add_argument('--save-path', type=str, default='optimal_gnn_model.pt', help='Path to save model')
    parser.add_argument('--log-interval', type=int, default=5, help='Log every N epochs')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Check if graphs exist
    full_train_dataset = EgoGraphDataset(split='train')
    test_dataset = EgoGraphDataset(split='test')

    if len(full_train_dataset) == 0:
        logger.error("No training graphs found! Run the graph pipeline first:")
        logger.error("  python -m graph_pipeline.pipeline --split both")
        return

    logger.info(f"Total training graphs: {len(full_train_dataset):,}")
    logger.info(f"Test graphs: {len(test_dataset):,}")

    # DataLoaders (train split into train + val; test held out)
    train_loader, val_loader = get_train_val_loaders(
        batch_size=args.batch_size, val_ratio=0.15
    )
    test_loader = get_test_loader(batch_size=args.batch_size)
    logger.info(f"Train/Val split: {len(train_loader.dataset):,} train, {len(val_loader.dataset):,} val")

    # Model
    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        hidden_dim=args.hidden_dim
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # Loss function
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=0.5, gamma=2.0)
        logger.info("Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        logger.info(f"Using CrossEntropyLoss with label_smoothing={args.label_smoothing}")

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'lr': []
    }

    best_f1 = 0
    best_epoch = 0
    start_epoch = 0

    # Resume from checkpoint if specified
    checkpoint_path = args.save_path.replace('.pt', '_checkpoint.pt')
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"Resuming from checkpoint: {args.resume}")
            start_epoch, history, best_f1, best_epoch = load_checkpoint(
                args.resume, model, optimizer, scheduler
            )
            start_epoch += 1  # Start from next epoch
            logger.info(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}, starting fresh")

    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Evaluate on validation set (used for model selection + early stopping)
        val_metrics = evaluate(model, val_loader, device)

        # Evaluate on training set only at log intervals (expensive)
        is_log_epoch = (epoch % args.log_interval == 0 or epoch == args.epochs - 1)
        if is_log_epoch:
            train_metrics = evaluate(model, train_loader, device)
            history['train_acc'].append(train_metrics['accuracy'])
        else:
            history['train_acc'].append(None)

        # Record history
        current_lr = scheduler.get_last_lr()[0]
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['lr'].append(current_lr)

        # Track best model based on VALIDATION F1 (not test)
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            torch.save(model.state_dict(), args.save_path)

        # Logging
        if is_log_epoch:
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Loss: {train_loss:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

        # Save periodic checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, history,
                best_f1, best_epoch, args, checkpoint_path
            )

        # Early stopping on VALIDATION F1
        if early_stopping(val_metrics['f1'], model):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    early_stopping.load_best_model(model)

    # Final evaluation on held-out TEST set (evaluated only once)
    logger.info("=" * 60)
    logger.info("Final Evaluation on Held-Out Test Set (Best Model)")
    logger.info("=" * 60)

    final_metrics = evaluate(model, test_loader, device)

    logger.info(f"""
Final Results:
  Best Epoch:     {best_epoch} (selected by Val F1: {best_f1:.4f})
  Test Accuracy:  {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)
  Test Precision: {final_metrics['precision']:.4f}
  Test Recall:    {final_metrics['recall']:.4f}
  Test F1 Score:  {final_metrics['f1']:.4f}

Confusion Matrix:
  {final_metrics['confusion_matrix']}

Model saved to: {args.save_path}
    """)

    # Save training history
    history_path = args.save_path.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'history': history,
            'final_metrics': {
                'accuracy': final_metrics['accuracy'],
                'precision': final_metrics['precision'],
                'recall': final_metrics['recall'],
                'f1': final_metrics['f1'],
                'confusion_matrix': final_metrics['confusion_matrix']
            },
            'best_epoch': best_epoch,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"Training history saved to: {history_path}")

    return model, history, final_metrics


if __name__ == "__main__":
    main()
