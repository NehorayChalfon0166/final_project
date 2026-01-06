"""
K-Fold Cross-Validation
=======================
Stratified K-fold CV for GNN wallet classification.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import List, Dict, Optional, Callable
import logging
from tqdm import tqdm
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .metrics import compute_metrics, MetricsReport, aggregate_cv_metrics, print_cv_summary

logger = logging.getLogger(__name__)


@dataclass
class CVConfig:
    """Configuration for cross-validation."""
    n_folds: int = 5
    epochs: int = 100
    batch_size: int = 64
    lr: float = 0.001
    max_lr: float = 0.005
    weight_decay: float = 0.01
    patience: int = 15
    random_state: int = 42


class KFoldCV:
    """
    K-Fold Cross-Validation for GNN models.

    Stratified by label and source to ensure balanced representation.
    """

    def __init__(
        self,
        model_class: type,
        model_kwargs: Dict,
        config: Optional[CVConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize cross-validator.

        Args:
            model_class: Model class to instantiate for each fold
            model_kwargs: Keyword arguments for model instantiation
            config: CV configuration
            device: Device to use (auto-detected if None)
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.config = config or CVConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.fold_results: List[MetricsReport] = []
        self.fold_models: List[nn.Module] = []

    def _get_stratify_key(self, graphs: List[Data]) -> np.ndarray:
        """
        Create stratification key combining label and source.

        Args:
            graphs: List of PyG Data objects

        Returns:
            Array of stratification keys
        """
        keys = []
        for g in graphs:
            label = g.y[0].item()  # Center node label
            source = getattr(g, 'source', 'unknown')
            keys.append(f"{label}_{source}")
        return np.array(keys)

    def _get_labels(self, graphs: List[Data]) -> np.ndarray:
        """Extract labels from graphs."""
        return np.array([g.y[0].item() for g in graphs])

    def _train_fold(
        self,
        train_graphs: List[Data],
        val_graphs: List[Data],
        fold_idx: int
    ) -> tuple:
        """
        Train model on a single fold.

        Args:
            train_graphs: Training graphs
            val_graphs: Validation graphs
            fold_idx: Fold index

        Returns:
            (model, metrics, history)
        """
        # Create dataloaders
        train_loader = DataLoader(
            train_graphs,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_graphs,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        # Initialize model
        model = self.model_class(**self.model_kwargs).to(self.device)

        # Optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.max_lr,
            epochs=self.config.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )

        # Loss function
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Training loop with early stopping
        best_f1 = 0
        patience_counter = 0
        best_state = None
        history = {'train_loss': [], 'val_f1': []}

        for epoch in range(self.config.epochs):
            # Train
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                labels = self._get_batch_labels(batch)

                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # Validate
            val_metrics = self._evaluate(model, val_loader)
            history['val_f1'].append(val_metrics.f1)

            # Early stopping
            if val_metrics.f1 > best_f1:
                best_f1 = val_metrics.f1
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Fold {fold_idx}: Early stopping at epoch {epoch}")
                    break

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation
        final_metrics = self._evaluate(model, val_loader)

        return model, final_metrics, history

    def _get_batch_labels(self, batch) -> torch.Tensor:
        """Extract center node labels from batch."""
        if hasattr(batch, 'ptr'):
            center_indices = batch.ptr[:-1]
        else:
            batch_size = batch.batch.max().item() + 1
            center_indices = []
            for i in range(batch_size):
                mask = (batch.batch == i)
                first_idx = mask.nonzero(as_tuple=True)[0][0]
                center_indices.append(first_idx)
            center_indices = torch.tensor(center_indices, device=batch.y.device)

        return batch.y[center_indices]

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, loader: DataLoader) -> MetricsReport:
        """Evaluate model on a dataloader."""
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        for batch in loader:
            batch = batch.to(self.device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            labels = self._get_batch_labels(batch)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        return compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )

    def run(self, graphs: List[Data]) -> Dict:
        """
        Run K-fold cross-validation.

        Args:
            graphs: List of all graph data objects

        Returns:
            Dictionary with fold results and aggregated metrics
        """
        logger.info(f"Starting {self.config.n_folds}-fold cross-validation")
        logger.info(f"Total graphs: {len(graphs)}")

        # Get stratification key
        stratify_key = self._get_stratify_key(graphs)
        labels = self._get_labels(graphs)

        # Create folds
        skf = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )

        self.fold_results = []
        self.fold_models = []
        all_histories = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(graphs, stratify_key)):
            logger.info(f"\n{'='*50}")
            logger.info(f"Fold {fold_idx + 1}/{self.config.n_folds}")
            logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

            train_graphs = [graphs[i] for i in train_idx]
            val_graphs = [graphs[i] for i in val_idx]

            # Train
            model, metrics, history = self._train_fold(train_graphs, val_graphs, fold_idx)

            self.fold_results.append(metrics)
            self.fold_models.append(model)
            all_histories.append(history)

            logger.info(f"Fold {fold_idx + 1} Results:")
            logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
            logger.info(f"  F1 Score: {metrics.f1:.4f}")
            logger.info(f"  ROC-AUC:  {metrics.roc_auc:.4f}")

        # Aggregate results
        aggregated = aggregate_cv_metrics(self.fold_results)

        logger.info("\n" + print_cv_summary(aggregated))

        return {
            'fold_metrics': [m.to_dict() for m in self.fold_results],
            'aggregated': aggregated,
            'histories': all_histories,
            'config': self.config.__dict__
        }

    def get_best_model(self) -> nn.Module:
        """
        Get the best model based on F1 score.

        Returns:
            Best performing model
        """
        if not self.fold_results:
            raise ValueError("No CV results. Run cross-validation first.")

        best_idx = np.argmax([m.f1 for m in self.fold_results])
        return self.fold_models[best_idx]


def run_cv_from_dataset(
    model_class: type,
    model_kwargs: Dict,
    dataset,
    config: Optional[CVConfig] = None
) -> Dict:
    """
    Convenience function to run CV from a PyG Dataset.

    Args:
        model_class: Model class
        model_kwargs: Model arguments
        dataset: PyG Dataset
        config: CV configuration

    Returns:
        CV results dictionary
    """
    graphs = [dataset[i] for i in range(len(dataset))]

    cv = KFoldCV(
        model_class=model_class,
        model_kwargs=model_kwargs,
        config=config
    )

    return cv.run(graphs)
