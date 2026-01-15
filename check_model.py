"""
Model Evaluation Script
=======================
Evaluates the trained GNN model on all data and prints comprehensive stats.
"""
import os
import sys
import torch
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.optimal_gnn import OptimalBitcoinGNN
from graph.dataloader import EgoGraphDataset
from torch_geometric.loader import DataLoader
from evaluation.metrics import compute_metrics


def load_model(model_path: str, device: torch.device) -> OptimalBitcoinGNN:
    """Load the trained model."""
    model = OptimalBitcoinGNN(
        num_node_features=12,
        num_edge_features=3,
        hidden_dim=64,
        num_heads_1=4,
        num_heads_2=4,
        num_heads_3=2,
        dropout=0.2,
        final_dropout=0.3
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def evaluate_on_split(model, dataset, device, batch_size=32):
    """Evaluate model on a dataset split."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            # Get center node labels (first node of each graph)
            labels = []
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                labels.append(batch.y[mask][0].item())

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = os.path.join(os.path.dirname(__file__), 'outputs', 'gnn_model.pt')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        sys.exit(1)

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print_header("MODEL ARCHITECTURE")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size:           {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # Results storage
    all_results = {}
    combined_labels = []
    combined_preds = []
    combined_probs = []

    # Evaluate on each split
    for split in ['train', 'test']:
        print_header(f"{split.upper()} SET EVALUATION")

        try:
            dataset = EgoGraphDataset(split=split)
            print(f"Loaded {len(dataset)} graphs")

            if len(dataset) == 0:
                print(f"No graphs found for {split} split")
                continue

            labels, preds, probs = evaluate_on_split(model, dataset, device)

            # Compute metrics
            metrics = compute_metrics(labels, preds, probs)
            all_results[split] = metrics

            # Add to combined
            combined_labels.extend(labels)
            combined_preds.extend(preds)
            combined_probs.extend(probs)

            # Print metrics
            print(f"\nSamples:    {len(labels):,}")
            print(f"  Benign:   {np.sum(labels == 0):,} ({np.mean(labels == 0)*100:.1f}%)")
            print(f"  Criminal: {np.sum(labels == 1):,} ({np.mean(labels == 1)*100:.1f}%)")
            print(f"\nPerformance Metrics:")
            print(f"  Accuracy:  {metrics.accuracy*100:.2f}%")
            print(f"  Precision: {metrics.precision*100:.2f}%")
            print(f"  Recall:    {metrics.recall*100:.2f}%")
            print(f"  F1 Score:  {metrics.f1*100:.2f}%")
            print(f"  ROC-AUC:   {metrics.roc_auc:.4f}")
            print(f"  PR-AUC:    {metrics.pr_auc:.4f}")

            print(f"\nConfusion Matrix:")
            print(f"                  Predicted")
            print(f"               Benign  Criminal")
            print(f"  Actual Benign   {metrics.tn:<8} {metrics.fp:<8}")
            print(f"  Actual Criminal {metrics.fn:<8} {metrics.tp:<8}")

        except Exception as e:
            print(f"Error evaluating {split}: {e}")
            import traceback
            traceback.print_exc()

    # Combined results
    if len(combined_labels) > 0:
        print_header("COMBINED (ALL DATA)")
        combined_labels = np.array(combined_labels)
        combined_preds = np.array(combined_preds)
        combined_probs = np.array(combined_probs)

        metrics = compute_metrics(combined_labels, combined_preds, combined_probs)

        print(f"\nTotal Samples: {len(combined_labels):,}")
        print(f"  Benign:      {np.sum(combined_labels == 0):,} ({np.mean(combined_labels == 0)*100:.1f}%)")
        print(f"  Criminal:    {np.sum(combined_labels == 1):,} ({np.mean(combined_labels == 1)*100:.1f}%)")
        print(f"\nOverall Performance:")
        print(f"  Accuracy:  {metrics.accuracy*100:.2f}%")
        print(f"  Precision: {metrics.precision*100:.2f}%")
        print(f"  Recall:    {metrics.recall*100:.2f}%")
        print(f"  F1 Score:  {metrics.f1*100:.2f}%")
        print(f"  ROC-AUC:   {metrics.roc_auc:.4f}")
        print(f"  PR-AUC:    {metrics.pr_auc:.4f}")

        print(f"\nConfusion Matrix (Total):")
        print(f"                  Predicted")
        print(f"               Benign  Criminal")
        print(f"  Actual Benign   {metrics.tn:<8} {metrics.fp:<8}")
        print(f"  Actual Criminal {metrics.fn:<8} {metrics.tp:<8}")

        # Error analysis
        print_header("ERROR ANALYSIS")
        false_positives = np.sum((combined_labels == 0) & (combined_preds == 1))
        false_negatives = np.sum((combined_labels == 1) & (combined_preds == 0))

        print(f"False Positives (Benign predicted as Criminal): {false_positives}")
        print(f"False Negatives (Criminal predicted as Benign): {false_negatives}")

        # Confidence analysis
        correct_mask = combined_labels == combined_preds
        incorrect_mask = ~correct_mask

        if np.sum(correct_mask) > 0:
            correct_confidence = combined_probs[correct_mask & (combined_labels == 1)]
            if len(correct_confidence) > 0:
                print(f"\nCorrect Criminal Predictions:")
                print(f"  Mean confidence: {np.mean(correct_confidence):.4f}")
                print(f"  Min confidence:  {np.min(correct_confidence):.4f}")

        if np.sum(incorrect_mask) > 0:
            incorrect_probs_vals = combined_probs[incorrect_mask]
            print(f"\nIncorrect Predictions:")
            print(f"  Mean confidence: {np.mean(incorrect_probs_vals):.4f}")


if __name__ == '__main__':
    main()
