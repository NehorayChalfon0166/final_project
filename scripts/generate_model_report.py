"""
Generate model evaluation report with confusion matrix and score distribution plots.
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.dataloader import get_test_loader
from src.graph.config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES
from src.models.optimal_gnn import OptimalBitcoinGNN
from src.models.utils import get_center_labels
from src.evaluation.metrics import compute_metrics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
REPORT_DIR = os.path.join(OUTPUTS_DIR, 'evaluation')
os.makedirs(REPORT_DIR, exist_ok=True)


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(OUTPUTS_DIR, 'gnn_model.pt')

    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        hidden_dim=64
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load temperature
    temperature = 1.0
    temp_path = os.path.join(OUTPUTS_DIR, 'temperature.pt')
    if os.path.exists(temp_path):
        temp_data = torch.load(temp_path, map_location=device, weights_only=True)
        temperature = temp_data['temperature']

    return model, device, temperature


def run_inference(model, device, temperature):
    test_loader = get_test_loader(batch_size=64)
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            logits = out / temperature
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            labels = get_center_labels(batch)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(report, save_path):
    cm = np.array([[report.tn, report.fp],
                   [report.fn, report.tp]])
    cm_pct = cm / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, cmap='Blues', aspect='equal')

    class_labels = ['Benign (0)', 'Criminal (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels, fontsize=12)
    ax.set_yticklabels(class_labels, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)',
                    ha='center', va='center', fontsize=14, fontweight='bold', color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_score_distribution(labels, probs, save_path):
    benign_scores = probs[labels == 0]
    criminal_scores = probs[labels == 1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1.5]})

    # Top: overlapping histograms
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    ax.hist(benign_scores, bins=bins, alpha=0.6, color='#2196F3', label=f'Benign (n={len(benign_scores):,})', edgecolor='white', linewidth=0.5)
    ax.hist(criminal_scores, bins=bins, alpha=0.6, color='#F44336', label=f'Criminal (n={len(criminal_scores):,})', edgecolor='white', linewidth=0.5)
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Decision threshold (0.5)')
    ax.set_xlabel('Criminal Probability Score', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('Distribution of Criminal Probability Scores on Test Set', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)

    # Bottom: strip/swarm-style scatter
    ax2 = axes[1]
    np.random.seed(42)
    jitter_b = np.random.uniform(-0.15, 0.15, size=len(benign_scores))
    jitter_c = np.random.uniform(-0.15, 0.15, size=len(criminal_scores))

    # Subsample if too many points
    max_points = 2000
    if len(benign_scores) > max_points:
        idx = np.random.choice(len(benign_scores), max_points, replace=False)
        plot_benign = benign_scores[idx]
        jitter_b = jitter_b[idx]
    else:
        plot_benign = benign_scores

    if len(criminal_scores) > max_points:
        idx = np.random.choice(len(criminal_scores), max_points, replace=False)
        plot_criminal = criminal_scores[idx]
        jitter_c = jitter_c[idx]
    else:
        plot_criminal = criminal_scores

    ax2.scatter(plot_benign, 0 + jitter_b, alpha=0.3, s=8, color='#2196F3', label='Benign')
    ax2.scatter(plot_criminal, 1 + jitter_c, alpha=0.3, s=8, color='#F44336', label='Criminal')
    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Benign', 'Criminal'], fontsize=12)
    ax2.set_xlabel('Criminal Probability Score', fontsize=13)
    ax2.set_ylabel('True Class', fontsize=13)
    ax2.set_xlim(0, 1)
    ax2.set_title('Per-Sample Score Distribution', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_and_pr(report, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    if report.roc_curve is not None:
        fpr, tpr, _ = report.roc_curve
        axes[0].plot(fpr, tpr, color='#1976D2', linewidth=2, label=f'ROC (AUC = {report.roc_auc:.4f})')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        axes[0].set_xlabel('False Positive Rate', fontsize=13)
        axes[0].set_ylabel('True Positive Rate', fontsize=13)
        axes[0].set_title('ROC Curve', fontsize=15, fontweight='bold')
        axes[0].legend(fontsize=12)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1.02)
        axes[0].grid(alpha=0.3)

    # PR curve
    if report.pr_curve is not None:
        precision_c, recall_c, _ = report.pr_curve
        axes[1].plot(recall_c, precision_c, color='#E64A19', linewidth=2, label=f'PR (AUC = {report.pr_auc:.4f})')
        baseline = report.n_positive / report.n_samples
        axes[1].axhline(y=baseline, color='k', linestyle='--', linewidth=1, alpha=0.5, label=f'Baseline ({baseline:.2f})')
        axes[1].set_xlabel('Recall', fontsize=13)
        axes[1].set_ylabel('Precision', fontsize=13)
        axes[1].set_title('Precision-Recall Curve', fontsize=15, fontweight='bold')
        axes[1].legend(fontsize=12)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1.02)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def save_statistics(report, temperature, save_path):
    stats = {
        'model': 'OptimalBitcoinGNN (3-layer GATv2)',
        'temperature_scaling': temperature,
        'test_set_size': report.n_samples,
        'metrics': {
            'accuracy': round(report.accuracy, 4),
            'precision': round(report.precision, 4),
            'recall': round(report.recall, 4),
            'f1_score': round(report.f1, 4),
            'roc_auc': round(report.roc_auc, 4),
            'pr_auc': round(report.pr_auc, 4),
        },
        'confusion_matrix': {
            'true_negatives': int(report.tn),
            'false_positives': int(report.fp),
            'false_negatives': int(report.fn),
            'true_positives': int(report.tp),
        },
        'class_distribution': {
            'benign': int(report.n_negative),
            'criminal': int(report.n_positive),
            'benign_pct': round(report.n_negative / report.n_samples * 100, 1),
            'criminal_pct': round(report.n_positive / report.n_samples * 100, 1),
        },
        'per_class_metrics': {
            'benign': {
                'precision': round(report.precision_per_class.get(0, 0), 4),
                'recall': round(report.recall_per_class.get(0, 0), 4),
                'f1': round(report.f1_per_class.get(0, 0), 4),
            },
            'criminal': {
                'precision': round(report.precision_per_class.get(1, 0), 4),
                'recall': round(report.recall_per_class.get(1, 0), 4),
                'f1': round(report.f1_per_class.get(1, 0), 4),
            }
        },
        'error_rates': {
            'false_positive_rate': round(report.fp / (report.fp + report.tn), 4) if (report.fp + report.tn) > 0 else 0,
            'false_negative_rate': round(report.fn / (report.fn + report.tp), 4) if (report.fn + report.tp) > 0 else 0,
        }
    }

    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {save_path}")

    return stats


def print_report(stats):
    m = stats['metrics']
    cm = stats['confusion_matrix']
    cd = stats['class_distribution']
    pc = stats['per_class_metrics']
    er = stats['error_rates']

    print(f"""
{'='*60}
  GNN MODEL EVALUATION REPORT
{'='*60}
  Model:        {stats['model']}
  Temperature:  {stats['temperature_scaling']:.3f}
  Test samples: {stats['test_set_size']:,}
{'='*60}

  OVERALL METRICS
  ---------------
  Accuracy:     {m['accuracy']*100:.2f}%
  Precision:    {m['precision']*100:.2f}%
  Recall:       {m['recall']*100:.2f}%
  F1 Score:     {m['f1_score']*100:.2f}%
  ROC-AUC:      {m['roc_auc']*100:.2f}%
  PR-AUC:       {m['pr_auc']*100:.2f}%

  CONFUSION MATRIX
  ----------------
                    Predicted
                 Benign    Criminal
  Actual Benign   {cm['true_negatives']:<10}{cm['false_positives']:<10}
  Actual Criminal {cm['false_negatives']:<10}{cm['true_positives']:<10}

  PER-CLASS METRICS
  -----------------
  Benign:   P={pc['benign']['precision']:.4f}  R={pc['benign']['recall']:.4f}  F1={pc['benign']['f1']:.4f}
  Criminal: P={pc['criminal']['precision']:.4f}  R={pc['criminal']['recall']:.4f}  F1={pc['criminal']['f1']:.4f}

  ERROR RATES
  -----------
  False Positive Rate: {er['false_positive_rate']*100:.2f}%
  False Negative Rate: {er['false_negative_rate']*100:.2f}%

  CLASS DISTRIBUTION
  ------------------
  Benign:   {cd['benign']:,} ({cd['benign_pct']}%)
  Criminal: {cd['criminal']:,} ({cd['criminal_pct']}%)
{'='*60}
""")


def main():
    print("\n[1/5] Loading model...")
    model, device, temperature = load_model()

    print("[2/5] Running inference on test set...")
    labels, preds, probs = run_inference(model, device, temperature)

    print("[3/5] Computing metrics...")
    report = compute_metrics(labels, preds, probs)

    print("[4/5] Saving statistics...")
    stats = save_statistics(report, temperature, os.path.join(REPORT_DIR, 'model_statistics.json'))
    print_report(stats)

    print("[5/5] Generating plots...")
    plot_confusion_matrix(report, os.path.join(REPORT_DIR, 'confusion_matrix.png'))
    plot_score_distribution(labels, probs, os.path.join(REPORT_DIR, 'score_distribution.png'))
    plot_roc_and_pr(report, os.path.join(REPORT_DIR, 'roc_pr_curves.png'))

    print("\nDone! All outputs saved to outputs/evaluation/")


if __name__ == '__main__':
    main()
