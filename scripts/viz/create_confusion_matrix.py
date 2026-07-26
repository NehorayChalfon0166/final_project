import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'


def create_confusion_matrix():
    # Confusion matrix from test set evaluation
    # [[TN, FP], [FN, TP]]
    cm = np.array([[821, 179],
                   [199, 801]])

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Criminal'],
                yticklabels=['Benign', 'Criminal'],
                cbar_kws={'label': 'Count'})

    # Add annotations with count and percentage
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            percent = cm_percent[i, j]
            ax.text(j + 0.5, i + 0.5, f'{count}\n({percent:.1f}%)',
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color='white' if count > 500 else 'black')

    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')

    # Add metrics text
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    metrics_text = f'Accuracy: {accuracy:.1%}  |  Precision: {precision:.1%}  |  Recall: {recall:.1%}  |  F1: {f1:.1%}'
    plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = OUTPUTS_DIR / 'confusion_matrix.png'
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Confusion matrix saved to {out}")

if __name__ == '__main__':
    create_confusion_matrix()
