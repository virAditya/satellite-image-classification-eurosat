"""
Visualization Utilities
Confusion matrices, training curves, etc.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from .data_loader import EuroSATConfig

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_confusion_matrix(y_true, y_pred, class_names=EuroSATConfig.CLASS_NAMES, save_path=None):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    plt.show()

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training history.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_title('Loss Curves')
    ax1.legend()
    
    ax2.plot(train_accs, label='Train Acc', marker='o')
    ax2.plot(val_accs, label='Val Acc', marker='s')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    plt.tight_layout()
    plt.show()

def plot_per_class_metrics(per_class_metrics, save_path=None):
    """
    Bar plot for per-class F1-scores.
    """
    classes = list(per_class_metrics.keys())
    f1_scores = [per_class_metrics[cls]['f1'] for cls in classes]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(classes, f1_scores, color='skyblue')
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('Per-Class F1-Scores')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
