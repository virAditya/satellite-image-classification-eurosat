"""
Evaluation Metrics for EuroSAT
Accuracy, F1, Kappa, confusion analysis.
"""

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef, precision_recall_fscore_support
)
import numpy as np
from .data_loader import EuroSATConfig

def compute_metrics(y_true, y_pred, class_names=EuroSATConfig.CLASS_NAMES):
    """
    Compute comprehensive metrics.
    
    Args:
        y_true, y_pred: Numpy arrays of labels/preds
        class_names: List of class names
    
    Returns:
        dict: Metrics (accuracy, f1_macro, kappa, etc.)
    """
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    precision_macro *= 100
    recall_macro *= 100
    f1_macro *= 100
    
    kappa = cohen_kappa_score(y_true, y_pred) * 100
    matthews = matthews_corrcoef(y_true, y_pred) * 100
    
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    
    per_class = {}
    for cls in class_names:
        per_class[cls] = {
            'acc': report[cls]['support'] > 0 and (report[cls]['support'] * report[cls]['f1-score']) / 100 or 0,
            'precision': report[cls]['precision'] * 100,
            'recall': report[cls]['recall'] * 100,
            'f1': report[cls]['f1-score'] * 100
        }
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'kappa': kappa,
        'matthews': matthews,
        'per_class': per_class,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

def confusion_analysis(cm, class_names=EuroSATConfig.CLASS_NAMES):
    """
    Analyze top misclassifications.
    """
    misclass = cm.copy()
    np.fill_diagonal(misclass, 0)
    
    top_errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and misclass[i, j] > 0:
                top_errors.append((misclass[i, j], class_names[i], class_names[j]))
    
    top_errors.sort(reverse=True)
    return top_errors[:10]

def confidence_analysis(outputs, y_true, y_pred):
    """
    Analyze prediction confidences.
    """
    probs = torch.softmax(outputs, dim=1)
    confidences = torch.max(probs, dim=1)[0].cpu().numpy()
    
    correct_mask = (y_pred == y_true)
    correct_conf = np.mean(confidences[correct_mask]) * 100
    incorrect_conf = np.mean(confidences[~correct_mask]) * 100
    
    return {'correct_avg': correct_conf, 'incorrect_avg': incorrect_conf, 'gap': correct_conf - incorrect_conf}
