import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
)
from src.config import CLASS_LABELS, CLASS_NAMES

def compute_classification_metrics(y_true, y_pred):
    """
    Computes accuracy, macro precision/recall/F1, weighted F1, confusion matrix,
    and detailed per-class report dictionary.
    Handles partial class sets by explicitly defining class label indices.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    class_indices = list(range(len(CLASS_NAMES)))
    display_class_names = [CLASS_LABELS[i] for i in class_indices]

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=class_indices,
        target_names=display_class_names,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=class_indices)

    return {
        'test_accuracy': acc,
        'macro_precision': float(p_macro),
        'macro_recall': float(r_macro),
        'macro_f1_score': float(f1_macro),
        'weighted_f1_score': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict
    }
