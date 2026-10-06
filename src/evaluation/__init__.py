"""
Evaluation module for computing classification metrics, confusion matrices, and test reports.
"""
from .evaluate import evaluate_model
from .metrics import compute_classification_metrics

__all__ = ['evaluate_model', 'compute_classification_metrics']
