import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

from utils import (
    CLASS_NAMES, CLASS_LABELS, DEFAULT_MODEL_PATH,
    DEFAULT_RESULTS_DIR, get_data_generators
)

def evaluate_model(model_path=DEFAULT_MODEL_PATH, results_dir=DEFAULT_RESULTS_DIR):
    """
    Evaluates trained model on unseen test dataset split.
    Generates classification metrics, JSON report, and confusion matrix plot.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

    os.makedirs(results_dir, exist_ok=True)

    print(f"[INFO] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("[INFO] Fetching test generator...")
    _, _, test_gen = get_data_generators()

    test_gen.reset()
    y_true = test_gen.classes
    num_samples = len(y_true)

    print(f"[INFO] Running evaluation on {num_samples} test CT images...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    display_class_names = [CLASS_LABELS[i] for i in range(len(CLASS_NAMES))]
    report_dict = classification_report(y_true, y_pred, target_names=display_class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=display_class_names)

    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*50)
    print("EVALUATION RESULTS ON TEST SET")
    print("="*50)
    print(f"Test Accuracy:         {acc * 100:.2f}%")
    print(f"Macro Precision:       {p_macro:.4f}")
    print(f"Macro Recall:          {r_macro:.4f}")
    print(f"Macro F1-Score:        {f1_macro:.4f}")
    print(f"Weighted F1-Score:     {f1_weighted:.4f}")
    print("\nClassification Report:\n", report_str)

    # Save Confusion Matrix Plot
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=display_class_names,
        yticklabels=display_class_names,
        cbar=True
    )
    plt.title('Test Dataset Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=20, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix heatmap to: {cm_path}")

    # Save Metrics JSON
    metrics_summary = {
        'test_accuracy': acc,
        'macro_precision': float(p_macro),
        'macro_recall': float(r_macro),
        'macro_f1_score': float(f1_macro),
        'weighted_f1_score': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict
    }

    json_path = os.path.join(results_dir, 'evaluation_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[INFO] Saved evaluation metrics to: {json_path}")

    return metrics_summary

if __name__ == '__main__':
    evaluate_model()
