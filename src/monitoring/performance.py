import os
import json
from datetime import datetime
from src.config import FEEDBACK_LOG_PATH, CLASS_NAMES, CLASS_LABELS
from src.evaluation.metrics import compute_classification_metrics

def log_ground_truth_feedback(prediction_id, verified_class, predicted_class=None):
    """
    Records verified ground truth feedback from a clinical researcher or validator.
    """
    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
    payload = {
        'prediction_id': prediction_id,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'verified_class': verified_class,
        'predicted_class': predicted_class
    }
    try:
        with open(FEEDBACK_LOG_PATH, 'a') as f:
            f.write(json.dumps(payload) + '\n')
        print(f"[FEEDBACK LOGGED] Saved verified label for prediction: {prediction_id}")
        return True, "Feedback recorded successfully."
    except Exception as e:
        return False, f"Failed to log feedback: {str(e)}"

def get_verified_performance_metrics(log_path=FEEDBACK_LOG_PATH):
    """
    Calculates verified performance metrics (accuracy, precision, recall, F1)
    on predictions where ground truth feedback has been provided.

    Always includes notice:
    "Performance metrics are calculated only for predictions with verified ground truth."
    """
    notice = "Performance metrics are calculated only for predictions with verified ground truth."

    if not os.path.exists(log_path):
        return {
            'has_verified_data': False,
            'verified_count': 0,
            'notice': notice,
            'metrics': None
        }

    records = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
    except Exception:
        pass

    if not records:
        return {
            'has_verified_data': False,
            'verified_count': 0,
            'notice': notice,
            'metrics': None
        }

    label_to_idx = {name: i for i, name in CLASS_LABELS.items()}
    # map lowercase versions as well
    for i, name in CLASS_LABELS.items():
        label_to_idx[name.lower()] = i
        label_to_idx[CLASS_NAMES[i]] = i

    y_true = []
    y_pred = []

    for r in records:
        v_cls = r.get('verified_class')
        p_cls = r.get('predicted_class')
        if v_cls in label_to_idx and p_cls in label_to_idx:
            y_true.append(label_to_idx[v_cls])
            y_pred.append(label_to_idx[p_cls])

    if not y_true:
        return {
            'has_verified_data': False,
            'verified_count': len(records),
            'notice': notice,
            'metrics': None
        }

    metrics = compute_classification_metrics(y_true, y_pred)
    return {
        'has_verified_data': True,
        'verified_count': len(y_true),
        'notice': notice,
        'metrics': metrics
    }
