import numpy as np
from scipy.stats import ks_2samp

def compute_image_stats(preprocessed_tensor):
    """
    Extracts statistical features from preprocessed image tensor (1, 299, 299, 3).
    Computes mean intensity, standard deviation, and pixel intensity percentiles.
    """
    arr = np.array(preprocessed_tensor).flatten()
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'p25': float(np.percentile(arr, 25)),
        'median': float(np.median(arr)),
        'p75': float(np.percentile(arr, 75))
    }

def detect_data_drift(current_img_stats, reference_baseline=None):
    """
    Lightweight distribution drift detector comparing inference image statistics against a baseline reference.

    Baseline parameters (default Xception preprocessed intensity range [-1.0, 1.0]):
      - reference_mean: ~0.0
      - reference_std: ~0.55

    Returns:
        dict: drift_status ('LOW DRIFT', 'MEDIUM DRIFT', 'HIGH DRIFT'),
              z_score_diff, distance_metric, message
    """
    if reference_baseline is None:
        reference_baseline = {
            'mean': 0.021,
            'std': 0.548
        }

    mean_diff = abs(current_img_stats['mean'] - reference_baseline['mean'])
    std_diff = abs(current_img_stats['std'] - reference_baseline['std'])

    # Standardized distance score
    drift_score = (mean_diff / (reference_baseline['std'] + 1e-6)) + (std_diff / (reference_baseline['std'] + 1e-6))

    if drift_score < 0.35:
        drift_status = "LOW DRIFT"
        message = "Input data distribution matches baseline training profile."
    elif drift_score < 0.85:
        drift_status = "MEDIUM DRIFT"
        message = "Moderate variation in CT image brightness/contrast observed."
    else:
        drift_status = "HIGH DRIFT"
        message = "Significant statistical drift detected in image intensity profile."

    return {
        'drift_status': drift_status,
        'drift_score': round(float(drift_score), 4),
        'mean_diff': round(float(mean_diff), 4),
        'std_diff': round(float(std_diff), 4),
        'current_stats': current_img_stats,
        'reference_baseline': reference_baseline,
        'message': message,
        'disclaimer': "Data drift metric is an engineering telemetry signal for input distribution monitoring, not a clinical diagnostic assessment."
    }
