import os

# Project Root Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Directory Paths
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, 'models')
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, 'best_model.keras')
MODEL_REGISTRY_PATH = os.path.join(DEFAULT_MODEL_DIR, 'model_registry.json')
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MONITORING_DIR = os.path.join(DEFAULT_RESULTS_DIR, 'monitoring')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
AUDIT_LOG_PATH = os.path.join(LOGS_DIR, 'audit_log.jsonl')
INFERENCE_LOG_PATH = os.path.join(MONITORING_DIR, 'inference_log.jsonl')
FEEDBACK_LOG_PATH = os.path.join(MONITORING_DIR, 'feedback_log.jsonl')

# Model Parameters
IMAGE_SIZE = (299, 299)
INPUT_SHAPE = (299, 299, 3)
BATCH_SIZE = 16
NUM_CLASSES = 4

# Standardized Class Mappings
CLASS_LABELS = {
    0: 'Adenocarcinoma',
    1: 'Large Cell Carcinoma',
    2: 'Normal',
    3: 'Squamous Cell Carcinoma'
}

CLASS_NAMES = ['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma']

# Confidence Thresholds
CONFIDENCE_THRESHOLDS = {
    'HIGH': 80.0,
    'MEDIUM': 60.0,
}

# Retraining & Promotion Criteria
PROMOTION_CRITERIA = {
    'minimum_macro_f1': 0.75,
    'minimum_accuracy': 0.75
}

# Medical Disclaimer
MEDICAL_DISCLAIMER = (
    "⚠️ DISCLAIMER: For educational and research purposes only. "
    "This application is not a medical diagnostic tool and should not be used "
    "for clinical decision-making or patient diagnosis."
)
