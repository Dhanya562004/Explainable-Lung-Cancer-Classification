import os
import json
from datetime import datetime
from src.config import DEFAULT_RESULTS_DIR

def run_hyperparameter_experiment(
    learning_rate_stage1=1e-3,
    learning_rate_stage2=1e-5,
    dropout_rate=0.4,
    batch_size=16,
    fine_tune_depth=30,
    optimizer="Adam",
    experiment_name="experiment_default"
):
    """
    Hyperparameter configuration tracking system.
    Records experiment parameters and metadata to results/experiment_logs.json.
    """
    log_file = os.path.join(DEFAULT_RESULTS_DIR, 'experiment_logs.json')
    
    experiment_entry = {
        'experiment_id': f"exp-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'experiment_name': experiment_name,
        'architecture': 'Xception Transfer Learning',
        'hyperparameters': {
            'learning_rate_stage1': learning_rate_stage1,
            'learning_rate_stage2': learning_rate_stage2,
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'fine_tune_depth': fine_tune_depth,
            'optimizer': optimizer
        },
        'status': 'CONFIGURED'
    }

    experiments = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                experiments = json.load(f)
        except Exception:
            experiments = []

    experiments.append(experiment_entry)
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(experiments, f, indent=4)

    print(f"[INFO] Logged hyperparameter experiment: {experiment_entry['experiment_id']}")
    return experiment_entry

if __name__ == '__main__':
    run_hyperparameter_experiment()
