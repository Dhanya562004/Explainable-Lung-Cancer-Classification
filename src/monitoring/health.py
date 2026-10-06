import os
from src.config import (
    DEFAULT_MODEL_PATH, DEFAULT_DATASET_DIR, MONITORING_DIR, LOGS_DIR
)
from src.models.model_registry import get_active_model_info

def check_system_health():
    """
    Performs comprehensive system health checks:
    - Model availability and active version
    - Dataset directory accessibility
    - Monitoring & log storage write permissions
    - System CPU & Memory usage (with fallback if psutil is unavailable)
    """
    model_exists = os.path.exists(DEFAULT_MODEL_PATH)
    active_info = get_active_model_info()
    dataset_exists = os.path.exists(DEFAULT_DATASET_DIR)
    
    monitoring_writable = os.access(os.path.dirname(MONITORING_DIR), os.W_OK) if os.path.exists(os.path.dirname(MONITORING_DIR)) else True
    logs_writable = os.access(os.path.dirname(LOGS_DIR), os.W_OK) if os.path.exists(os.path.dirname(LOGS_DIR)) else True

    cpu_usage_pct = 0.0
    memory_usage_pct = 0.0
    try:
        import psutil
        cpu_usage_pct = psutil.cpu_percent(interval=None)
        mem_info = psutil.virtual_memory()
        memory_usage_pct = mem_info.percent
    except (ImportError, Exception):
        pass

    checks = {
        'model_status': 'AVAILABLE' if model_exists else 'FALLBACK_DEMO_MODE',
        'active_model_version': active_info.get('version', 'v1.0.0-xception'),
        'dataset_accessible': dataset_exists,
        'monitoring_storage_ready': monitoring_writable,
        'logs_storage_ready': logs_writable,
        'cpu_usage_percent': cpu_usage_pct,
        'memory_usage_percent': memory_usage_pct
    }

    all_ok = checks['dataset_accessible'] and checks['monitoring_storage_ready']
    status = "HEALTHY" if all_ok else "DEGRADED"

    return {
        'status': status,
        'checks': checks
    }

if __name__ == '__main__':
    print(check_system_health())
