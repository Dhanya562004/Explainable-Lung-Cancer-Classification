import os
import json
import hashlib
from src.config import MODEL_REGISTRY_PATH

def load_model_registry(registry_path=MODEL_REGISTRY_PATH):
    """
    Loads model registry JSON metadata.
    """
    if not os.path.exists(registry_path):
        return {
            "active_version": "v1.0.0-xception",
            "registry": {}
        }
    try:
        with open(registry_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {"active_version": "v1.0.0-xception", "registry": {}}

def save_model_registry(registry_data, registry_path=MODEL_REGISTRY_PATH):
    """
    Saves updated model registry JSON.
    """
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, 'w') as f:
        json.dump(registry_data, f, indent=4)

def calculate_file_hash(file_path):
    """
    Computes SHA256 checksum of model binary file.
    """
    if not os.path.exists(file_path):
        return "sha256:not_found"
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()[:32]}"

def get_active_model_info(registry_path=MODEL_REGISTRY_PATH):
    """
    Gets details of currently active production model version.
    """
    reg = load_model_registry(registry_path)
    active_ver = reg.get('active_version', 'v1.0.0-xception')
    version_info = reg.get('registry', {}).get(active_ver, {
        'version': active_ver,
        'architecture': 'Xception Transfer Learning',
        'metrics': {'test_accuracy': 0.7651, 'macro_f1_score': 0.7859}
    })
    return version_info

def register_model(version, architecture, metrics, training_config, model_path, set_active=True):
    """
    Registers a new model version entry in model_registry.json.
    """
    reg = load_model_registry()
    checksum = calculate_file_hash(model_path)
    
    entry = {
        'version': version,
        'architecture': architecture,
        'created_at': os.getenv('DATE', '2026-10-06'),
        'metrics': metrics,
        'training_config': training_config,
        'checksum': checksum
    }
    
    reg['registry'][version] = entry
    if set_active:
        reg['active_version'] = version

    save_model_registry(reg)
    return entry
