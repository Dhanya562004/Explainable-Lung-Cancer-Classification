import pytest
from PIL import Image
from src.models.inference import predict_ct_scan

def test_prediction_output_schema():
    img = Image.new('RGB', (299, 299), color='blue')
    res = predict_ct_scan(img)

    assert 'prediction_id' in res
    assert 'predicted_class' in res
    assert 'confidence' in res
    assert 'confidence_level' in res
    assert res['confidence_level'] in ['HIGH', 'MEDIUM', 'LOW']
    assert 'probabilities' in res
    assert len(res['probabilities']) == 4
    assert 'model_version' in res
    assert 'inference_latency_ms' in res
    assert res['inference_latency_ms'] >= 0.0
    assert 'requires_human_review' in res
    assert isinstance(res['requires_human_review'], bool)
