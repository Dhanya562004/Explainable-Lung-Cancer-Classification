import pytest
import numpy as np
from PIL import Image
from src.data.preprocessing import validate_image_file, load_and_preprocess_image
from src.config import IMAGE_SIZE

def test_validate_image_file():
    img = Image.new('RGB', (100, 100), color='white')
    is_valid, msg = validate_image_file(img)
    assert is_valid is True

    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    is_valid, msg = validate_image_file(arr)
    assert is_valid is True

    is_valid, msg = validate_image_file("non_existent_file.png")
    assert is_valid is False

def test_load_and_preprocess_image():
    img = Image.new('RGB', (200, 200), color='red')
    tensor, orig_pil = load_and_preprocess_image(img, target_size=IMAGE_SIZE)

    assert tensor.shape == (1, 299, 299, 3)
    assert tensor.dtype == np.float32
    assert orig_pil.size == (200, 200)
