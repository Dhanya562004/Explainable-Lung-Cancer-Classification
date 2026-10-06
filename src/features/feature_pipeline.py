import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

from src.config import IMAGE_SIZE, DEFAULT_MODEL_PATH
from src.data.preprocessing import load_and_preprocess_image

_feature_extractor_model = None

def get_feature_extractor(model=None):
    """
    Returns feature extraction sub-model that outputs Xception backbone embeddings (2048 dimensions).
    """
    global _feature_extractor_model
    if _feature_extractor_model is not None and model is None:
        return _feature_extractor_model

    if model is not None:
        # Try to find global average pooling layer or base model layer
        try:
            gap_layer = model.get_layer('global_avg_pool')
            extractor = Model(inputs=model.input, outputs=gap_layer.output)
            return extractor
        except ValueError:
            pass

    # Fallback to pure Xception ImageNet backbone with GAP
    base = Xception(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    out = GlobalAveragePooling2D(name='global_avg_pool')(base.output)
    extractor = Model(inputs=base.input, outputs=out, name='Xception_Feature_Extractor')
    
    if model is None:
        _feature_extractor_model = extractor
    return extractor

def get_image_embedding(img_input, model=None):
    """
    Extracts 2048-dimensional feature embedding vector for a single image.
    """
    img_tensor, _ = load_and_preprocess_image(img_input)
    extractor = get_feature_extractor(model)
    embedding = extractor.predict(img_tensor, verbose=0)[0]
    return embedding

class FeatureExtractor:
    """
    Utility class for batch feature extraction and representation analysis.
    """
    def __init__(self, model=None):
        self.extractor = get_feature_extractor(model)
        self.embedding_dim = 2048

    def extract_batch(self, image_tensors):
        """
        Extracts embeddings for a batch of preprocessed image tensors.
        """
        embeddings = self.extractor.predict(image_tensors, verbose=0)
        return embeddings

    def analyze_dataset_embeddings(self, dataset_dir, max_samples_per_class=10):
        """
        Generates sample embedding dataset for representation analysis across classes.
        """
        from src.config import CLASS_NAMES
        records = []
        embeddings_list = []

        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(dataset_dir, 'test', cls_name)
            if not os.path.exists(cls_dir):
                continue
            
            fnames = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            fnames = fnames[:max_samples_per_class]

            for fn in fnames:
                fpath = os.path.join(cls_dir, fn)
                try:
                    emb = get_image_embedding(fpath, model=self.extractor)
                    embeddings_list.append(emb)
                    records.append({
                        'filename': fn,
                        'class_name': cls_name,
                        'embedding_mean': float(np.mean(emb)),
                        'embedding_std': float(np.std(emb))
                    })
                except Exception:
                    continue

        embeddings_arr = np.array(embeddings_list) if embeddings_list else np.empty((0, self.embedding_dim))
        
        return {
            'embedding_dim': self.embedding_dim,
            'total_analyzed': len(records),
            'records': records,
            'embeddings_shape': embeddings_arr.shape,
            'overall_mean': float(np.mean(embeddings_arr)) if len(embeddings_arr) > 0 else 0.0,
            'overall_std': float(np.std(embeddings_arr)) if len(embeddings_arr) > 0 else 0.0
        }
