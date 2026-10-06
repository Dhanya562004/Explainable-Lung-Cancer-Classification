"""
Feature extraction pipeline module around Xception backbone.
"""
from .feature_pipeline import FeatureExtractor, get_image_embedding

__all__ = ['FeatureExtractor', 'get_image_embedding']
