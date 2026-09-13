"""
Unit tests for Feature Extraction.
"""

import pytest
import numpy as np
from src.feature_engineering import FeatureExtractor


def test_feature_extractor_output_dimension():
    extractor = FeatureExtractor()
    dummy_landmarks = np.random.rand(21, 3).astype(np.float32)

    features = extractor.extract_features(dummy_landmarks)
    assert len(features) == len(extractor.feature_names)
    assert len(features) == 90  # 63 + 5 + 5 + 5 + 7 + 3 + 2 = 90


def test_feature_batch_extraction():
    extractor = FeatureExtractor()
    batch_landmarks = np.random.rand(10, 63).astype(np.float32)

    batch_features = extractor.extract_batch(batch_landmarks)
    assert batch_features.shape == (10, 90)


def test_angle_calculation_range():
    extractor = FeatureExtractor()
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([0.0, 1.0, 0.0])

    angle = extractor._calculate_angle(p1, p2, p3)
    assert pytest.approx(angle, 0.1) == 90.0
