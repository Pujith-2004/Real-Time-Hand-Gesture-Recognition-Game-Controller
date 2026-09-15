"""
Unit tests for Model Training and Gesture Predictor.
"""

import os
import pytest
import numpy as np
import pandas as pd
from scripts.generate_sample_data import generate_dataset
from src.model_training import ModelTrainer
from src.gesture_predictor import GesturePredictor


@pytest.fixture(scope="module")
def setup_dataset_and_model(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("data_model")
    csv_path = str(tmp_dir / "gestures.csv")
    models_dir = str(tmp_dir / "models")

    # Generate synthetic dataset
    generate_dataset(samples_per_class=50, output_path=csv_path, seed=42)

    # Train model
    trainer = ModelTrainer(random_seed=42)
    output = trainer.train_and_evaluate_all(raw_csv_path=csv_path)

    best_name = output["best_model_name"]
    best_model = output["results"][best_name]["model"]
    trainer.save_artifacts(best_model_name=best_name, best_model=best_model, models_dir=models_dir)

    return {
        "csv_path": csv_path,
        "models_dir": models_dir,
        "model_path": os.path.join(models_dir, "best_model.pkl"),
        "preprocessor_path": os.path.join(models_dir, "preprocessor.pkl"),
        "classes": output["classes"]
    }


def test_model_training_and_artifacts(setup_dataset_and_model):
    info = setup_dataset_and_model
    assert os.path.exists(info["model_path"])
    assert os.path.exists(info["preprocessor_path"])
    assert len(info["classes"]) == 3


def test_gesture_predictor_inference(setup_dataset_and_model):
    info = setup_dataset_and_model
    predictor = GesturePredictor(
        model_path=info["model_path"],
        preprocessor_path=info["preprocessor_path"],
        confidence_threshold=0.5,
        smoothing_window=3
    )

    dummy_landmarks = np.random.rand(21, 3).astype(np.float32)
    res = predictor.predict(dummy_landmarks)

    assert "predicted_gesture" in res
    assert "confidence" in res
    assert "latency_ms" in res
    assert res["predicted_gesture"] in info["classes"]
