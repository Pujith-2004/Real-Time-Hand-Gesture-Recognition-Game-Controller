"""
Model Training Pipeline.
Trains multiple classical ML models (KNN, SVM, Random Forest, XGBoost),
performs Stratified K-Fold Cross-Validation, benchmarks inference latency,
and exports model artifacts.
"""

import os
import sys
import time
import json
import joblib
import logging
import numpy as np
import pandas as pd

from typing import Dict, Tuple, Any, List
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import FeatureExtractor

logger = logging.getLogger("ModelTraining")


class ModelTrainer:
    """
    Trains candidate ML models on hand landmark features and selects the best model.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = self._initialize_candidate_models()

    def _initialize_candidate_models(self) -> Dict[str, Any]:
        """Initialize classifier models."""
        models = {
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="distance"),
            "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, random_state=self.random_seed),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_seed),
            "Multi-Layer Perceptron": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=self.random_seed)
        }

        if HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_seed,
                eval_metric="mlogloss"
            )
            logger.info("XGBoost classifier included in model candidates.")
        else:
            logger.info("XGBoost not installed. Skipping XGBoost model.")

        return models

    def train_and_evaluate_all(
        self,
        raw_csv_path: str = "data/raw/gestures.csv",
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Execute full training, cross-validation, evaluation, and model selection.
        """
        if not os.path.exists(raw_csv_path):
            raise FileNotFoundError(f"Dataset CSV not found at {raw_csv_path}")

        df_raw = pd.read_csv(raw_csv_path)
        logger.info(f"Loaded dataset containing {len(df_raw)} samples.")

        landmark_cols = [f"{axis}_{i}" for i in range(21) for axis in ["x", "y", "z"]]
        X_landmarks = df_raw[landmark_cols].values
        y_labels = df_raw["gesture"].values

        # 1. Feature Extraction
        logger.info("Extracting 90 engineered features...")
        X_features = self.feature_extractor.extract_batch(X_landmarks)

        # 2. Encode Labels
        y_encoded = self.label_encoder.fit_transform(y_labels)

        # 3. Stratified Train/Test Split (Prevent Data Leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X_features,
            y_encoded,
            test_size=test_size,
            stratify=y_encoded,
            random_state=self.random_seed
        )

        # 4. Standard Scaling (Fitted ONLY on X_train)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Dataset split: Train={len(X_train)} samples, Test={len(X_test)} samples.")

        # 5. Train & Evaluate Candidates
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        results = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            # Cross-validation score on train set
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1_weighted")

            # Fit on full training set
            start_train = time.time()
            model.fit(X_train_scaled, y_train)
            train_time_ms = (time.time() - start_train) * 1000

            # Measure Inference Latency (average time per sample over 100 iterations)
            start_inf = time.time()
            for _ in range(100):
                _ = model.predict(X_test_scaled)
            total_inf_time = time.time() - start_inf
            latency_ms_per_sample = (total_inf_time / (100 * len(X_test_scaled))) * 1000

            # Test Accuracy and F1
            y_pred = model.predict(X_test_scaled)
            from sklearn.metrics import accuracy_score, f1_score
            test_acc = float(accuracy_score(y_test, y_pred))
            test_f1 = float(f1_score(y_test, y_pred, average="weighted"))

            results[name] = {
                "model": model,
                "cv_f1_mean": float(np.mean(cv_scores)),
                "cv_f1_std": float(np.std(cv_scores)),
                "test_accuracy": test_acc,
                "test_f1_weighted": test_f1,
                "train_time_ms": train_time_ms,
                "latency_ms_per_sample": latency_ms_per_sample,
                "y_pred": y_pred
            }

            logger.info(f"[{name}] Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | Latency: {latency_ms_per_sample:.4f} ms")

        # 6. Select Best Model using cross-validation performance
        best_name = max(
            results.keys(),
            key=lambda k: (
                results[k]["cv_f1_mean"],
                -results[k]["latency_ms_per_sample"]
            )
        )
        best_model_info = results[best_name]

        logger.info(
            f"🏆 Best Model Selected: '{best_name}' "
            f"with CV F1-Score={best_model_info['cv_f1_mean']:.4f}"
        )

        return {
            "results": results,
            "best_model_name": best_name,
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "classes": self.label_encoder.classes_.tolist()
        }

    def save_artifacts(
        self,
        best_model_name: str,
        best_model: Any,
        models_dir: str = "models"
    ):
        """Save best model, scaler, label encoder, and metadata json."""
        os.makedirs(models_dir, exist_ok=True)

        # Save best model
        joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))

        # Save preprocessor pipeline dict
        preprocessor_artifacts = {
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_extractor.feature_names
        }
        joblib.dump(preprocessor_artifacts, os.path.join(models_dir, "preprocessor.pkl"))

        # Save metadata JSON
        metadata = {
            "best_model_name": best_model_name,
            "classes": self.label_encoder.classes_.tolist(),
            "num_features": len(self.feature_extractor.feature_names),
            "random_seed": self.random_seed,
            "has_xgboost": HAS_XGBOOST
        }
        with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Successfully saved best model and metadata to '{models_dir}'.")
