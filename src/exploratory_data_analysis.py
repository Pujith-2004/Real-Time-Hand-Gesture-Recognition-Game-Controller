"""
Exploratory Data Analysis (EDA) Module.
Analyzes dataset distributions, missing/duplicate records, feature correlations,
and visualizes gesture separability using PCA.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import FeatureExtractor

logger = logging.getLogger("EDA")


class GestureEDA:
    """
    Performs EDA on hand-landmark gesture dataset.
    """

    def __init__(self, raw_csv_path: str = "data/raw/gestures.csv", output_dir: str = "reports"):
        self.raw_csv_path = raw_csv_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.feature_extractor = FeatureExtractor()

    def run_analysis(self) -> dict:
        """Execute full EDA pipeline and output plots & summary report."""
        if not os.path.exists(self.raw_csv_path):
            raise FileNotFoundError(f"Raw dataset not found at {self.raw_csv_path}")

        df_raw = pd.read_csv(self.raw_csv_path)
        logger.info(f"Loaded raw dataset with {len(df_raw)} rows.")

        # 1. Dataset stats
        total_samples = len(df_raw)
        class_counts = df_raw["gesture"].value_counts().to_dict()
        missing_count = int(df_raw.isnull().sum().sum())
        landmark_cols = [f"{axis}_{i}" for i in range(21) for axis in ["x", "y", "z"]]
        duplicate_count = int(df_raw.duplicated(subset=landmark_cols).sum())

        # 2. Extract engineered features
        X_landmarks = df_raw[landmark_cols].values
        y_labels = df_raw["gesture"].values
        X_features = self.feature_extractor.extract_batch(X_landmarks)
        feature_names = self.feature_extractor.feature_names

        df_features = pd.DataFrame(X_features, columns=feature_names)
        df_features["gesture"] = y_labels

        # Save processed features CSV
        processed_path = "data/processed/processed_features.csv"
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df_features.to_csv(processed_path, index=False)
        logger.info(f"Saved processed feature dataset ({df_features.shape[1]-1} features) to {processed_path}")

        # 3. Generate Visualizations
        self._plot_class_distribution(class_counts)
        self._plot_correlation_heatmap(df_features, feature_names)
        self._plot_pca_separability(X_features, y_labels)

        # 4. Generate Summary Text Report
        summary_txt = (
            f"========================================\n"
            f"EXPLORATORY DATA ANALYSIS REPORT\n"
            f"========================================\n"
            f"Total Samples: {total_samples}\n"
            f"Missing Values: {missing_count}\n"
            f"Duplicate Samples: {duplicate_count}\n"
            f"Engineered Features: {len(feature_names)}\n\n"
            f"Class Distribution:\n"
        )
        for g, count in class_counts.items():
            summary_txt += f"  - {g}: {count} ({count/total_samples*100:.1f}%)\n"

        report_path = os.path.join(self.output_dir, "eda_summary.txt")
        with open(report_path, "w") as f:
            f.write(summary_txt)

        logger.info(f"EDA Summary saved to {report_path}")
        return {
            "total_samples": total_samples,
            "class_counts": class_counts,
            "missing": missing_count,
            "duplicates": duplicate_count,
            "num_features": len(feature_names)
        }

    def _plot_class_distribution(self, class_counts: dict):
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette="viridis")
        plt.title("Gesture Class Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Gesture Class")
        plt.ylabel("Sample Count")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "eda_class_distribution.png"), dpi=300)
        plt.close()

    def _plot_correlation_heatmap(self, df_features: pd.DataFrame, feature_names: list):
        # Select top interesting features to keep heatmap legible
        selected_cols = [
            "dist_thumb_index", "dist_index_middle", "dist_ring_pinky",
            "dist_wrist_index", "dist_wrist_middle", "ext_index", "ext_thumb",
            "angle_index_pip", "angle_middle_pip", "palm_tilt_roll", "palm_tilt_pitch"
        ]
        cols_present = [c for c in selected_cols if c in df_features.columns]
        if cols_present:
            plt.figure(figsize=(10, 8))
            corr = df_features[cols_present].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
            plt.title("Engineered Feature Correlation Heatmap", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "eda_correlation_heatmap.png"), dpi=300)
            plt.close()

    def _plot_pca_separability(self, X_features: np.ndarray, y_labels: np.ndarray):
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_features)

        plt.figure(figsize=(9, 7))
        classes = np.unique(y_labels)
        for c in classes:
            mask = y_labels == c
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=c, alpha=0.7, edgecolors="k", s=40)

        plt.title(f"Gesture Separability in PCA Space (Explained Var: {sum(pca.explained_variance_ratio_)*100:.1f}%)", fontsize=14, fontweight="bold")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Gestures")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "eda_pca_separability.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    eda = GestureEDA()
    eda.run_analysis()
