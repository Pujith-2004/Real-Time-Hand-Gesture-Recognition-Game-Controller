"""
Model Evaluation and Metrics Reporting Module.
Computes evaluation metrics (Accuracy, Precision, Recall, F1-score),
confusion matrices, comparison charts, and exports resume metrics.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

logger = logging.getLogger("ModelEvaluation")


class ModelEvaluator:
    """
    Evaluates ML model candidate results and generates visualization artifacts.
    """

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def evaluate_and_export_all(
        self,
        training_output: Dict[str, Any],
        total_samples: int
    ):
        """
        Processes training output dict, exports CSV/text reports, plots, and resume metrics.
        """
        results = training_output["results"]
        best_model_name = training_output["best_model_name"]
        classes = training_output["classes"]
        y_test = training_output["y_test"]

        comparison_rows = []

        for name, info in results.items():
            y_pred = info["y_pred"]
            acc = accuracy_score(y_test, y_pred)
            prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
            prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

            comparison_rows.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Macro_Precision": round(prec_m, 4),
                "Macro_Recall": round(rec_m, 4),
                "Macro_F1": round(f1_m, 4),
                "Weighted_Precision": round(prec_w, 4),
                "Weighted_Recall": round(rec_w, 4),
                "Weighted_F1": round(f1_w, 4),
                "CV_F1_Mean": round(info["cv_f1_mean"], 4),
                "Latency_ms_per_sample": round(info["latency_ms_per_sample"], 4)
            })

        df_comparison = pd.DataFrame(comparison_rows)
        csv_path = os.path.join(self.reports_dir, "model_comparison.csv")
        df_comparison.to_csv(csv_path, index=False)
        logger.info(f"Model comparison table saved to {csv_path}")

        # Export Classification Report for Best Model
        best_info = results[best_model_name]
        best_y_pred = best_info["y_pred"]
        cls_report_str = classification_report(y_test, best_y_pred, target_names=classes, digits=4)

        report_txt_path = os.path.join(self.reports_dir, "classification_report.txt")
        with open(report_txt_path, "w") as f:
            f.write(f"=== CLASSIFICATION REPORT FOR BEST MODEL ({best_model_name}) ===\n\n")
            f.write(cls_report_str)
        logger.info(f"Classification report saved to {report_txt_path}")

        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, best_y_pred)
        self._plot_confusion_matrix(cm, classes, best_model_name)

        # Plot Model Comparison Bar Chart
        self._plot_model_comparison(df_comparison)

        # Generate Resume Metrics TXT file
        best_row = df_comparison[df_comparison["Model"] == best_model_name].iloc[0]
        resume_txt = (
            f"==================================================\n"
            f"ACTUAL RESUME METRICS - GESTURE CONTROL ML SYSTEM\n"
            f"==================================================\n"
            f"Best Model Selected: {best_model_name}\n"
            f"Accuracy: {best_row['Accuracy']*100:.2f}%\n"
            f"Macro F1-Score: {best_row['Macro_F1']:.4f}\n"
            f"Weighted F1-Score: {best_row['Weighted_F1']:.4f}\n"
            f"Inference Latency: {best_row['Latency_ms_per_sample']:.3f} ms / sample\n"
            f"Gesture Classes: {len(classes)} ({', '.join(classes)})\n"
            f"Total Samples Analyzed: {total_samples}\n"
            f"Engineered Features: 90 features (relative coords, finger distances, angles, tilt)\n"
            f"Cross-Validation: 5-Fold Stratified K-Fold CV\n"
            f"==================================================\n"
        )

        resume_path = os.path.join(self.reports_dir, "resume_metrics.txt")
        with open(resume_path, "w") as f:
            f.write(resume_txt)
        logger.info(f"Resume metrics saved to {resume_path}")

        return df_comparison

    def _plot_confusion_matrix(self, cm: np.ndarray, classes: List[str], model_name: str):
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=False)
        plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "confusion_matrix.png"), dpi=300)
        plt.close()

    def _plot_model_comparison(self, df_comparison: pd.DataFrame):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Model Candidates', fontweight='bold')
        ax1.set_ylabel('Weighted F1 Score', color=color, fontweight='bold')
        bars = ax1.bar(df_comparison['Model'], df_comparison['Weighted_F1'], color=color, alpha=0.7, width=0.4)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1.1)

        # Add data labels on bars
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=9)

        # Secondary Y axis for latency
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Inference Latency (ms)', color=color, fontweight='bold')
        ax2.plot(df_comparison['Model'], df_comparison['Latency_ms_per_sample'], color=color, marker='o', linewidth=2, markersize=8)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title("Model Comparison: Weighted F1-Score vs Inference Latency", fontsize=14, fontweight="bold")
        fig.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "model_comparison.png"), dpi=300)
        plt.close()
