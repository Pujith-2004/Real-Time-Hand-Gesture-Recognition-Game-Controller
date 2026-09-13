"""
Main Training Pipeline Entry Point.
python train.py
"""

import os
import sys
import logging
import yaml
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from scripts.generate_sample_data import generate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainPipeline")


def load_config(config_path: str = "config/config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_training_pipeline(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    raw_data_path = config["paths"]["raw_data"]

    # If raw data does not exist, bootstrap with sample data generator
    if not os.path.exists(raw_data_path) or os.path.getsize(raw_data_path) == 0:
        logger.warning(f"Raw dataset file '{raw_data_path}' not found or empty.")
        logger.info("Generating synthetic gesture dataset to bootstrap training...")
        generate_dataset(samples_per_class=500, output_path=raw_data_path, seed=config["system"]["random_seed"])

    df_raw = pd.read_csv(raw_data_path)
    total_samples = len(df_raw)

    # Execute Model Trainer
    trainer = ModelTrainer(random_seed=config["system"]["random_seed"])
    training_output = trainer.train_and_evaluate_all(raw_csv_path=raw_data_path)

    # Save artifacts
    best_name = training_output["best_model_name"]
    best_model = training_output["results"][best_name]["model"]
    trainer.save_artifacts(best_model_name=best_name, best_model=best_model, models_dir="models")

    # Evaluate & generate reports
    evaluator = ModelEvaluator(reports_dir=config["paths"]["reports_dir"])
    evaluator.evaluate_and_export_all(training_output=training_output, total_samples=total_samples)

    logger.info("🎉 Training Pipeline Completed Successfully!")
    logger.info(f"Best Model Saved: {best_name} -> {config['paths']['model_path']}")


if __name__ == "__main__":
    run_training_pipeline()
