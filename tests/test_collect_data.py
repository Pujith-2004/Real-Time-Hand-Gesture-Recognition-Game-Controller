"""
Unit tests for Dataset Collection module.
"""

import os
import pytest
import pandas as pd
from collect_data import load_config, get_csv_columns, _save_buffer_to_csv


def test_get_csv_columns():
    cols = get_csv_columns()
    assert len(cols) == 4 + 63  # timestamp, gesture, handedness, confidence + 21*3 coords
    assert cols[0] == "timestamp"
    assert cols[1] == "gesture"
    assert cols[2] == "handedness"
    assert cols[3] == "confidence"
    assert cols[4] == "x_0"
    assert cols[66] == "z_20"


def test_load_config():
    config = load_config("config/config.yaml")
    assert "gestures" in config
    assert "classes" in config["gestures"]
    assert len(config["gestures"]["classes"]) == 3
    assert "ACCELERATE" in config["gestures"]["classes"]


def test_save_buffer_to_csv(tmp_path):
    csv_file = str(tmp_path / "test_gestures.csv")
    cols = get_csv_columns()

    dummy_row = ["2026-09-04T18:00:00", "ACCELERATE", "Right", 0.95] + [0.1] * 63
    buffer = [dummy_row, dummy_row]

    _save_buffer_to_csv(buffer, csv_file)

    assert os.path.exists(csv_file)
    df = pd.read_csv(csv_file)
    assert len(df) == 2
    assert list(df.columns) == cols
    assert df["gesture"].iloc[0] == "ACCELERATE"
