"""
Synthetic Sample Dataset Generator for Hand Gestures.
Generates geometrically realistic 21-landmark 3D data for all 5 gesture classes
to bootstrap model training, evaluation, and test suites.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collect_data import get_csv_columns


def generate_base_gesture_landmarks(gesture: str) -> np.ndarray:
    """
    Generate representative base 21x3 3D landmarks for each gesture pose.
    Landmarks are relative to wrist at (0.5, 0.8, 0.0).
    """
    landmarks = np.zeros((21, 3), dtype=np.float32)
    wrist = np.array([0.5, 0.8, 0.0])
    landmarks[0] = wrist

    if gesture == "ACCELERATE":
        # Fist / Clenched hand pose
        # MCP joints
        landmarks[1] = wrist + np.array([-0.06, -0.05, -0.02]) # Thumb CMC
        landmarks[2] = wrist + np.array([-0.08, -0.10, -0.03]) # Thumb MCP
        landmarks[3] = wrist + np.array([-0.06, -0.13, -0.02]) # Thumb IP
        landmarks[4] = wrist + np.array([-0.03, -0.14, -0.01]) # Thumb Tip (folded over index)

        landmarks[5] = wrist + np.array([-0.04, -0.15, 0.0])  # Index MCP
        landmarks[6] = wrist + np.array([-0.04, -0.12, 0.03]) # Index PIP (curled)
        landmarks[7] = wrist + np.array([-0.04, -0.09, 0.02]) # Index DIP
        landmarks[8] = wrist + np.array([-0.04, -0.07, 0.01]) # Index Tip

        landmarks[9] = wrist + np.array([0.0, -0.16, 0.0])    # Middle MCP
        landmarks[10] = wrist + np.array([0.0, -0.13, 0.03])  # Middle PIP
        landmarks[11] = wrist + np.array([0.0, -0.10, 0.02])  # Middle DIP
        landmarks[12] = wrist + np.array([0.0, -0.08, 0.01])  # Middle Tip

        landmarks[13] = wrist + np.array([0.04, -0.15, 0.0])  # Ring MCP
        landmarks[14] = wrist + np.array([0.04, -0.12, 0.03]) # Ring PIP
        landmarks[15] = wrist + np.array([0.04, -0.09, 0.02]) # Ring DIP
        landmarks[16] = wrist + np.array([0.04, -0.07, 0.01]) # Ring Tip

        landmarks[17] = wrist + np.array([0.07, -0.13, 0.0])  # Pinky MCP
        landmarks[18] = wrist + np.array([0.07, -0.10, 0.03]) # Pinky PIP
        landmarks[19] = wrist + np.array([0.07, -0.08, 0.02]) # Pinky DIP
        landmarks[20] = wrist + np.array([0.07, -0.06, 0.01]) # Pinky Tip

    elif gesture == "BRAKE":
        # Fully open palm facing camera (high finger extension)
        landmarks[1] = wrist + np.array([-0.06, -0.05, -0.02])
        landmarks[2] = wrist + np.array([-0.10, -0.08, -0.04])
        landmarks[3] = wrist + np.array([-0.14, -0.12, -0.05])
        landmarks[4] = wrist + np.array([-0.18, -0.16, -0.06]) # Thumb Tip extended far left

        landmarks[5] = wrist + np.array([-0.05, -0.15, 0.0])
        landmarks[6] = wrist + np.array([-0.07, -0.23, 0.0])
        landmarks[7] = wrist + np.array([-0.08, -0.30, 0.0])
        landmarks[8] = wrist + np.array([-0.09, -0.36, 0.0])  # Index Tip extended straight up

        landmarks[9] = wrist + np.array([0.0, -0.16, 0.0])
        landmarks[10] = wrist + np.array([0.0, -0.25, 0.0])
        landmarks[11] = wrist + np.array([0.0, -0.33, 0.0])
        landmarks[12] = wrist + np.array([0.0, -0.40, 0.0])   # Middle Tip extended straight up

        landmarks[13] = wrist + np.array([0.05, -0.15, 0.0])
        landmarks[14] = wrist + np.array([0.07, -0.23, 0.0])
        landmarks[15] = wrist + np.array([0.08, -0.30, 0.0])
        landmarks[16] = wrist + np.array([0.09, -0.36, 0.0])  # Ring Tip extended straight up

        landmarks[17] = wrist + np.array([0.09, -0.13, 0.0])
        landmarks[18] = wrist + np.array([0.13, -0.19, 0.0])
        landmarks[19] = wrist + np.array([0.16, -0.25, 0.0])
        landmarks[20] = wrist + np.array([0.19, -0.30, 0.0])  # Pinky Tip extended far right

    elif gesture == "TILT_LEFT":
        # Open hand tilted ~ -45 degrees left (rotated relative to wrist)
        brake_base = generate_base_gesture_landmarks("BRAKE") - wrist
        theta = np.radians(-45)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,             0,              1]
        ])
        landmarks = np.dot(brake_base, rot_matrix.T) + wrist

    elif gesture == "TILT_RIGHT":
        # Open hand tilted ~ +45 degrees right (rotated relative to wrist)
        brake_base = generate_base_gesture_landmarks("BRAKE") - wrist
        theta = np.radians(45)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,             0,              1]
        ])
        landmarks = np.dot(brake_base, rot_matrix.T) + wrist

    else:  # NEUTRAL
        # Relaxed hand, fingers half-curled / resting pose
        landmarks[1] = wrist + np.array([-0.04, -0.04, -0.01])
        landmarks[2] = wrist + np.array([-0.06, -0.07, -0.02])
        landmarks[3] = wrist + np.array([-0.07, -0.10, -0.02])
        landmarks[4] = wrist + np.array([-0.07, -0.13, -0.02])

        landmarks[5] = wrist + np.array([-0.03, -0.14, 0.0])
        landmarks[6] = wrist + np.array([-0.04, -0.19, 0.01])
        landmarks[7] = wrist + np.array([-0.04, -0.23, 0.01])
        landmarks[8] = wrist + np.array([-0.04, -0.26, 0.01])

        landmarks[9] = wrist + np.array([0.0, -0.15, 0.0])
        landmarks[10] = wrist + np.array([0.0, -0.20, 0.01])
        landmarks[11] = wrist + np.array([0.0, -0.24, 0.01])
        landmarks[12] = wrist + np.array([0.0, -0.27, 0.01])

        landmarks[13] = wrist + np.array([0.03, -0.14, 0.0])
        landmarks[14] = wrist + np.array([0.04, -0.19, 0.01])
        landmarks[15] = wrist + np.array([0.04, -0.23, 0.01])
        landmarks[16] = wrist + np.array([0.04, -0.25, 0.01])

        landmarks[17] = wrist + np.array([0.06, -0.12, 0.0])
        landmarks[18] = wrist + np.array([0.07, -0.16, 0.01])
        landmarks[19] = wrist + np.array([0.07, -0.19, 0.01])
        landmarks[20] = wrist + np.array([0.07, -0.21, 0.01])

    return landmarks


def generate_dataset(samples_per_class: int = 500, output_path: str = "data/raw/gestures.csv", seed: int = 42):
    """
    Generate dataset CSV with Gaussian noise per sample.
    """
    np.random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    gestures = ["ACCELERATE", "BRAKE", "TILT_LEFT", "TILT_RIGHT", "NEUTRAL"]
    cols = get_csv_columns()
    rows = []

    print(f"Generating {samples_per_class} samples per gesture ({len(gestures)} classes)...")

    for g in gestures:
        base_lm = generate_base_gesture_landmarks(g)
        for i in range(samples_per_class):
            # Add subtle hand scale variation (0.85 to 1.15)
            scale = np.random.uniform(0.85, 1.15)
            # Add spatial translation shift (x, y, z)
            shift = np.random.normal(0, 0.03, size=(1, 3))
            # Add per-landmark noise
            noise = np.random.normal(0, 0.008, size=(21, 3))

            sample_lm = (base_lm - base_lm[0]) * scale + base_lm[0] + shift + noise
            confidence = float(np.clip(np.random.normal(0.95, 0.03), 0.75, 1.0))
            handedness = "Right"

            row = [
                datetime.now().isoformat(),
                g,
                handedness,
                round(confidence, 4)
            ]
            row.extend(sample_lm.flatten().tolist())
            rows.append(row)

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully! Saved {len(df)} samples to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Gesture Dataset Generator")
    parser.add_argument("--samples", type=int, default=500, help="Samples per gesture class")
    parser.add_argument("--output", type=str, default="data/raw/gestures.csv", help="Output CSV path")
    args = parser.parse_args()

    generate_dataset(samples_per_class=args.samples, output_path=args.output)
