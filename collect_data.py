"""
Dataset Collection Utility for Real-Time Hand Landmarks.
Captures 21 3D hand landmarks via webcam and saves structured raw samples to CSV.
"""

import os
import sys
import time
import argparse
import logging
import cv2
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.hand_tracker import HandTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataCollector")


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_csv_columns() -> list:
    """Generate CSV header column names."""
    cols = ["timestamp", "gesture", "handedness", "confidence"]
    for i in range(21):
        cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    return cols


def collect_data(gesture: str, num_samples: int = 1000, config_path: str = "config/config.yaml"):
    """
    Main loop for webcam hand-landmark collection.
    """
    config = load_config(config_path)
    valid_gestures = config["gestures"]["classes"]

    if gesture not in valid_gestures:
        logger.error(f"Invalid gesture '{gesture}'. Must be one of: {valid_gestures}")
        sys.exit(1)

    csv_path = config["paths"]["raw_data"]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Initialize tracker and camera
    camera_id = config["camera"]["device_id"]
    tracker = HandTracker(
        max_num_hands=config["mediapipe"]["max_num_hands"],
        min_detection_confidence=config["mediapipe"]["min_detection_confidence"],
        min_tracking_confidence=config["mediapipe"]["min_tracking_confidence"]
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Cannot open webcam (device ID {camera_id}). Check camera connection.")
        sys.exit(1)

    # Set frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["frame_height"])

    # Count existing samples for this gesture in CSV
    existing_count = 0
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            if "gesture" in df_existing.columns:
                existing_count = len(df_existing[df_existing["gesture"] == gesture])
        except Exception as e:
            logger.warning(f"Could not read existing CSV to count samples: {e}")

    logger.info(f"=== Gesture Collection Started for '{gesture}' ===")
    logger.info(f"Session Target: {num_samples} new samples | Existing: {existing_count}")
    logger.info("Press 'SPACE' to toggle recording on/off. Press 'Q' or 'ESC' to finish.")

    is_recording = False
    samples_collected_session = 0
    buffer = []


    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to grab camera frame.")
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)  # Mirror frame
            result = tracker.process_frame(frame)
            annotated_frame = tracker.draw_landmarks(frame, result)

            # Check recording logic
            if is_recording and result.hand_detected and result.landmarks_3d is not None:
                row = [
                    datetime.now().isoformat(),
                    gesture,
                    result.handedness,
                    round(result.confidence, 4)
                ]
                row.extend(result.landmarks_3d.flatten().tolist())
                buffer.append(row)
                samples_collected_session += 1
                if samples_collected_session >= num_samples:
                    logger.info(f"Session target ({num_samples}) reached!")
                    is_recording = False
                    break

            # Draw HUD UI Overlay
            h, w, _ = annotated_frame.shape

            # Top Header Bar
            cv2.rectangle(annotated_frame, (0, 0), (w, 80), (30, 30, 30), -1)

            cv2.putText(
                annotated_frame,
                f"GESTURE: {gesture}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            rec_status = "RECORDING" if is_recording else "PAUSED (Press SPACE)"
            status_color = (0, 255, 0) if is_recording else (0, 165, 255)
            cv2.putText(
                annotated_frame,
                f"STATUS: {rec_status}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2
            )

            count_str = f"Samples: {samples_collected_session}/{num_samples}"
            cv2.putText(
                annotated_frame,
                count_str,
                (w - 240, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Hand Detection indicator
            hand_str = f"Hand: {result.handedness} ({result.confidence:.2f})" if result.hand_detected else "Hand: NOT DETECTED"
            hand_color = (0, 255, 0) if result.hand_detected else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                hand_str,
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                hand_color,
                2
            )

            cv2.imshow("Gesture Data Collector", annotated_frame)
            key = cv2.waitKey(config["data_collection"]["delay_between_frames_ms"]) & 0xFF

            if key in [ord('q'), ord('Q'), 27]:  # 27 is ESC
                break
            elif key == 32:  # SPACE
                is_recording = not is_recording
                logger.info(f"Recording state changed to: {'ACTIVE' if is_recording else 'PAUSED'}")

            # Flush buffer to CSV periodically (every 50 samples)
            if len(buffer) >= 50:
                _save_buffer_to_csv(buffer, csv_path)
                buffer.clear()

    finally:
        if buffer:
            _save_buffer_to_csv(buffer, csv_path)
            buffer.clear()

        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        logger.info(f"Session finished. Collected {samples_collected_session} new samples for '{gesture}'. Saved to {csv_path}.")


def _save_buffer_to_csv(buffer: list, csv_path: str):
    """Save buffered rows to CSV file."""
    cols = get_csv_columns()
    df_new = pd.DataFrame(buffer, columns=cols)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    df_new.to_csv(csv_path, mode="a" if file_exists else "w", header=not file_exists, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Gesture Dataset Collector")
    parser.add_argument(
        "--gesture",
        type=str,
        required=True,
        help="Gesture label (ACCELERATE, BRAKE, NEUTRAL)"
    )
    parser.add_argument("--samples", type=int, default=1000, help="Target number of samples to collect")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    collect_data(gesture=args.gesture.upper(), num_samples=args.samples, config_path=args.config)
