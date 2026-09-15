"""
Real-Time Gesture Recognition & Game Control Application.

Run:
    python app.py
"""

import os
import sys
import time
import logging
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import yaml

from src.hand_tracker import HandTracker
from src.gesture_predictor import GesturePredictor
from src.chrome_game_controller import ChromeGameController
from src.game_simulator import VehicleSimulator
from src.telemetry import TelemetryLogger





logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("AppMain")


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load application configuration."""

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at {config_path}"
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_startup_banner(config: dict, model_path: str):
    """Print application startup information."""

    print("\n" + "=" * 45)
    print("      GESTURE GAME CONTROLLER SYSTEM")
    print("=" * 45)

    print(
        f"Camera Device      : "
        f"{config['camera']['device_id']} "
        f"({config['camera']['frame_width']}x"
        f"{config['camera']['frame_height']})"
    )

    print(f"Model Path         : {model_path}")

    print(
        f"Gesture Classes    : "
        f"{len(config['gestures']['classes'])} "
        f"({', '.join(config['gestures']['classes'])})"
    )

    print(
        f"Smoothing Window   : "
        f"{config['inference']['smoothing_window']} frames"
    )

    print(
        f"Conf Threshold     : "
        f"{config['inference']['confidence_threshold']}"
    )

    print(
        f"Telemetry File     : "
        f"{config['paths']['telemetry_file']}"
    )

    print("Controls           : Press 'Q' or 'ESC' to Quit")
    print("=" * 45 + "\n")


def run_application(config_path: str = "config/config.yaml"):
    """Run the real-time gesture controller."""

    config = load_config(config_path)

    # ---------------------------------------------------------
    # 1. Initialize Components
    # ---------------------------------------------------------

    model_path = config["paths"]["model_path"]
    preprocessor_path = config["paths"]["preprocessor_path"]

    if not os.path.exists(model_path):
        logger.error(
            f"Model file '{model_path}' not found! "
            f"Train the model first by running: python train.py"
        )
        sys.exit(1)

    tracker = HandTracker(
        max_num_hands=config["mediapipe"]["max_num_hands"],
        min_detection_confidence=config["mediapipe"][
            "min_detection_confidence"
        ],
        min_tracking_confidence=config["mediapipe"][
            "min_tracking_confidence"
        ]
    )

    predictor = GesturePredictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        confidence_threshold=config["inference"][
            "confidence_threshold"
        ],
        smoothing_window=config["inference"][
            "smoothing_window"
        ]
    )

    controller = ChromeGameController(
        stable_frames=config["inference"].get(
            "control_stability_frames",
            1
        )
    )

    controller.connect()

    simulator = VehicleSimulator(
        width=480,
        height=360
    )

    telemetry = TelemetryLogger(
        telemetry_file=config["paths"]["telemetry_file"]
    )

    # ---------------------------------------------------------
    # 2. Open Camera
    # ---------------------------------------------------------

    camera_id = config["camera"]["device_id"]

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        logger.error(
            f"Failed to open camera device ID {camera_id}."
        )
        sys.exit(1)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        config["camera"]["frame_width"]
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        config["camera"]["frame_height"]
    )

    print_startup_banner(config, model_path)

    fps = 30.0
    prev_frame_time = time.time()

    # -------------------------------------------------
    # Create display windows once
    # -------------------------------------------------

    cv2.namedWindow("Gesture Control HUD", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture Control HUD", 480, 360)

    cv2.namedWindow("2D Hill Climb Game Simulator", cv2.WINDOW_NORMAL)

    try:

        while True:

            # -------------------------------------------------
            # Read Camera Frame
            # -------------------------------------------------

            ret, frame = cap.read()

            if not ret or frame is None:

                logger.warning(
                    "Camera frame read error."
                )

                controller.release_all_keys()

                time.sleep(0.05)

                continue

            # -------------------------------------------------
            # Calculate FPS
            # -------------------------------------------------

            curr_frame_time = time.time()

            fps = (
                0.9 * fps
                + 0.1
                * (
                    1.0
                    / max(
                        curr_frame_time
                        - prev_frame_time,
                        1e-5
                    )
                )
            )

            prev_frame_time = curr_frame_time

            # -------------------------------------------------
            # Mirror Camera
            # -------------------------------------------------

            frame = cv2.flip(frame, 1)

            # -------------------------------------------------
            # Hand Tracking
            # -------------------------------------------------

            result = tracker.process_frame(frame)

            annotated_frame = tracker.draw_landmarks(
                frame,
                result
            )

            # -------------------------------------------------
            # Gesture Prediction
            # -------------------------------------------------

            if (
                result.hand_detected
                and result.landmarks_3d is not None
            ):

                pred_dict = predictor.predict(
                    result.landmarks_3d
                )

                predicted_gesture = (
                    pred_dict["predicted_gesture"]
                )

                confidence = pred_dict["confidence"]

                latency_ms = pred_dict["latency_ms"]

            else:

                predictor.reset_smoothing()

                predicted_gesture = "NEUTRAL"

                confidence = 0.0

                latency_ms = 0.0

            # -------------------------------------------------
            # Game Control
            # -------------------------------------------------

            if not result.hand_detected:

                controller.release_all_keys()

                active_action = None

            else:

                active_action = controller.update_action(
                    predicted_gesture
                )

            # -------------------------------------------------
            # Vehicle Simulator
            # -------------------------------------------------

            sim_state = simulator.update(
                predicted_gesture
            )

            sim_canvas = simulator.render(
                predicted_gesture
            )

            # -------------------------------------------------
            # Telemetry
            # -------------------------------------------------

            telemetry.log_frame(
                predicted_gesture=predicted_gesture,
                confidence=confidence,
                current_action=active_action,
                latency_ms=latency_ms,
                hand_detected=result.hand_detected,
                fps=fps,
                vehicle_speed=sim_state["speed"],
                vehicle_angle=0.0
            )

            # -------------------------------------------------
            # HUD
            # -------------------------------------------------

            h, w, _ = annotated_frame.shape

            cv2.rectangle(
                annotated_frame,
                (0, 0),
                (w, 75),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                annotated_frame,
                f"GESTURE: {predicted_gesture}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                f"CONFIDENCE: {confidence * 100:.1f}%",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            act_str = (
                f"ACTION: "
                f"{active_action if active_action else 'NONE'}"
            )

            cv2.putText(
                annotated_frame,
                act_str,
                (w - 230, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f} | "
                f"Latency: {latency_ms:.1f}ms",
                (w - 250, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1
            )

            # -------------------------------------------------
            # Status
            # -------------------------------------------------

            status_text = (
                f"Hand: {result.handedness}"
                if result.hand_detected
                else "NO HAND DETECTED"
            )

            status_color = (
                (0, 255, 0)
                if result.hand_detected
                else (0, 0, 255)
            )

            cv2.putText(
                annotated_frame,
                status_text,
                (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2
            )

            # -------------------------------------------------
            # Display
            # -------------------------------------------------
            

            cv2.imshow(
                "Gesture Control HUD",
                annotated_frame
            )

            cv2.imshow(
                "2D Hill Climb Game Simulator",
                sim_canvas
            )

            # -------------------------------------------------
            # Keyboard
            # -------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:

                logger.info(
                    "User requested exit."
                )

                break

    finally:

        logger.info(
            "Cleaning up resources and releasing held keys..."
        )

        controller.release_all_keys()

        telemetry.close()

        cap.release()

        cv2.destroyAllWindows()

        tracker.close()

        logger.info(
            "Application shutdown complete."
        )


if __name__ == "__main__":
    run_application()