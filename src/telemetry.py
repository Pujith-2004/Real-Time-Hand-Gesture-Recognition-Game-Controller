"""
Telemetry Logger Module.
Records frame-by-frame inference metrics, gesture transitions, vehicle state, and system performance.
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("Telemetry")


class TelemetryLogger:
    """
    Logs gameplay and model inference telemetry data to CSV.
    """

    def __init__(self, telemetry_file: str = "data/telemetry/session_telemetry.csv"):
        self.telemetry_file = telemetry_file
        os.makedirs(os.path.dirname(telemetry_file), exist_ok=True)

        self.last_gesture = "NEUTRAL"
        self.last_gesture_change_time = time.time()
        self.buffer = []
        self.buffer_size = 30  # Flush every 30 records (approx 1 sec at 30 FPS)

        self.columns = [
            "timestamp",
            "predicted_gesture",
            "confidence",
            "current_action",
            "latency_ms",
            "reaction_time_ms",
            "hand_detected",
            "fps",
            "vehicle_speed",
            "vehicle_angle"
        ]

    def log_frame(
        self,
        predicted_gesture: str,
        confidence: float,
        current_action: Optional[str],
        latency_ms: float,
        hand_detected: bool,
        fps: float,
        vehicle_speed: float = 0.0,
        vehicle_angle: float = 0.0
    ):
        """
        Record a single telemetry frame.
        """
        now = time.time()
        # Calculate transition delay / system reaction time
        reaction_time_ms = 0.0
        if predicted_gesture != self.last_gesture:
            reaction_time_ms = round((now - self.last_gesture_change_time) * 1000, 2)
            self.last_gesture = predicted_gesture
            self.last_gesture_change_time = now

        row = [
            datetime.now().isoformat(),
            predicted_gesture,
            round(confidence, 4),
            current_action if current_action else "NONE",
            round(latency_ms, 2),
            reaction_time_ms,
            hand_detected,
            round(fps, 1),
            round(vehicle_speed, 1),
            round(vehicle_angle, 1)
        ]

        self.buffer.append(row)

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Write buffered telemetry rows to disk."""
        if not self.buffer:
            return

        file_exists = os.path.exists(self.telemetry_file) and os.path.getsize(self.telemetry_file) > 0
        df = pd.DataFrame(self.buffer, columns=self.columns)
        df.to_csv(self.telemetry_file, mode="a" if file_exists else "w", header=not file_exists, index=False)
        self.buffer.clear()

    def close(self):
        """Flush remaining buffer on exit."""
        self.flush()
        logger.info(f"Telemetry session flushed and saved to {self.telemetry_file}")
