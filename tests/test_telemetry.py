"""
Unit tests for Telemetry Logging and Analytics calculation.
"""

import os
import pytest
from src.telemetry import TelemetryLogger
from src.analytics import TelemetryAnalyzer


def test_telemetry_logging_and_analytics(tmp_path):
    telemetry_file = str(tmp_path / "test_telemetry.csv")
    logger = TelemetryLogger(telemetry_file=telemetry_file)

    for i in range(10):
        logger.log_frame(
            predicted_gesture="ACCELERATE" if i % 2 == 0 else "BRAKE",
            confidence=0.92,
            current_action="right" if i % 2 == 0 else "left",
            latency_ms=2.5,
            hand_detected=True,
            fps=30.0,
            vehicle_speed=12.0,
            vehicle_angle=0.0
        )
    logger.flush()

    assert os.path.exists(telemetry_file)

    analyzer = TelemetryAnalyzer(telemetry_file=telemetry_file)
    summary = analyzer.get_summary()

    assert summary["total_frames"] == 10
    assert summary["avg_confidence"] == 0.92
    assert summary["file_exists"] is True
    assert "ACCELERATE" in summary["gesture_counts"]
