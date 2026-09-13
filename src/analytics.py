"""
Telemetry Performance Analytics Module.
Parses telemetry logs to analyze gesture frequencies, inference latency,
uncertainty rates, FPS, and gesture transition statistics.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger("Analytics")


class TelemetryAnalyzer:
    """
    Analyzes logged telemetry session data for performance and latency insights.
    """

    def __init__(self, telemetry_file: str = "data/telemetry/session_telemetry.csv"):
        self.telemetry_file = telemetry_file

    def get_summary(self) -> Dict[str, Any]:
        """
        Compute analytics summary statistics.
        Returns graceful defaults if file does not exist or is empty.
        """
        empty_summary = {
            "total_frames": 0,
            "avg_confidence": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_fps": 0.0,
            "uncertainty_rate_pct": 0.0,
            "gesture_counts": {},
            "gesture_percentages": {},
            "transition_matrix": {},
            "avg_reaction_time_ms": 0.0,
            "file_exists": False
        }

        if not os.path.exists(self.telemetry_file) or os.path.getsize(self.telemetry_file) == 0:
            return empty_summary

        try:
            df = pd.read_csv(self.telemetry_file)
            if df.empty or "predicted_gesture" not in df.columns:
                return empty_summary

            total_frames = len(df)
            avg_conf = float(df["confidence"].mean()) if "confidence" in df.columns else 0.0
            avg_lat = float(df["latency_ms"].mean()) if "latency_ms" in df.columns else 0.0
            p95_lat = float(np.percentile(df["latency_ms"], 95)) if "latency_ms" in df.columns else 0.0
            avg_fps = float(df["fps"].mean()) if "fps" in df.columns else 0.0

            # Low confidence / uncertainty count (e.g. current_action == 'NONE' or gesture == 'NEUTRAL' fallback)
            uncertain_count = len(df[df["current_action"] == "NONE"]) if "current_action" in df.columns else 0
            uncertainty_rate = float((uncertain_count / total_frames) * 100) if total_frames > 0 else 0.0

            # Gesture counts & percentages
            counts = df["predicted_gesture"].value_counts().to_dict()
            pcts = {g: round((c / total_frames) * 100, 2) for g, c in counts.items()}

            # Reaction time (non-zero transitions)
            non_zero_rt = df[df["reaction_time_ms"] > 0]["reaction_time_ms"] if "reaction_time_ms" in df.columns else pd.Series()
            avg_rt = float(non_zero_rt.mean()) if not non_zero_rt.empty else 0.0

            # Gesture transition statistics
            transitions = {}
            gestures = df["predicted_gesture"].values
            for i in range(len(gestures) - 1):
                from_g, to_g = gestures[i], gestures[i+1]
                if from_g != to_g:
                    key = f"{from_g} -> {to_g}"
                    transitions[key] = transitions.get(key, 0) + 1

            return {
                "total_frames": total_frames,
                "avg_confidence": round(avg_conf, 4),
                "avg_latency_ms": round(avg_lat, 3),
                "p95_latency_ms": round(p95_lat, 3),
                "avg_fps": round(avg_fps, 1),
                "uncertainty_rate_pct": round(uncertainty_rate, 2),
                "gesture_counts": counts,
                "gesture_percentages": pcts,
                "transition_matrix": transitions,
                "avg_reaction_time_ms": round(avg_rt, 2),
                "file_exists": True
            }

        except Exception as e:
            logger.warning(f"Error parsing telemetry file: {e}")
            return empty_summary
