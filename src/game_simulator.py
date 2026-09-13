"""
2D Hill Climb Racing Style Vehicle Simulator (OpenCV Fallback Demo).
Provides a physics-based 2D vehicle sandbox to test gesture controls live.
"""

import cv2
import numpy as np
import math
from typing import Dict


class VehicleSimulator:
    """
    Simulates a 2D Hill Climb vehicle on hilly terrain.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height

        # Vehicle State
        self.distance = 0.0
        self.speed = 0.0
        self.max_speed = 15.0
        self.acceleration = 0.4
        self.braking = 0.6
        self.friction = 0.1

        # Terrain parameters
        self.terrain_base_y = 350

    def reset(self):
        """Reset vehicle state."""
        self.distance = 0.0
        self.speed = 0.0

    def update(self, action: str) -> Dict[str, float]:
        """
        Update vehicle physics based on input action.

        Args:
            action: ACCELERATE, BRAKE, or NEUTRAL

        Returns:
            Dict containing distance and speed.
        """

        if action == "ACCELERATE":
            self.speed = min(
                self.speed + self.acceleration,
                self.max_speed
            )

        elif action == "BRAKE":
            self.speed = max(
                self.speed - self.braking,
                -5.0
            )

        else:
            # Friction / Coasting
            if self.speed > 0:
                self.speed = max(
                    0.0,
                    self.speed - self.friction
                )
            elif self.speed < 0:
                self.speed = min(
                    0.0,
                    self.speed + self.friction
                )

        self.distance += max(
            0.0,
            self.speed * 0.1
        )

        return {
            "distance": round(self.distance, 1),
            "speed": round(self.speed * 10, 1),
        }

    def render(self, active_action: str) -> np.ndarray:
        """
        Render canvas with vehicle physics and terrain.
        """

        canvas = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8
        )

        # Sky background gradient
        for y in range(self.terrain_base_y):
            r = int(
                135 -
                (y / self.terrain_base_y) * 40
            )
            g = int(
                206 -
                (y / self.terrain_base_y) * 30
            )
            b = int(
                235 -
                (y / self.terrain_base_y) * 20
            )

            canvas[y, :] = (b, g, r)

        # Ground fill
        canvas[self.terrain_base_y:, :] = (
            34,
            139,
            34
        )

        # Draw Hilly Terrain Surface Line
        pts = []

        for x in range(0, self.width, 5):
            world_x = self.distance * 10 + x

            y = int(
                self.terrain_base_y
                - 25 * math.sin(world_x * 0.02)
                - 15 * math.cos(world_x * 0.05)
            )

            pts.append((x, y))

        for i in range(len(pts) - 1):
            cv2.line(
                canvas,
                pts[i],
                pts[i + 1],
                (0, 100, 0),
                4
            )

        # Vehicle position
        car_x = self.width // 2

        world_x_car = (
            self.distance * 10 + car_x
        )

        car_y = int(
            self.terrain_base_y
            - 25 * math.sin(world_x_car * 0.02)
            - 15 * math.cos(world_x_car * 0.05)
        ) - 25

        # Draw Vehicle Body
        cv2.rectangle(
            canvas,
            (
                car_x - 30,
                car_y - 15
            ),
            (
                car_x + 30,
                car_y + 15
            ),
            (0, 0, 220),
            -1
        )

        cv2.rectangle(
            canvas,
            (
                car_x - 30,
                car_y - 15
            ),
            (
                car_x + 30,
                car_y + 15
            ),
            (255, 255, 255),
            2
        )

        # Draw Wheels
        wheel_offset = 20

        w1_x = car_x - wheel_offset
        w2_x = car_x + wheel_offset
        wheel_y = car_y + 15

        cv2.circle(
            canvas,
            (w1_x, wheel_y),
            10,
            (30, 30, 30),
            -1
        )

        cv2.circle(
            canvas,
            (w1_x, wheel_y),
            10,
            (200, 200, 200),
            2
        )

        cv2.circle(
            canvas,
            (w2_x, wheel_y),
            10,
            (30, 30, 30),
            -1
        )

        cv2.circle(
            canvas,
            (w2_x, wheel_y),
            10,
            (200, 200, 200),
            2
        )

        # Telemetry
        cv2.rectangle(
            canvas,
            (10, 10),
            (280, 105),
            (0, 0, 0),
            -1
        )

        cv2.rectangle(
            canvas,
            (10, 10),
            (280, 105),
            (0, 255, 255),
            1
        )

        cv2.putText(
            canvas,
            "HILL CLIMB SIMULATOR",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            canvas,
            f"ACTION: {active_action}",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            canvas,
            f"SPEED: {self.speed * 10:.1f} km/h",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1
        )

        cv2.putText(
            canvas,
            f"DISTANCE: {self.distance:.1f} m",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            1
        )

        return canvas