"""
Chrome Game Controller.

Controls Hill Climb Racing through Chrome DevTools Protocol (CDP).

Gestures:
    ACCELERATE -> ArrowRight held
    BRAKE      -> gradual ArrowLeft pulses
    NEUTRAL    -> release keys
"""

import json
import time

import requests
import websocket


class ChromeGameController:
    """Control Hill Climb Racing through Chrome DevTools Protocol."""

    def __init__(
        self,
        port=9222,
        stable_frames=1,
        brake_pulse_ms=40,
        brake_cooldown_ms=180,
    ):
        self.port = port
        self.websocket = None
        self.message_id = 0

        # Current held movement key
        self.current_key = None

        # Gesture stability
        self.stable_frames = max(1, int(stable_frames))
        self.pending_gesture = None
        self.pending_count = 0
        self.active_gesture = None

        # Gradual braking
        self.brake_pulse_seconds = max(
            0.01,
            brake_pulse_ms / 1000.0
        )
        self.brake_cooldown_seconds = max(
            0.0,
            brake_cooldown_ms / 1000.0
        )
        self.last_brake_time = 0.0

    def connect(self):
        """Connect to the Hill Climb Racing tab."""

        tabs = requests.get(
            f"http://localhost:{self.port}/json",
            timeout=5,
        ).json()

        game_tab = next(
            (
                tab
                for tab in tabs
                if "hill-climb-racing.com" in tab.get("url", "")
            ),
            None,
        )

        if game_tab is None:
            raise RuntimeError(
                "Hill Climb Racing tab not found in Chrome."
            )

        self.websocket = websocket.create_connection(
            game_tab["webSocketDebuggerUrl"],
            origin=f"http://localhost:{self.port}",
        )

        print("Connected to Hill Climb Racing browser game.")

    def _send_key(self, event_type, key, key_code):
        """Send a keyboard event through Chrome."""

        if self.websocket is None:
            self.connect()

        self.message_id += 1

        message = {
            "id": self.message_id,
            "method": "Input.dispatchKeyEvent",
            "params": {
                "type": event_type,
                "key": key,
                "code": key,
                "windowsVirtualKeyCode": key_code,
                "nativeVirtualKeyCode": key_code,
            },
        }

        self.websocket.send(json.dumps(message))

    def press(self, key):
        """Press and hold a movement key."""

        key_codes = {
            "right": ("ArrowRight", 39),
            "left": ("ArrowLeft", 37),
        }

        if key not in key_codes:
            return

        key_name, key_code = key_codes[key]

        if self.current_key == key:
            return

        # Release previous movement key.
        if self.current_key is not None:
            self.release(self.current_key)

        self._send_key(
            "keyDown",
            key_name,
            key_code,
        )

        self.current_key = key

    def release(self, key):
        """Release a movement key."""

        key_codes = {
            "right": ("ArrowRight", 39),
            "left": ("ArrowLeft", 37),
        }

        if key not in key_codes:
            return

        key_name, key_code = key_codes[key]

        self._send_key(
            "keyUp",
            key_name,
            key_code,
        )

        if self.current_key == key:
            self.current_key = None

    def _brake_pulse(self):
        """Apply a short brake pulse."""

        now = time.monotonic()

        if (
            now - self.last_brake_time
            < self.brake_cooldown_seconds
        ):
            return

        self._send_key(
            "keyDown",
            "ArrowLeft",
            37,
        )

        time.sleep(self.brake_pulse_seconds)

        self._send_key(
            "keyUp",
            "ArrowLeft",
            37,
        )

        self.last_brake_time = time.monotonic()

    def update_action(self, gesture, tilt_angle=0.0):
        """Convert gesture into game control."""

        valid_gestures = {
            "ACCELERATE",
            "BRAKE",
            "NEUTRAL",
        }

        if gesture not in valid_gestures:
            self.release_all_keys()
            return None

        # Gesture stability
        if gesture == self.pending_gesture:
            self.pending_count += 1
        else:
            self.pending_gesture = gesture
            self.pending_count = 1

        if self.pending_count < self.stable_frames:
            return self.current_key

        # ACCELERATE
        if gesture == "ACCELERATE":
            self.press("right")
            self.active_gesture = "ACCELERATE"

        # BRAKE
        elif gesture == "BRAKE":
            # Release acceleration first.
            if self.current_key == "right":
                self.release("right")

            self._brake_pulse()
            self.active_gesture = "BRAKE"

        # NEUTRAL
        elif gesture == "NEUTRAL":
            if self.current_key is not None:
                self.release(self.current_key)

            self.active_gesture = "NEUTRAL"
            self.last_brake_time = 0.0

            return None

        return self.current_key

    def release_all_keys(self):
        """Release any currently held key."""

        if self.current_key is not None:
            self.release(self.current_key)

        self.current_key = None
        self.pending_gesture = None
        self.pending_count = 0
        self.active_gesture = None
        self.last_brake_time = 0.0

    def close(self):
        """Release keys and close Chrome connection."""

        self.release_all_keys()

        if self.websocket is not None:
            self.websocket.close()
            self.websocket = None