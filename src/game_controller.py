"""
Game Controller Module.

Translates gesture predictions into OS keyboard inputs using pynput.
Includes gesture stability filtering to prevent rapid key switching.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger("GameController")

try:
    from pynput.keyboard import Controller as KeyboardController, Key

    HAS_PYNPUT = True
except (ImportError, Exception) as e:
    logger.warning(
        f"pynput not available or failed to load ({e}). "
        "Controller will run in simulation mode."
    )
    HAS_PYNPUT = False


class GameController:
    """
    Translates gesture names into keyboard controls.

    Gestures:
        ACCELERATE -> Right Arrow
        BRAKE      -> Left Arrow
        NEUTRAL    -> Release keys
    """

    def __init__(
        self,
        key_mapping: Optional[Dict[str, Optional[str]]] = None,
        stable_frames: int = 1,
    ):
        self.key_mapping = key_mapping or {
            "ACCELERATE": "right",
            "BRAKE": "left",
            "NEUTRAL": None,
        }

        self.stable_frames = max(1, int(stable_frames))

        self.currently_pressed_keys = set()

        self.pending_gesture = None
        self.pending_count = 0
        self.active_gesture = None

        self.keyboard = KeyboardController() if HAS_PYNPUT else None

    def _resolve_key(self, key_str: Optional[str]):
        """Map string key name to pynput Key or character."""

        if not key_str or not HAS_PYNPUT:
            return None

        key_str_lower = key_str.lower()

        special_keys = {
            "right": Key.right,
            "left": Key.left,
            "up": Key.up,
            "down": Key.down,
            "space": Key.space,
            "enter": Key.enter,
        }

        if key_str_lower in special_keys:
            return special_keys[key_str_lower]

        return key_str_lower

    def _is_gesture_stable(self, gesture: str) -> bool:
        """Check whether the gesture has remained stable."""

        if gesture == self.pending_gesture:
            self.pending_count += 1
        else:
            self.pending_gesture = gesture
            self.pending_count = 1

        return self.pending_count >= self.stable_frames

    def update_action(self, gesture: str) -> Optional[str]:
        """
        Update keyboard state based on predicted gesture.

        Gestures:
            ACCELERATE
            BRAKE
            NEUTRAL
        """

        if not self._is_gesture_stable(gesture):
            return self._get_active_key()

        if gesture == self.active_gesture:
            return self._get_active_key()

        self.active_gesture = gesture

        target_key_str = self.key_mapping.get(gesture, None)

        # NEUTRAL = release everything.
        if target_key_str is None or gesture == "NEUTRAL":
            self.release_all_keys()
            return None

        # Release keys that are no longer needed.
        keys_to_release = [
            key
            for key in list(self.currently_pressed_keys)
            if key != target_key_str
        ]

        for key_str in keys_to_release:
            self._release_key(key_str)

        # Press target key.
        if target_key_str not in self.currently_pressed_keys:
            self._press_key(target_key_str)

        return target_key_str

    def _get_active_key(self) -> Optional[str]:
        """Return the currently active keyboard action."""

        if not self.currently_pressed_keys:
            return None

        return next(iter(self.currently_pressed_keys))

    def _press_key(self, key_str: str):
        """Press a key safely."""

        key_obj = self._resolve_key(key_str)

        if key_obj and self.keyboard:
            try:
                self.keyboard.press(key_obj)
            except Exception as e:
                logger.warning(
                    f"Error pressing key '{key_str}': {e}"
                )

        self.currently_pressed_keys.add(key_str)

    def _release_key(self, key_str: str):
        """Release a key safely."""

        key_obj = self._resolve_key(key_str)

        if key_obj and self.keyboard:
            try:
                self.keyboard.release(key_obj)
            except Exception as e:
                logger.warning(
                    f"Error releasing key '{key_str}': {e}"
                )

        self.currently_pressed_keys.discard(key_str)

    def release_all_keys(self):
        """Release all currently pressed keys."""

        for key_str in list(self.currently_pressed_keys):
            self._release_key(key_str)

        self.currently_pressed_keys.clear()

        self.pending_gesture = None
        self.pending_count = 0
        self.active_gesture = None