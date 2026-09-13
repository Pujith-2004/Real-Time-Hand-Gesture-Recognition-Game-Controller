"""
Unit tests for Game Controller key mapping and safety release.
"""

import pytest
from src.game_controller import GameController


def test_game_controller_action_mapping():
    mapping = {
        "ACCELERATE": "right",
        "BRAKE": "left",
        "NEUTRAL": None,
    }

    controller = GameController(key_mapping=mapping)

    act_acc = controller.update_action("ACCELERATE")
    assert act_acc == "right"
    assert "right" in controller.currently_pressed_keys

    act_neu = controller.update_action("NEUTRAL")
    assert act_neu is None
    assert len(controller.currently_pressed_keys) == 0


def test_emergency_release_all():
    controller = GameController()

    controller.update_action("BRAKE")
    assert len(controller.currently_pressed_keys) > 0

    controller.release_all_keys()
    assert len(controller.currently_pressed_keys) == 0