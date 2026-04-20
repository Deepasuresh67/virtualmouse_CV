"""
cursor_controller.py — Adaptive Smoothing & Mouse Action Dispatch.

v3 FIXES:
  - Cursor always moves (zone-independent movement update)
  - Click fires from gesture engine on RELEASE edge (FSM-driven) — no hold repeat
  - Scroll uses continuous delta, not confidence gate alone
  - Scroll throttle reduced to 80ms for more responsive feel
  - _last_gesture reset on IDLE so next gesture transition always fires
"""

from __future__ import annotations
import time
from typing import Optional, Tuple

import numpy as np
import pyautogui

import config
from gesture_engine import Gesture, GestureResult
from hand_tracker import HandState

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.0


def _get_screen_size() -> Tuple[int, int]:
    try:
        from screeninfo import get_monitors
        m = get_monitors()[0]
        return m.width, m.height
    except Exception:
        return pyautogui.size()


class CursorController:
    """Translates HandState + GestureResult into actual mouse actions."""

    def __init__(self) -> None:
        self._screen_w, self._screen_h = _get_screen_size()
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None
        self._last_scroll_time: float   = 0.0
        self._last_action_gesture: str  = Gesture.IDLE  # tracks fired actions only

        self.current_alpha: float = config.SMOOTH_ALPHA_MAX
        self.cursor_x: int = 0
        self.cursor_y: int = 0

    # ── Public entry ──────────────────────────────────────

    def process(self, state: HandState, result: GestureResult,
                zone: str) -> None:
        """
        Called every frame.
        Cursor movement: ALWAYS (zone-independent).
        Actions: depend on zone + gesture.
        """
        # 1. Always move cursor to index fingertip
        self._update_smooth_cursor(state)

        # 2. Dispatch zone actions
        if zone == "TOP":
            self._handle_scroll(result)
        elif zone in ("LEFT", "RIGHT"):
            self._handle_navigation(result, zone)
        else:
            # CENTER / BOTTOM — click actions
            self._handle_clicks(result)

    def get_smooth_position(self) -> Tuple[Optional[float], Optional[float]]:
        return self._smooth_x, self._smooth_y

    # ── Cursor (always runs) ──────────────────────────────

    def _update_smooth_cursor(self, state: HandState) -> None:
        alpha = self._compute_alpha(state.index_tip_velocity)
        self.current_alpha = alpha

        rx, ry = self._map_to_screen(state.index_tip)

        if self._smooth_x is None:
            self._smooth_x, self._smooth_y = rx, ry
        else:
            self._smooth_x = alpha * rx + (1 - alpha) * self._smooth_x
            self._smooth_y = alpha * ry + (1 - alpha) * self._smooth_y

        sx = int(np.clip(self._smooth_x, 0, self._screen_w - 1))
        sy = int(np.clip(self._smooth_y, 0, self._screen_h - 1))
        pyautogui.moveTo(sx, sy)
        self.cursor_x = sx
        self.cursor_y = sy

    # ── Click / Double / Right-click ──────────────────────

    def _handle_clicks(self, result: GestureResult) -> None:
        """
        Fire click on the gesture engine's RELEASE edge.
        The FSM in gesture_engine emits CLICK/DOUBLE_CLICK for exactly one frame
        on release, then transitions back to IDLE. We fire on state ENTRY only.
        """
        if not result.ready:
            # Only reset the action tracker on genuine IDLE (hand open)
            if result.gesture == Gesture.IDLE:
                self._last_action_gesture = Gesture.IDLE
            return

        # Prevent re-firing if we see the same action multiple frames
        if result.gesture == self._last_action_gesture:
            return

        if result.gesture == Gesture.CLICK:
            print(f"[CLICK]  @ ({self.cursor_x},{self.cursor_y})")
            pyautogui.click()
            self._last_action_gesture = Gesture.CLICK

        elif result.gesture == Gesture.DOUBLE_CLICK:
            print(f"[DBLCLK] @ ({self.cursor_x},{self.cursor_y})")
            pyautogui.doubleClick()
            self._last_action_gesture = Gesture.DOUBLE_CLICK

        elif result.gesture == Gesture.RIGHT_CLICK:
            print(f"[RCLICK] @ ({self.cursor_x},{self.cursor_y})")
            pyautogui.rightClick()
            self._last_action_gesture = Gesture.RIGHT_CLICK

    # ── Scroll ────────────────────────────────────────────

    def _handle_scroll(self, result: GestureResult) -> None:
        """
        Scroll fires on continuous SCROLL_UP / SCROLL_DOWN gestures.
        We don't gate on result.ready alone — the gesture engine already
        handles the dead zone. We just throttle the pyautogui call.
        """
        if result.gesture not in (Gesture.SCROLL_UP, Gesture.SCROLL_DOWN):
            return

        now = time.time()
        if now - self._last_scroll_time < 0.08:   # ~12 Hz max scroll rate
            return
        self._last_scroll_time = now

        direction = 1 if result.gesture == Gesture.SCROLL_UP else -1
        # Scale scroll amount by how far hand moved from reference
        dy_norm  = max(result.scroll_dy, 0.005)  # at least a tiny amount
        amount   = max(1, int(dy_norm * 80 * config.SCROLL_SENSITIVITY))
        amount   = min(amount, 15)  # cap max scroll per tick

        print(f"[SCROLL {'UP' if direction>0 else 'DN'}] dy={result.scroll_dy:.3f} amt={amount}")
        pyautogui.scroll(direction * amount)

    # ── Navigation ────────────────────────────────────────

    def _handle_navigation(self, result: GestureResult, zone: str) -> None:
        if not result.ready:
            return
        if result.gesture != Gesture.CLICK:
            return
        if result.gesture == self._last_action_gesture:
            return
        if zone == "LEFT":
            pyautogui.hotkey("alt", "left")
        elif zone == "RIGHT":
            pyautogui.hotkey("alt", "right")
        self._last_action_gesture = result.gesture

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _compute_alpha(velocity_norm: float) -> float:
        v_px  = float(np.clip(velocity_norm * config.FRAME_WIDTH,
                              0, config.SMOOTH_VELOCITY_MAX))
        t     = v_px / config.SMOOTH_VELOCITY_MAX
        alpha = config.SMOOTH_ALPHA_MIN + t * (config.SMOOTH_ALPHA_MAX -
                                               config.SMOOTH_ALPHA_MIN)
        return float(np.clip(alpha, config.SMOOTH_ALPHA_MIN,
                             config.SMOOTH_ALPHA_MAX))

    def _map_to_screen(self, tip: Tuple[float, float]) -> Tuple[float, float]:
        m  = config.ACTIVE_AREA_MARGIN
        nx = (tip[0] - m) / (1 - 2 * m)
        ny = (tip[1] - m) / (1 - 2 * m)
        nx = float(np.clip(nx, 0.0, 1.0))
        ny = float(np.clip(ny, 0.0, 1.0))
        return nx * self._screen_w, ny * self._screen_h
