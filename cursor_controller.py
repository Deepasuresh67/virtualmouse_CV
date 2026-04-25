"""
cursor_controller.py — Adaptive Smoothing & Mouse Action Dispatch (v5).

Handles all 8 gestures:
  MOVE          — EMA-smoothed cursor tracking (velocity-adaptive alpha)
  CLICK         — left click (fires on pinch-release edge)
  DOUBLE_CLICK  — double click (two quick pinches)
  DRAG/DROP     — mouseDown on DRAG entry, mouseUp on DROP
  RIGHT_CLICK   — right click (fires when dwell completes in engine)
  FREEZE        — cursor movement suppressed
  SCROLL_UP/DN  — native WM_MOUSEWHEEL via ctypes
  SWIPE_*       — Alt+Left / Alt+Right hotkeys with cooldown
"""

from __future__ import annotations
import time
from typing import Optional, Tuple

import numpy as np
import pyautogui
import ctypes

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


def _native_scroll(direction: int, notches: int, x: int, y: int) -> None:
    """Native WM_MOUSEWHEEL — works in ALL Windows apps including browsers."""
    ctypes.windll.user32.SetCursorPos(x, y)
    ctypes.windll.user32.mouse_event(
        0x0800, 0, 0, ctypes.c_int(direction * notches * 120), 0)


class CursorController:
    """Translates HandState + GestureResult into actual mouse actions."""

    def __init__(self) -> None:
        self._screen_w, self._screen_h = _get_screen_size()
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None

        self._dragging:          bool  = False
        self._last_scroll_time:  float = 0.0
        self._last_action:       str   = Gesture.IDLE

        self.current_alpha: float = config.SMOOTH_ALPHA_MIN
        self.cursor_x:      int   = 0
        self.cursor_y:      int   = 0

    # ── Public entry ──────────────────────────────────────

    def process(self, state: HandState, result: GestureResult,
                zone: str = "CENTER") -> None:
        """Called every frame. Dispatches mouse actions based on gesture."""
        g = result.gesture

        # ── FREEZE: stop cursor, auto-drop if dragging ────
        if g == Gesture.FREEZE:
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging = False
                print("[DROP]  freeze")
            self._last_action = Gesture.FREEZE
            return  # do NOT move cursor

        # ── Always move cursor (except freeze) ────────────
        self._update_smooth_cursor(state)

        # ── DROP: release mouseDown ───────────────────────
        if g == Gesture.DROP:
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging = False
                print(f"[DROP]  @ ({self.cursor_x},{self.cursor_y})")
            self._last_action = Gesture.IDLE
            return

        # ── DRAG: press mouseDown on first entry ──────────
        if g == Gesture.DRAG:
            if not self._dragging:
                pyautogui.mouseDown()
                self._dragging = True
                print(f"[DRAG]  @ ({self.cursor_x},{self.cursor_y})")
            self._last_action = Gesture.DRAG
            return

        # ── SCROLL ────────────────────────────────────────
        if g in (Gesture.SCROLL_UP, Gesture.SCROLL_DOWN):
            if result.scroll_dy > 0:
                self._handle_scroll(result)
            return

        # ── SWIPE ─────────────────────────────────────────
        if g == Gesture.SWIPE_BACK:
            if self._last_action != Gesture.SWIPE_BACK:
                pyautogui.hotkey("alt", "left")
                print("[NAV]   Browser Back")
                self._last_action = Gesture.SWIPE_BACK
            return

        if g == Gesture.SWIPE_FORWARD:
            if self._last_action != Gesture.SWIPE_FORWARD:
                pyautogui.hotkey("alt", "right")
                print("[NAV]   Browser Forward")
                self._last_action = Gesture.SWIPE_FORWARD
            return

        # ── RIGHT_CLICK ───────────────────────────────────
        if g == Gesture.RIGHT_CLICK:
            if self._last_action != Gesture.RIGHT_CLICK:
                pyautogui.rightClick()
                print(f"[RCLICK] @ ({self.cursor_x},{self.cursor_y})")
                self._last_action = Gesture.RIGHT_CLICK
            return

        # ── CLICK ─────────────────────────────────────────
        if g == Gesture.CLICK:
            if self._last_action != Gesture.CLICK:
                pyautogui.click()
                print(f"[CLICK]  @ ({self.cursor_x},{self.cursor_y})")
                self._last_action = Gesture.CLICK
            return

        # ── DOUBLE CLICK ──────────────────────────────────
        if g == Gesture.DOUBLE_CLICK:
            if self._last_action != Gesture.DOUBLE_CLICK:
                pyautogui.doubleClick()
                print(f"[DBLCLK] @ ({self.cursor_x},{self.cursor_y})")
                self._last_action = Gesture.DOUBLE_CLICK
            return

        # ── MOVE / IDLE — reset dedup tracker ─────────────
        if g in (Gesture.MOVE, Gesture.IDLE, Gesture.RIGHT_DWELL, Gesture.PINCH):
            if g == Gesture.IDLE:
                self._last_action = Gesture.IDLE

    def get_smooth_position(self) -> Tuple[Optional[float], Optional[float]]:
        return self._smooth_x, self._smooth_y

    # ── Cursor smoothing ──────────────────────────────────

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

    # ── Scroll ────────────────────────────────────────────

    def _handle_scroll(self, result: GestureResult) -> None:
        now = time.time()
        if now - self._last_scroll_time < config.SCROLL_COOL:
            return
        self._last_scroll_time = now

        direction = 1 if result.gesture == Gesture.SCROLL_UP else -1
        dy_norm   = max(result.scroll_dy, 0.005)
        notches   = min(max(1, int(dy_norm * 80 * config.SCROLL_SENSITIVITY)), 15)

        print(f"[SCROLL {'UP' if direction > 0 else 'DN'}] "
              f"dy={result.scroll_dy:.3f} notches={notches}")
        _native_scroll(direction, notches, self.cursor_x, self.cursor_y)

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _compute_alpha(velocity_norm: float) -> float:
        v_px  = float(np.clip(velocity_norm * config.FRAME_WIDTH,
                              0, config.SMOOTH_VELOCITY_MAX))
        t     = v_px / config.SMOOTH_VELOCITY_MAX
        alpha = config.SMOOTH_ALPHA_MIN + t * (config.SMOOTH_ALPHA_MAX -
                                               config.SMOOTH_ALPHA_MIN)
        return float(np.clip(alpha, config.SMOOTH_ALPHA_MIN, config.SMOOTH_ALPHA_MAX))

    def _map_to_screen(self, tip: Tuple[float, float]) -> Tuple[float, float]:
        m  = config.ACTIVE_AREA_MARGIN
        nx = (tip[0] - m) / (1 - 2 * m)
        ny = (tip[1] - m) / (1 - 2 * m)
        return (float(np.clip(nx, 0.0, 1.0)) * self._screen_w,
                float(np.clip(ny, 0.0, 1.0)) * self._screen_h)
