"""
visual_feedback.py — Real-Time HUD Overlay Renderer.

Draws the following on every frame:
  • Hand skeleton (via hand_tracker)
  • Active zone badge  (top-left)
  • Gesture name       (top-left, below zone)
  • Confidence bar     (colour-coded: red → yellow → green)
  • Smoothing level    (alpha indicator)
  • FPS counter        (top-right)
  • Help legend        (bottom-left, small text)
"""

from __future__ import annotations
import time
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

import config
from gesture_engine import GestureResult


# Google-Noto style: use cv2's built-in font (no TTF dependency needed)
FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


class VisualFeedback:
    """Stateful renderer — tracks FPS internally."""

    def __init__(self) -> None:
        self._frame_times: Deque[float] = deque(maxlen=30)

    # ── Main render entry ─────────────────────────────────

    def render(self,
               frame: np.ndarray,
               result: GestureResult,
               zone: str,
               alpha: float,
               debug_info: dict = None) -> None:
        """Draw all HUD elements onto *frame* in-place."""
        self._frame_times.append(time.time())

        h, w = frame.shape[:2]
        self._draw_zone_badge(frame, zone)
        self._draw_gesture_panel(frame, result)
        self._draw_confidence_bar(frame, result.confidence, w)
        self._draw_smoothing_indicator(frame, alpha, h)
        self._draw_fps(frame, w)
        self._draw_legend(frame, h)
        if debug_info:
            self._draw_debug_panel(frame, debug_info)

    # ── Zone Badge ────────────────────────────────────────

    @staticmethod
    def _draw_zone_badge(frame: np.ndarray, zone: str) -> None:
        color  = config.ZONE_COLORS.get(zone, (120, 120, 120))
        label  = f"  ZONE: {zone}  "
        (tw, th), bl = cv2.getTextSize(label, FONT, config.HUD_FONT_SCALE,
                                       config.HUD_THICKNESS)
        x, y = 16, 40
        # Background pill
        cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + bl + 2),
                      color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + bl + 2),
                      (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, label, (x, y), FONT,
                    config.HUD_FONT_SCALE, (255, 255, 255),
                    config.HUD_THICKNESS, cv2.LINE_AA)

    # ── Gesture Panel ─────────────────────────────────────

    @staticmethod
    def _draw_gesture_panel(frame: np.ndarray, result: GestureResult) -> None:
        # NOTE: cv2.putText only supports ASCII — no Unicode symbols
        label = result.gesture if result.gesture != "IDLE" else "IDLE"
        ready_mark = " [OK]" if result.ready else ""
        text = f"  {label}{ready_mark}  "

        (tw, th), bl = cv2.getTextSize(text, FONT, config.HUD_FONT_SCALE,
                                       config.HUD_THICKNESS)
        x, y = 16, 90
        bg = (30, 30, 30)
        cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + bl + 2),
                      bg, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + bl + 2),
                      (100, 100, 100), 1, cv2.LINE_AA)
        fg = (0, 220, 80) if result.ready else (200, 200, 200)
        cv2.putText(frame, text, (x, y), FONT,
                    config.HUD_FONT_SCALE, fg,
                    config.HUD_THICKNESS, cv2.LINE_AA)

    # ── Confidence Bar ────────────────────────────────────

    @staticmethod
    def _draw_confidence_bar(frame: np.ndarray, confidence: float,
                             frame_w: int) -> None:
        bar_x, bar_y = 16, 110
        bar_w = 220
        bar_h = 10
        filled_w = int(bar_w * confidence)

        # Determine bar colour
        if confidence < 0.4:
            color = config.CONF_LOW_COLOR
        elif confidence < 0.75:
            color = config.CONF_MID_COLOR
        else:
            color = config.CONF_HIGH_COLOR

        # Background track
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      (50, 50, 50), -1, cv2.LINE_AA)
        # Filled portion
        if filled_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + filled_w, bar_y + bar_h),
                          color, -1, cv2.LINE_AA)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      (150, 150, 150), 1, cv2.LINE_AA)
        # Label
        pct_text = f"Conf: {int(confidence * 100)}%"
        cv2.putText(frame, pct_text, (bar_x + bar_w + 8, bar_y + bar_h),
                    FONT_SMALL, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Smoothing Indicator ───────────────────────────────

    @staticmethod
    def _draw_smoothing_indicator(frame: np.ndarray, alpha: float,
                                  frame_h: int) -> None:
        x, y = 16, 136
        label = f"Smooth: {'Precision' if alpha < 0.25 else 'Responsive'}"
        bar_w = int(alpha / config.SMOOTH_ALPHA_MAX * 100)
        cv2.putText(frame, label, (x, y), FONT_SMALL,
                    0.42, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y + 4), (x + 100, y + 10),
                      (50, 50, 50), -1, cv2.LINE_AA)
        if bar_w > 0:
            cv2.rectangle(frame, (x, y + 4), (x + bar_w, y + 10),
                          (0, 180, 255), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y + 4), (x + 100, y + 10),
                      (120, 120, 120), 1, cv2.LINE_AA)

    # ── FPS ───────────────────────────────────────────────

    def _draw_fps(self, frame: np.ndarray, frame_w: int) -> None:
        if len(self._frame_times) < 2:
            fps = 0.0
        else:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            fps     = (len(self._frame_times) - 1) / max(elapsed, 1e-6)
        text = f"FPS: {fps:.1f}"
        (tw, _), _ = cv2.getTextSize(text, FONT_SMALL, 0.55, 1)
        cv2.putText(frame, text, (frame_w - tw - 16, 30),
                    FONT_SMALL, 0.55, (0, 255, 180), 1, cv2.LINE_AA)

    # ── Legend ────────────────────────────────────────────

    @staticmethod
    def _draw_legend(frame: np.ndarray, frame_h: int) -> None:
        lines = [
            "1-finger=MOVE | Pinch=CLICK | Fast-pinch=DBLCLICK",
            "Mid+Thumb=RCLICK | 2-fin(TOP zone)=SCROLL",
            "Press Q to quit",
        ]
        x = 16
        for i, line in enumerate(lines):
            y = frame_h - 16 - (len(lines) - 1 - i) * 18
            cv2.putText(frame, line, (x, y), FONT_SMALL,
                        0.38, (140, 140, 140), 1, cv2.LINE_AA)

    # ── Debug Panel ───────────────────────────────────────

    @staticmethod
    def _draw_debug_panel(frame: np.ndarray, info: dict) -> None:
        """
        Draws a semi-transparent debug panel on the right side showing:
        - Finger states (T I M R P)
        - Pinch distance
        - Cursor screen coordinates
        """
        h, w = frame.shape[:2]
        panel_x = w - 230
        panel_y = 50
        panel_w = 220
        panel_h = 145

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x - 5, panel_y - 5),
                      (panel_x + panel_w, panel_y + panel_h),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel_x - 5, panel_y - 5),
                      (panel_x + panel_w, panel_y + panel_h),
                      (80, 80, 80), 1)

        # Title
        cv2.putText(frame, "DEBUG", (panel_x, panel_y + 14),
                    FONT_SMALL, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        # Finger states — T I M R P circles (green=up, red=down)
        fu = info.get("fingers_up", [False]*5)
        labels = ["T", "I", "M", "R", "P"]
        for i, (lbl, st) in enumerate(zip(labels, fu)):
            color = (0, 220, 80) if st else (0, 50, 200)
            cx2 = panel_x + i * 42
            cv2.circle(frame, (cx2 + 12, panel_y + 38), 11, color, -1)
            cv2.circle(frame, (cx2 + 12, panel_y + 38), 11, (255,255,255), 1)
            cv2.putText(frame, lbl, (cx2 + 7, panel_y + 43),
                        FONT_SMALL, 0.44, (255,255,255), 1, cv2.LINE_AA)

        # Pinch distance + visual bar
        dist   = info.get("pinch_dist", 0.0)
        thresh = info.get("pinch_thresh", 0.065)
        dc = (0, 220, 80) if dist < thresh else (150, 150, 150)
        cv2.putText(frame, f"Pinch:{dist:.3f}  th:{thresh:.3f}",
                    (panel_x, panel_y + 68), FONT_SMALL, 0.39, dc, 1, cv2.LINE_AA)
        bar_max = 0.20
        bw  = int(min(dist / bar_max, 1.0) * panel_w)
        tw2 = int(thresh / bar_max * panel_w)
        cv2.rectangle(frame, (panel_x, panel_y + 72),
                      (panel_x + panel_w, panel_y + 79), (40,40,40), -1)
        cv2.rectangle(frame, (panel_x, panel_y + 72),
                      (panel_x + bw, panel_y + 79), dc, -1)
        cv2.line(frame, (panel_x + tw2, panel_y + 70),
                 (panel_x + tw2, panel_y + 81), (0, 200, 255), 2)

        # Cursor coords
        cx_val = info.get("cursor_x", 0)
        cy_val = info.get("cursor_y", 0)
        cv2.putText(frame, f"Cursor:({cx_val},{cy_val})",
                    (panel_x, panel_y + 97), FONT_SMALL,
                    0.39, (200,200,200), 1, cv2.LINE_AA)

        # Tip norm + current zone
        tx   = info.get("tip_x", 0.0)
        ty   = info.get("tip_y", 0.0)
        zone = info.get("zone", "?")
        cv2.putText(frame, f"Tip:({tx:.2f},{ty:.2f}) Z:{zone}",
                    (panel_x, panel_y + 114), FONT_SMALL,
                    0.37, (160,160,160), 1, cv2.LINE_AA)

        # Scroll dy
        scroll_dy = info.get("scroll_dy", 0.0)
        sc = (0, 200, 255) if abs(scroll_dy) > 0.005 else (70,70,70)
        cv2.putText(frame, f"ScrollDY:{scroll_dy:.3f}",
                    (panel_x, panel_y + 131), FONT_SMALL,
                    0.39, sc, 1, cv2.LINE_AA)
