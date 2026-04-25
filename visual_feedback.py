"""
visual_feedback.py — Real-Time HUD Overlay Renderer (v2).

HUD elements drawn on every frame:
  • Dark header strip (top)
  • Zone badge           — active zone name, colour-coded
  • Gesture label        — current gesture, colour-coded, [OK] when ready
  • Flash action text    — last fired action (CLICK, DRAG START, etc.)
  • Pinch distance bar   — thumb-index distance with CLOSE/OPEN markers
  • RC dwell progress bar— fills over 1.4 s for right-click
  • I M R P circles      — green = finger up, blue = down
  • FPS + smoothing alpha
  • Cursor screen coords
  • Gesture legend strip  (bottom)
  • Debug panel           (right side, when debug_info supplied)
"""

from __future__ import annotations
import time
from collections import deque
from typing import Deque

import cv2
import numpy as np

import config
from gesture_engine import Gesture, GestureResult


FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

# Gesture → display colour (BGR)
GESTURE_COLORS = {
    Gesture.MOVE:          (0,   255, 100),
    Gesture.FREEZE:        (0,   220, 255),
    Gesture.SCROLL_UP:     (200,  80, 255),
    Gesture.SCROLL_DOWN:   (200,  80, 255),
    Gesture.DRAG:          (0,   180, 255),
    Gesture.DROP:          (0,   220, 130),
    Gesture.RIGHT_DWELL:   (255,  80,  80),
    Gesture.RIGHT_CLICK:   (255,  80,  80),
    Gesture.CLICK:         (0,   255, 200),
    Gesture.DOUBLE_CLICK:  (0,   255, 200),
    Gesture.SWIPE_BACK:    (255, 200,   0),
    Gesture.SWIPE_FORWARD: (255, 200,   0),
    Gesture.PINCH:         (0,   160, 255),
    Gesture.IDLE:          (150, 150, 150),
}


class VisualFeedback:
    """Stateful renderer — tracks FPS internally."""

    def __init__(self) -> None:
        self._frame_times: Deque[float] = deque(maxlen=60)

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

        # Dark header strip
        cv2.rectangle(frame, (0, 0), (w, 162), (12, 12, 12), -1)

        self._draw_zone_badge(frame, zone)
        self._draw_gesture_label(frame, result)
        self._draw_flash(frame, result)
        self._draw_pinch_bar(frame, result)
        self._draw_rc_dwell_bar(frame, result)
        self._draw_finger_circles(frame, debug_info, w)
        self._draw_fps_alpha(frame, alpha, w)
        self._draw_cursor_coords(frame, debug_info, w)
        self._draw_legend(frame, h, w)

        if debug_info:
            self._draw_debug_panel(frame, debug_info)

    # ── Zone Badge ────────────────────────────────────────

    @staticmethod
    def _draw_zone_badge(frame: np.ndarray, zone: str) -> None:
        color = config.ZONE_COLORS.get(zone, (120, 120, 120))
        label = f"  ZONE: {zone}  "
        (tw, th), bl = cv2.getTextSize(label, FONT, 0.55, 1)
        x, y = 10, 28
        cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + bl + 2), color, -1)
        cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + bl + 2),
                      (255, 255, 255), 1)
        cv2.putText(frame, label, (x, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Gesture Label ─────────────────────────────────────

    @staticmethod
    def _draw_gesture_label(frame: np.ndarray, result: GestureResult) -> None:
        g     = result.gesture
        color = GESTURE_COLORS.get(g, (180, 180, 180))
        tag   = " [OK]" if result.ready else ""
        extra = ""
        if g == Gesture.PINCH and result.metadata.get("held_s"):
            extra = f" {result.metadata['held_s']:.1f}s"
        text = f"Gesture: {g}{extra}{tag}"
        cv2.putText(frame, text, (10, 65), FONT, 0.70, color, 2, cv2.LINE_AA)

    # ── Flash Action Text ─────────────────────────────────

    @staticmethod
    def _draw_flash(frame: np.ndarray, result: GestureResult) -> None:
        if result.flash_text:
            cv2.putText(frame, result.flash_text, (10, 102),
                        FONT, 0.90, result.flash_color, 2, cv2.LINE_AA)

    # ── Pinch Distance Bar ────────────────────────────────

    @staticmethod
    def _draw_pinch_bar(frame: np.ndarray, result: GestureResult) -> None:
        pd    = result.pinch_dist
        BAR   = 220
        bfill = int(min(pd / 0.20, 1.0) * BAR)
        cx_ln = int(config.PINCH_CLOSE / 0.20 * BAR)
        op_ln = int(config.PINCH_OPEN  / 0.20 * BAR)
        x, y  = 10, 118

        # Background track
        cv2.rectangle(frame, (x, y), (x + BAR, y + 11), (40, 40, 40), -1)
        # Fill colour: green=pinching, blue=dragging, grey=open
        if pd < config.PINCH_CLOSE:
            bc = (0, 220, 80)
        elif result.drag_state == "DRAG":
            bc = (0, 160, 255)
        else:
            bc = (100, 100, 180)
        cv2.rectangle(frame, (x, y), (x + bfill, y + 11), bc, -1)
        # Threshold markers
        cv2.line(frame, (x + cx_ln, y - 2), (x + cx_ln, y + 13), (0, 220, 255), 2)
        cv2.line(frame, (x + op_ln, y - 2), (x + op_ln, y + 13), (60,  180,  60), 1)
        cv2.putText(frame, f"Pinch:{pd:.3f}  DRAG>{config.DRAG_HOLD_S}s",
                    (x, y + 25), FONT_SMALL, 0.36, (150, 150, 150), 1, cv2.LINE_AA)

    # ── Right-Click Dwell Bar ─────────────────────────────

    @staticmethod
    def _draw_rc_dwell_bar(frame: np.ndarray, result: GestureResult) -> None:
        if result.dwell_progress <= 0:
            return
        prog = result.dwell_progress
        pw   = int(prog * 160)
        x, y = 10, 150
        cv2.rectangle(frame, (x, y), (x + 160, y + 10), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + pw,  y + 10), (80,  80, 255), -1)
        cv2.putText(frame, "RC dwell", (x + 168, y + 9),
                    FONT_SMALL, 0.36, (180, 80, 255), 1, cv2.LINE_AA)

    # ── Finger Circles I M R P ────────────────────────────

    @staticmethod
    def _draw_finger_circles(frame: np.ndarray, debug_info: dict,
                             frame_w: int) -> None:
        if not debug_info:
            return
        fu     = debug_info.get("fingers_up", [False] * 5)
        labels = ["I", "M", "R", "P"]
        for i, (lbl, st) in enumerate(zip(labels, fu[1:])):   # skip thumb
            col = (0, 220, 80) if st else (0, 50, 180)
            px  = frame_w - 178 + i * 44
            cv2.circle(frame, (px, 28), 14, col, -1)
            cv2.circle(frame, (px, 28), 14, (255, 255, 255), 1)
            cv2.putText(frame, lbl, (px - 5, 33), FONT_SMALL, 0.46,
                        (255, 255, 255), 1)

    # ── FPS + Smoothing Alpha ─────────────────────────────

    def _draw_fps_alpha(self, frame: np.ndarray, alpha: float,
                        frame_w: int) -> None:
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / max(elapsed, 1e-6)
        else:
            fps = 0.0
        text = f"FPS:{fps:.0f}  a:{alpha:.2f}"
        (tw, _), _ = cv2.getTextSize(text, FONT_SMALL, 0.40, 1)
        cv2.putText(frame, text, (frame_w - tw - 10, 55),
                    FONT_SMALL, 0.40, (0, 255, 180), 1, cv2.LINE_AA)

    # ── Cursor Screen Coords ──────────────────────────────

    @staticmethod
    def _draw_cursor_coords(frame: np.ndarray, debug_info: dict,
                            frame_w: int) -> None:
        if not debug_info:
            return
        cx = debug_info.get("cursor_x", 0)
        cy = debug_info.get("cursor_y", 0)
        cv2.putText(frame, f"({cx},{cy})", (frame_w - 115, 75),
                    FONT_SMALL, 0.38, (130, 130, 130), 1, cv2.LINE_AA)

    # ── Legend Strip (bottom) ─────────────────────────────

    @staticmethod
    def _draw_legend(frame: np.ndarray, frame_h: int, frame_w: int) -> None:
        cv2.rectangle(frame, (0, frame_h - 54), (frame_w, frame_h), (10, 10, 10), -1)
        lines = [
            "I=MOVE  Pinch=CLICK  Hold=DRAG  IMR(hold)=RCLICK  IMRP=FREEZE",
            "IM=SCROLL  Fist+Swipe=BROWSER NAV  Q/ESC=quit",
        ]
        for i, line in enumerate(lines):
            y = frame_h - 34 + i * 16
            cv2.putText(frame, line, (5, y), FONT_SMALL, 0.30,
                        (130, 130, 130), 1, cv2.LINE_AA)
        cv2.putText(frame, "GestureFlow Virtual Mouse  v2.0",
                    (5, frame_h - 4), FONT_SMALL, 0.35,
                    (60, 180, 60), 1, cv2.LINE_AA)

    # ── Debug Panel (right side) ──────────────────────────

    @staticmethod
    def _draw_debug_panel(frame: np.ndarray, info: dict) -> None:
        h, w    = frame.shape[:2]
        px      = w - 230
        py      = 170
        pw      = 220
        ph      = 115

        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 5, py - 5),
                      (px + pw, py + ph), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (px - 5, py - 5),
                      (px + pw, py + ph), (80, 80, 80), 1)

        cv2.putText(frame, "DEBUG", (px, py + 14),
                    FONT_SMALL, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        # Pinch distance bar
        dist   = info.get("pinch_dist", 0.0)
        thresh = config.PINCH_CLOSE
        dc     = (0, 220, 80) if dist < thresh else (150, 150, 150)
        cv2.putText(frame, f"Pinch:{dist:.3f}  th:{thresh:.3f}",
                    (px, py + 38), FONT_SMALL, 0.38, dc, 1, cv2.LINE_AA)
        bar_max = 0.20
        bw  = int(min(dist / bar_max, 1.0) * pw)
        tw2 = int(thresh / bar_max * pw)
        cv2.rectangle(frame, (px, py + 42), (px + pw, py + 49), (40, 40, 40), -1)
        cv2.rectangle(frame, (px, py + 42), (px + bw, py + 49), dc, -1)
        cv2.line(frame, (px + tw2, py + 40), (px + tw2, py + 51),
                 (0, 200, 255), 2)

        # Tip position + zone
        tx   = info.get("tip_x", 0.0)
        ty   = info.get("tip_y", 0.0)
        zone = info.get("zone", "?")
        cv2.putText(frame, f"Tip:({tx:.2f},{ty:.2f}) Z:{zone}",
                    (px, py + 66), FONT_SMALL, 0.37,
                    (160, 160, 160), 1, cv2.LINE_AA)

        # Scroll delta
        scroll_dy = info.get("scroll_dy", 0.0)
        sc = (0, 200, 255) if abs(scroll_dy) > 0.005 else (70, 70, 70)
        cv2.putText(frame, f"ScrollDY:{scroll_dy:.3f}",
                    (px, py + 83), FONT_SMALL, 0.38, sc, 1, cv2.LINE_AA)

        # Drag state
        ds = info.get("drag_state", "")
        cv2.putText(frame, f"DragState:{ds}",
                    (px, py + 100), FONT_SMALL, 0.38,
                    (200, 200, 100), 1, cv2.LINE_AA)
