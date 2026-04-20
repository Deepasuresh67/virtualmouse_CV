"""
zone_manager.py — Spatial Interaction Zone detection.

Divides the normalised camera frame into named zones with hysteresis
(a dead-band) to prevent rapid zone-switching at boundaries.

Zone layout (normalised coords):
  ┌──────────────────────────────┐
  │         TOP  (scroll)        │  y < TOP_THRESH
  ├──────┬───────────────┬───────┤
  │ LEFT │    CENTER     │ RIGHT │  LEFT_THRESH < x < RIGHT_THRESH
  │ (nav)│   (cursor)    │ (nav) │
  ├──────┴───────────────┴───────┤
  │         BOTTOM (reserved)    │  y > BOTTOM_THRESH
  └──────────────────────────────┘
"""

from __future__ import annotations
from typing import Tuple

import config

# All zone names
ZONES = ("CENTER", "TOP", "BOTTOM", "LEFT", "RIGHT")


class ZoneManager:
    """Detects which spatial zone a normalised (x, y) coordinate falls in."""

    def __init__(self) -> None:
        self._current_zone: str = "CENTER"

    # ── Public ───────────────────────────────────────────

    @property
    def current_zone(self) -> str:
        return self._current_zone

    def update(self, x: float, y: float) -> str:
        """
        Given normalised fingertip position (x, y) ∈ [0,1]²,
        return the active zone (with hysteresis applied).
        """
        candidate = self._raw_zone(x, y)
        if candidate != self._current_zone:
            # Only switch if we are far enough into the new zone
            if self._is_committed(x, y, candidate):
                self._current_zone = candidate
        return self._current_zone

    # ── Private ───────────────────────────────────────────

    @staticmethod
    def _raw_zone(x: float, y: float) -> str:
        """Determine zone without hysteresis."""
        if y < config.TOP_ZONE_THRESH:
            return "TOP"
        if y > config.BOTTOM_ZONE_THRESH:
            return "BOTTOM"
        if x < config.LEFT_ZONE_THRESH:
            return "LEFT"
        if x > config.RIGHT_ZONE_THRESH:
            return "RIGHT"
        return "CENTER"

    @staticmethod
    def _is_committed(x: float, y: float, zone: str) -> bool:
        """Return True when position is hysteresis-margin deep into zone."""
        h = config.ZONE_HYSTERESIS
        if zone == "TOP":
            return y < config.TOP_ZONE_THRESH - h
        if zone == "BOTTOM":
            return y > config.BOTTOM_ZONE_THRESH + h
        if zone == "LEFT":
            return x < config.LEFT_ZONE_THRESH - h
        if zone == "RIGHT":
            return x > config.RIGHT_ZONE_THRESH + h
        if zone == "CENTER":
            return (config.LEFT_ZONE_THRESH + h < x < config.RIGHT_ZONE_THRESH - h and
                    config.TOP_ZONE_THRESH  + h < y < config.BOTTOM_ZONE_THRESH  - h)
        return True

    def draw_zones(self, frame, alpha: float = 0.08) -> None:
        """
        Draw a subtle translucent overlay showing zone boundaries on `frame`.
        Only active zone is highlighted.
        """
        import cv2, numpy as np
        h, w = frame.shape[:2]
        overlay = frame.copy()

        zones_rects = {
            "TOP":    (0, 0, w, int(config.TOP_ZONE_THRESH * h)),
            "BOTTOM": (0, int(config.BOTTOM_ZONE_THRESH * h), w, h),
            "LEFT":   (0, int(config.TOP_ZONE_THRESH * h),
                       int(config.LEFT_ZONE_THRESH * w),
                       int(config.BOTTOM_ZONE_THRESH * h)),
            "RIGHT":  (int(config.RIGHT_ZONE_THRESH * w), int(config.TOP_ZONE_THRESH * h),
                       w, int(config.BOTTOM_ZONE_THRESH * h)),
            "CENTER": (int(config.LEFT_ZONE_THRESH * w), int(config.TOP_ZONE_THRESH * h),
                       int(config.RIGHT_ZONE_THRESH * w), int(config.BOTTOM_ZONE_THRESH * h)),
        }

        for zone, (x1, y1, x2, y2) in zones_rects.items():
            color = config.ZONE_COLORS.get(zone, (100, 100, 100))
            a = alpha * 3 if zone == self._current_zone else alpha
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, a, frame, 1 - a, 0, frame)
            overlay = frame.copy()

        # Draw zone boundary lines
        line_color = (80, 80, 80)
        t = int(config.TOP_ZONE_THRESH * h)
        b = int(config.BOTTOM_ZONE_THRESH * h)
        l = int(config.LEFT_ZONE_THRESH * w)
        r = int(config.RIGHT_ZONE_THRESH * w)
        cv2.line(frame, (0, t), (w, t), line_color, 1)
        cv2.line(frame, (0, b), (w, b), line_color, 1)
        cv2.line(frame, (l, t), (l, b), line_color, 1)
        cv2.line(frame, (r, t), (r, b), line_color, 1)
