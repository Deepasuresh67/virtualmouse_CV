"""
hand_tracker.py — MediaPipe Hands wrapper.

Provides a clean HandState dataclass with:
  - Normalised (0-1) and pixel landmark positions
  - Per-finger extension flags
  - Inter-landmark distances (used by gesture engine)
  - Raw velocity of the index fingertip
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import config

# ── MediaPipe landmark indices ───────────────────────────
WRIST         = 0
THUMB_TIP     = 4
INDEX_MCP     = 5;  INDEX_TIP  = 8
MIDDLE_MCP    = 9;  MIDDLE_TIP = 12
RING_MCP      = 13; RING_TIP   = 16
PINKY_MCP     = 17; PINKY_TIP  = 20
INDEX_PIP     = 6
MIDDLE_PIP    = 10
RING_PIP      = 14
PINKY_PIP     = 18


@dataclass
class HandState:
    """Snapshot of hand data for a single frame."""
    # Raw normalised landmarks [(x, y, z), ...]
    landmarks: List[Tuple[float, float, float]]
    # Pixel-space positions [(px, py), ...]  — for drawing
    pixel_landmarks: List[Tuple[int, int]]
    # Convenience accessors (normalised)
    index_tip:  Tuple[float, float] = (0.0, 0.0)
    middle_tip: Tuple[float, float] = (0.0, 0.0)
    thumb_tip:  Tuple[float, float] = (0.0, 0.0)
    ring_tip:   Tuple[float, float] = (0.0, 0.0)
    pinky_tip:  Tuple[float, float] = (0.0, 0.0)
    # Finger extension (True = finger extended upward)
    fingers_up: List[bool] = field(default_factory=lambda: [False]*5)
    # Key distances (normalised)
    index_thumb_dist:  float = 0.0
    middle_thumb_dist: float = 0.0
    # Velocity of index tip (normalised units / second)
    index_tip_velocity: float = 0.0
    # Wrist position and horizontal velocity (for swipe detection)
    wrist_x:            float = 0.0
    wrist_velocity_x:   float = 0.0
    # Timestamp
    timestamp: float = field(default_factory=time.time)


class HandTracker:
    """Wraps MediaPipe Hands and extracts structured HandState per frame."""

    # Connections to draw (pairs of landmark indices)
    _CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

    def __init__(self) -> None:
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_HANDS,
            min_detection_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )
        self._prev_index_tip: Optional[Tuple[float, float]] = None
        self._prev_wrist_x:   Optional[float]               = None
        self._prev_time: float = time.time()

    # ── Public ───────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray) -> Optional[HandState]:
        """
        Process a BGR frame and return HandState or None if no hand detected.
        """
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mp_hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            self._prev_index_tip = None
            return None

        hand_lms = results.multi_hand_landmarks[0]

        # Build lists
        landmarks: List[Tuple[float, float, float]] = [
            (lm.x, lm.y, lm.z) for lm in hand_lms.landmark
        ]
        pixel_landmarks: List[Tuple[int, int]] = [
            (int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark
        ]

        # Convenience tip positions
        index_tip  = (landmarks[INDEX_TIP][0],  landmarks[INDEX_TIP][1])
        middle_tip = (landmarks[MIDDLE_TIP][0], landmarks[MIDDLE_TIP][1])
        thumb_tip  = (landmarks[THUMB_TIP][0],  landmarks[THUMB_TIP][1])
        ring_tip   = (landmarks[RING_TIP][0],   landmarks[RING_TIP][1])
        pinky_tip  = (landmarks[PINKY_TIP][0],  landmarks[PINKY_TIP][1])

        # ── Finger extension ─────────────────────────────
        fingers_up = self._compute_fingers_up(landmarks)

        # ── Key distances ────────────────────────────────
        index_thumb_dist  = self._dist(index_tip,  thumb_tip)
        middle_thumb_dist = self._dist(middle_tip, thumb_tip)

        # ── Index tip velocity (normalised units / second) ─
        now = time.time()
        dt  = max(now - self._prev_time, 1e-6)
        if self._prev_index_tip is not None:
            dx = index_tip[0] - self._prev_index_tip[0]
            dy = index_tip[1] - self._prev_index_tip[1]
            velocity = np.hypot(dx, dy) / dt
        else:
            velocity = 0.0
        self._prev_index_tip = index_tip

        # ── Wrist horizontal velocity (for swipe) ────────────
        wrist_x = landmarks[WRIST][0]
        if self._prev_wrist_x is not None:
            wrist_velocity_x = (wrist_x - self._prev_wrist_x) / dt
        else:
            wrist_velocity_x = 0.0
        self._prev_wrist_x = wrist_x
        self._prev_time    = now

        return HandState(
            landmarks=landmarks,
            pixel_landmarks=pixel_landmarks,
            index_tip=index_tip,
            middle_tip=middle_tip,
            thumb_tip=thumb_tip,
            ring_tip=ring_tip,
            pinky_tip=pinky_tip,
            fingers_up=fingers_up,
            index_thumb_dist=index_thumb_dist,
            middle_thumb_dist=middle_thumb_dist,
            index_tip_velocity=velocity,
            wrist_x=wrist_x,
            wrist_velocity_x=wrist_velocity_x,
            timestamp=now,
        )

    def draw_skeleton(self, frame: np.ndarray, state: HandState) -> None:
        """Draw hand skeleton on frame (in-place)."""
        h, w = frame.shape[:2]
        px = state.pixel_landmarks

        # Draw connections
        for start_idx, end_idx in self._CONNECTIONS:
            cv2.line(frame, px[start_idx], px[end_idx],
                     config.SKELETON_COLOR, 1, cv2.LINE_AA)

        # Draw landmarks
        for i, (x, y) in enumerate(px):
            is_tip = i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
            color  = config.TIP_COLOR if is_tip else config.LANDMARK_COLOR
            radius = 6 if is_tip else 3
            cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

    def release(self) -> None:
        self._mp_hands.close()

    # ── Private helpers ───────────────────────────────────

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    @staticmethod
    def _compute_fingers_up(lms: List[Tuple[float, float, float]]) -> List[bool]:
        """
        Returns [thumb, index, middle, ring, pinky] extension booleans.
        Uses y-coordinate comparison (smaller y = higher on screen).
        """
        tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        pips = [2,         INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
        up   = []
        for tip_i, pip_i in zip(tips, pips):
            # For thumb: compare x instead of y (works for both hands approx.)
            if tip_i == THUMB_TIP:
                up.append(abs(lms[tip_i][0] - lms[WRIST][0]) >
                          abs(lms[pip_i][0] - lms[WRIST][0]))
            else:
                up.append(lms[tip_i][1] < lms[pip_i][1])
        return up
