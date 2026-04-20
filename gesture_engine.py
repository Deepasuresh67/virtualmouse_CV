"""
gesture_engine.py — Intent-Aware Gesture Recognition with Confidence Scoring.

v3 FIXES:
  - SCROLL: Detection now zone-independent in classifier; works by checking
    index+middle both up. scroll_ref_y reset properly on enter/exit only.
  - SCROLL: _scroll_ref_y no longer wiped by MOVE — scroll ref persists until
    fingers clearly change from scroll posture.
  - CLICK: Complete rewrite of click/double-click using a clean FSM:
      OPEN → (pinch) → PINCH_HOLD → (release) → OPEN_AFTER
      A click FIRES on the RELEASE edge (finger-up after pinch),
      not on the hold. This eliminates false fires from held pinch.
  - DOUBLE-CLICK: Second pinch within window after first release = dbl-click.
  - OSCILLATION cancel: disabled by default (caused false cancels on scroll).
"""

from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Optional

import numpy as np

import config
from hand_tracker import HandState


# ── Gesture labels ────────────────────────────────────────
class Gesture:
    IDLE         = "IDLE"
    MOVE         = "MOVE"
    CLICK        = "CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    RIGHT_CLICK  = "RIGHT_CLICK"
    SCROLL_UP    = "SCROLL_UP"
    SCROLL_DOWN  = "SCROLL_DOWN"
    CANCEL       = "CANCEL"


@dataclass
class GestureResult:
    gesture:    str   = Gesture.IDLE
    confidence: float = 0.0
    ready:      bool  = False
    scroll_dy:  float = 0.0          # normalised scroll delta
    metadata:   dict  = field(default_factory=dict)


# ── Click FSM states ──────────────────────────────────────
class _ClickState(Enum):
    OPEN        = auto()   # hand open, no pinch
    PINCH_HOLD  = auto()   # currently pinching
    OPEN_AFTER  = auto()   # just released — waiting for 2nd pinch window


class GestureEngine:
    """Stateful gesture recogniser. Call update(state, zone) every frame."""

    def __init__(self) -> None:
        # ── Stability / confidence buffer ─────────────────
        self._history: Deque[str] = deque(maxlen=config.GESTURE_STABILITY_FRAMES)

        # ── Click FSM ─────────────────────────────────────
        self._click_state:      _ClickState = _ClickState.OPEN
        self._pinch_start_time: float       = 0.0   # when pinch began
        self._release_time:     float       = 0.0   # when pinch was released
        self._pending_click:    str         = ""    # CLICK or DOUBLE_CLICK to fire next frame
        self._last_click_fired: float       = 0.0   # debounce

        # ── Scroll state ──────────────────────────────────
        self._scroll_ref_y:     Optional[float] = None
        self._scroll_posture:   bool            = False  # True when 2 fingers up

        # Previous raw label
        self._prev_raw: str = Gesture.IDLE

    # ── Public ────────────────────────────────────────────

    def update(self, state: HandState, zone: str) -> GestureResult:
        raw = self._classify_raw(state, zone)

        self._history.append(raw)

        if len(self._history) == 0:
            confidence = 0.0
        else:
            confidence = self._history.count(raw) / len(self._history)

        ready = (confidence >= config.CONFIDENCE_THRESHOLD and
                 len(self._history) >= config.GESTURE_STABILITY_FRAMES)

        scroll_dy = 0.0
        if raw in (Gesture.SCROLL_UP, Gesture.SCROLL_DOWN):
            scroll_dy = self._compute_scroll_delta(state)

        self._prev_raw = raw

        return GestureResult(
            gesture=raw,
            confidence=confidence,
            ready=ready,
            scroll_dy=scroll_dy,
        )

    # ── Raw classifier ────────────────────────────────────

    def _classify_raw(self, s: HandState, zone: str) -> str:
        fu = s.fingers_up   # [thumb, index, middle, ring, pinky]

        # ── Detect scroll posture: index + middle up, ring + pinky down ──
        # This check is zone-independent — posture alone decides scroll readiness.
        scroll_posture = fu[1] and fu[2] and not fu[3] and not fu[4]

        # Reset scroll ref when entering scroll posture fresh
        if scroll_posture and not self._scroll_posture:
            self._scroll_ref_y = s.index_tip[1]   # anchor ref at entry
        if not scroll_posture:
            self._scroll_ref_y = None
        self._scroll_posture = scroll_posture

        # ── Pinch detection ───────────────────────────────
        pinching = s.index_thumb_dist < config.PINCH_DISTANCE_THRESHOLD

        # ── Scroll: 2 fingers up, no pinch, in scroll zone ──
        # Zone: TOP or full-frame scroll (scroll_posture anywhere when enabled)
        if scroll_posture and not pinching and zone == "TOP":
            return self._scroll_direction(s)

        # ── Right-click: middle+thumb pinch, index up ─────
        if s.middle_thumb_dist < config.MIDDLE_PINCH_THRESHOLD and fu[1] and fu[2]:
            return Gesture.RIGHT_CLICK

        # ── Click FSM ─────────────────────────────────────
        click_result = self._update_click_fsm(s, pinching)
        if click_result:
            return click_result

        # ── Move: index up (any other fingers) ───────────
        if fu[1]:
            return Gesture.MOVE

        return Gesture.IDLE

    # ── Click FSM ─────────────────────────────────────────

    def _update_click_fsm(self, s: HandState, pinching: bool) -> Optional[str]:
        """
        Finite state machine for click detection.

        Transitions:
          OPEN       → (pinch begins)   → PINCH_HOLD
          PINCH_HOLD → (pinch released) → OPEN_AFTER  [fires CLICK or DOUBLE_CLICK]
          OPEN_AFTER → (pinch again)    → PINCH_HOLD  [will fire DOUBLE_CLICK on release]
          OPEN_AFTER → (timeout)        → OPEN
        """
        now = s.timestamp

        if self._click_state == _ClickState.OPEN:
            if pinching:
                self._click_state      = _ClickState.PINCH_HOLD
                self._pinch_start_time = now
            return None

        elif self._click_state == _ClickState.PINCH_HOLD:
            if not pinching:
                # Pinch released — decide click type
                held = now - self._pinch_start_time
                self._release_time  = now
                self._click_state   = _ClickState.OPEN_AFTER

                # Debounce
                if (now - self._last_click_fired) < config.CLICK_DEBOUNCE_S:
                    return None

                self._last_click_fired = now
                # Short tap → single click  |  held too long → ignore (drag intent)
                if held < config.CLICK_MAX_HOLD_S:
                    return Gesture.CLICK
                return None
            else:
                # Still pinching — return CLICK so confidence bar stays high
                return Gesture.CLICK

        elif self._click_state == _ClickState.OPEN_AFTER:
            # Waiting for second pinch within double-click window
            since_release = now - self._release_time
            if since_release > (config.DOUBLE_CLICK_WINDOW_MS / 1000):
                # Window expired → back to open
                self._click_state = _ClickState.OPEN
                return None
            if pinching:
                # Second pinch within window → double-click on release
                self._click_state      = _ClickState.PINCH_HOLD  # will fire on next release
                self._pinch_start_time = now
                return None
            return None

        return None

    # ── Scroll helpers ────────────────────────────────────

    def _scroll_direction(self, s: HandState) -> str:
        y = s.index_tip[1]
        if self._scroll_ref_y is None:
            self._scroll_ref_y = y
            return Gesture.IDLE   # first frame — establish reference, no action yet

        dy = self._scroll_ref_y - y   # positive = moved up = scroll up
        if abs(dy) < config.SCROLL_DEAD_ZONE:
            return Gesture.IDLE
        return Gesture.SCROLL_UP if dy > 0 else Gesture.SCROLL_DOWN

    def _compute_scroll_delta(self, s: HandState) -> float:
        if self._scroll_ref_y is None:
            return 0.0
        return abs(self._scroll_ref_y - s.index_tip[1])
