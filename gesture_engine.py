"""
gesture_engine.py — Intent-Aware Gesture Recognition (v5 — Full 8-Gesture Set).

Priority order (evaluated top-down per frame):
  1. FREEZE       — all 4 fingers up (open palm)
  2. RIGHT_DWELL  — index+middle+ring up, pinky down, hold RCLICK_HOLD_S
  3. SCROLL       — index+middle up, ring+pinky down  (anchor-based)
  4. SWIPE        — fist + wrist x-velocity threshold
  5. MOVE/CLICK/DRAG — index up, pinch FSM
  6. IDLE
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import config
from hand_tracker import HandState


# ── Gesture label constants ───────────────────────────────────
class Gesture:
    IDLE          = "IDLE"
    MOVE          = "MOVE"
    PINCH         = "PINCH"        # pinch held, waiting for drag/release
    CLICK         = "CLICK"
    DOUBLE_CLICK  = "DOUBLE_CLICK"
    DRAG          = "DRAG"         # dragging in progress
    DROP          = "DROP"         # drag released (one-frame event)
    RIGHT_DWELL   = "RIGHT_DWELL"  # right-click dwell in progress
    RIGHT_CLICK   = "RIGHT_CLICK"  # fired once when dwell completes
    SCROLL_UP     = "SCROLL_UP"
    SCROLL_DOWN   = "SCROLL_DOWN"
    FREEZE        = "FREEZE"
    SWIPE_BACK    = "SWIPE_BACK"
    SWIPE_FORWARD = "SWIPE_FORWARD"


@dataclass
class GestureResult:
    gesture:        str   = Gesture.IDLE
    confidence:     float = 0.0
    ready:          bool  = False
    scroll_dy:      float = 0.0        # normalised scroll delta (>0 when scrolling)
    dwell_progress: float = 0.0        # 0-1 for right-click progress bar
    drag_state:     str   = "READY"    # "READY" | "HOLD" | "DRAG"
    pinch_dist:     float = 0.0        # current thumb-index distance
    flash_text:     str   = ""         # action label for HUD flash
    flash_color:    tuple = field(default_factory=lambda: (0, 255, 200))
    metadata:       dict  = field(default_factory=dict)


class _PinchState(Enum):
    READY = auto()   # hand open — moving cursor
    HOLD  = auto()   # pinching, waiting to see if drag or click
    DRAG  = auto()   # drag active — mouseDown held


class GestureEngine:
    """Stateful gesture recogniser. Call update(state) every frame."""

    def __init__(self) -> None:
        # ── Pinch / click / drag FSM ─────────────────────
        self._pstate:       _PinchState = _PinchState.READY
        self._pstart:       float       = 0.0
        self._last_click:   float       = 0.0
        self._last_release: float       = 0.0

        # ── Right-click dwell ─────────────────────────────
        self._rc_start: float = 0.0
        self._rc_fired: bool  = False

        # ── Scroll anchor ─────────────────────────────────
        self._scroll_anchor: Optional[float] = None
        self._last_scroll:   float           = 0.0

        # ── Swipe cooldown ────────────────────────────────
        self._last_swipe: float = 0.0

        # ── Flash state (persisted across frames) ─────────
        self._flash_text:  str   = ""
        self._flash_color: tuple = (0, 255, 200)
        self._flash_exp:   float = 0.0

    # ── Public ────────────────────────────────────────────

    def update(self, state: HandState, zone: str = "CENTER") -> GestureResult:
        fu  = state.fingers_up   # [thumb, index, middle, ring, pinky]
        iu  = fu[1]              # index up
        mu  = fu[2]              # middle up
        ru  = fu[3]              # ring up
        pu  = fu[4]              # pinky up
        pd  = state.index_thumb_dist
        now = state.timestamp

        # Flash text: carry until expiry
        if now < self._flash_exp:
            ft, fc = self._flash_text, self._flash_color
        else:
            ft, fc = "", (0, 255, 200)

        def _result(**kw) -> GestureResult:
            kw.setdefault("pinch_dist", pd)
            kw.setdefault("drag_state", self._pstate.name)
            kw.setdefault("flash_text", ft)
            kw.setdefault("flash_color", fc)
            return GestureResult(**kw)

        # ── 1. PALM FREEZE — all 4 fingers up ────────────
        if iu and mu and ru and pu:
            self._scroll_anchor = None
            self._rc_start      = 0.0
            # If drag was active when palm appeared, signal DROP
            if self._pstate == _PinchState.DRAG:
                self._pstate = _PinchState.READY
                self._set_flash("DROP (freeze)", (0, 220, 130))
                ft, fc = self._flash_text, self._flash_color
                return _result(gesture=Gesture.DROP, confidence=1.0, ready=True)
            return _result(gesture=Gesture.FREEZE, confidence=1.0, ready=True)

        # ── 2. RIGHT-CLICK DWELL — IMR up, pinky down ────
        if iu and mu and ru and not pu:
            self._scroll_anchor = None
            if self._rc_start == 0.0:
                self._rc_start = now
                self._rc_fired = False
            held     = now - self._rc_start
            progress = min(held / config.RCLICK_HOLD_S, 1.0)
            if held >= config.RCLICK_HOLD_S and not self._rc_fired:
                self._rc_fired = True
                self._rc_start = 0.0
                self._set_flash("RIGHT CLICK", (255, 80, 80))
                ft, fc = self._flash_text, self._flash_color
                return _result(gesture=Gesture.RIGHT_CLICK, confidence=1.0,
                               ready=True, dwell_progress=1.0)
            return _result(gesture=Gesture.RIGHT_DWELL, confidence=progress,
                           ready=False, dwell_progress=progress)
        else:
            if not (iu and mu and ru):
                self._rc_start = 0.0
                self._rc_fired = False

        # ── 3. SCROLL — index+middle up, ring+pinky down ─
        if iu and mu and not ru and not pu:
            sy = state.index_tip[1]
            # Set anchor on first frame of scroll pose
            if self._scroll_anchor is None:
                self._scroll_anchor = sy
            dy = self._scroll_anchor - sy   # +ve = hand moved UP = scroll up
            scroll_dy = 0.0
            gesture   = Gesture.IDLE
            if (abs(dy) > config.SCROLL_DY_MIN and
                    (now - self._last_scroll) > config.SCROLL_COOL):
                direction           = 1 if dy > 0 else -1
                scroll_dy           = abs(dy)
                self._scroll_anchor = sy          # advance anchor
                self._last_scroll   = now
                gesture = Gesture.SCROLL_UP if direction > 0 else Gesture.SCROLL_DOWN
                self._set_flash(
                    f"SCROLL {'UP' if direction > 0 else 'DOWN'}", (180, 80, 255))
                ft, fc = self._flash_text, self._flash_color
            return _result(gesture=gesture, confidence=1.0,
                           ready=scroll_dy > 0, scroll_dy=scroll_dy)
        else:
            self._scroll_anchor = None

        # ── 4. SWIPE — fist + wrist x-velocity ───────────
        if not iu and not mu and not ru and not pu:
            wv      = state.wrist_velocity_x
            gesture = Gesture.IDLE
            if (abs(wv) > config.SWIPE_VEL and
                    (now - self._last_swipe) > config.SWIPE_COOL):
                self._last_swipe = now
                # Frame is mirrored: wv>0 = hand moved right = FORWARD
                if wv > 0:
                    gesture = Gesture.SWIPE_FORWARD
                    self._set_flash("SWIPE FORWARD >>", (255, 200, 0))
                else:
                    gesture = Gesture.SWIPE_BACK
                    self._set_flash("<< SWIPE BACK", (255, 200, 0))
                ft, fc = self._flash_text, self._flash_color
            return _result(gesture=gesture, confidence=1.0 if gesture != Gesture.IDLE else 0.0,
                           ready=gesture != Gesture.IDLE)

        # ── 5. MOVE / CLICK / DRAG — index up ────────────
        if iu and not mu:
            pinching = pd < config.PINCH_CLOSE
            released = pd > config.PINCH_OPEN

            if self._pstate == _PinchState.READY:
                if pinching:
                    self._pstate = _PinchState.HOLD
                    self._pstart = now
                    return _result(gesture=Gesture.PINCH, confidence=1.0,
                                   ready=False, drag_state="HOLD")
                else:
                    return _result(gesture=Gesture.MOVE, confidence=1.0, ready=True)

            elif self._pstate == _PinchState.HOLD:
                held = now - self._pstart
                if pinching and held >= config.DRAG_HOLD_S:
                    self._pstate = _PinchState.DRAG
                    self._set_flash("DRAG START", (0, 200, 255))
                    ft, fc = self._flash_text, self._flash_color
                    return _result(gesture=Gesture.DRAG, confidence=1.0,
                                   ready=True, drag_state="DRAG")
                elif released:
                    self._pstate = _PinchState.READY
                    gesture = Gesture.MOVE  # default if debounced
                    if (now - self._last_click) > config.CLICK_COOL:
                        if (now - self._last_release) < config.DBL_WIN:
                            gesture = Gesture.DOUBLE_CLICK
                            self._set_flash("DOUBLE CLICK", (0, 255, 200))
                            self._last_release = 0.0
                        else:
                            gesture = Gesture.CLICK
                            self._set_flash("CLICK", (0, 255, 200))
                            self._last_release = now
                        self._last_click = now
                    ft, fc = self._flash_text, self._flash_color
                    return _result(gesture=gesture, confidence=1.0,
                                   ready=True, drag_state="READY")
                else:
                    return _result(gesture=Gesture.PINCH, confidence=1.0,
                                   ready=False, drag_state="HOLD",
                                   metadata={"held_s": round(held, 2)})

            elif self._pstate == _PinchState.DRAG:
                if released:
                    self._pstate = _PinchState.READY
                    self._set_flash("DROP!", (0, 220, 130))
                    ft, fc = self._flash_text, self._flash_color
                    return _result(gesture=Gesture.DROP, confidence=1.0,
                                   ready=True, drag_state="READY")
                else:
                    return _result(gesture=Gesture.DRAG, confidence=1.0,
                                   ready=True, drag_state="DRAG")

        # ── 6. IDLE — catch-all ───────────────────────────
        # Auto-drop if pose changed while dragging
        if self._pstate == _PinchState.DRAG:
            self._pstate = _PinchState.READY
            self._set_flash("DROP (pose change)", (0, 220, 130))
            ft, fc = self._flash_text, self._flash_color
            return _result(gesture=Gesture.DROP, confidence=1.0,
                           ready=True, drag_state="READY")

        return _result(gesture=Gesture.IDLE, confidence=0.0)

    # ── Internal ──────────────────────────────────────────

    def _set_flash(self, text: str, color: tuple, duration: float = 1.1) -> None:
        self._flash_text  = text
        self._flash_color = color
        self._flash_exp   = time.time() + duration
