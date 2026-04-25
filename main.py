"""
main.py — Modular Pipeline Entry Point for GestureFlow Virtual Mouse.

Pipeline (per frame):
  1. Capture BGR frame from webcam
  2. Flip (mirror)
  3. Run MediaPipe hand tracker → HandState
  4. Detect current spatial zone
  5. Run gesture engine → GestureResult (full 8-gesture set)
  6. Cursor controller dispatches mouse actions
  7. Zone overlay + skeleton + HUD rendered on frame
  8. Display via OpenCV window

Press Q or ESC to quit.
"""

from __future__ import annotations
import sys
import time

import cv2

import config
from hand_tracker      import HandTracker
from gesture_engine    import GestureEngine, GestureResult, Gesture
from cursor_controller import CursorController
from zone_manager      import ZoneManager
from visual_feedback   import VisualFeedback


def main() -> None:
    # ── Initialise subsystems ─────────────────────────────
    tracker    = HandTracker()
    engine     = GestureEngine()
    controller = CursorController()
    zones      = ZoneManager()
    hud        = VisualFeedback()

    # ── Open webcam ───────────────────────────────────────
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {config.CAMERA_INDEX}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)     # minimise latency

    print("═" * 55)
    print("  GestureFlow Virtual Mouse  —  8 Gestures")
    print("  Press  Q / ESC  to quit.")
    print("═" * 55)

    # ── Main loop ─────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed — retrying …")
            time.sleep(0.01)
            continue

        if config.FLIP_FRAME:
            frame = cv2.flip(frame, 1)

        # ── Hand tracking ──────────────────────────────────
        state = tracker.process(frame)

        if state is not None:
            # ── Zone detection ─────────────────────────────
            tip_x, tip_y = state.index_tip
            zone = zones.update(tip_x, tip_y)

            # ── Gesture recognition ────────────────────────
            result = engine.update(state, zone)

            # ── Mouse / scroll / nav dispatch ──────────────
            controller.process(state, result, zone)

            # ── Build debug info for HUD ───────────────────
            debug_info = {
                "fingers_up" : state.fingers_up,
                "pinch_dist" : result.pinch_dist,
                "cursor_x"   : controller.cursor_x,
                "cursor_y"   : controller.cursor_y,
                "tip_x"      : tip_x,
                "tip_y"      : tip_y,
                "zone"       : zone,
                "scroll_dy"  : result.scroll_dy,
                "drag_state" : result.drag_state,
            }

            # ── Drawing ────────────────────────────────────
            zones.draw_zones(frame)
            tracker.draw_skeleton(frame, state)
            hud.render(frame, result, zone, controller.current_alpha,
                       debug_info=debug_info)
        else:
            # No hand — show idle HUD
            idle_result = GestureResult(gesture=Gesture.IDLE,
                                        confidence=0.0, ready=False)
            zones.draw_zones(frame)
            hud.render(frame, idle_result,
                       zones.current_zone, config.SMOOTH_ALPHA_MIN)

        # ── Show frame ─────────────────────────────────────
        cv2.imshow("GestureFlow Virtual Mouse  [Q=quit]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # Q or ESC
            break

    # ── Cleanup ───────────────────────────────────────────
    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    print("[INFO] GestureFlow stopped.")


if __name__ == "__main__":
    main()
