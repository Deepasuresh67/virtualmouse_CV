# -*- coding: utf-8 -*-
"""
GestureFlow Virtual Mouse  —  Self-Contained Entry Point
=========================================================
8 Gestures:
  MOVE        : index finger only        -> cursor follows
  CLICK       : pinch + quick release    -> left click
  DOUBLE-CLICK: two quick pinches        -> double click
  DRAG & DROP : pinch hold >0.5s + move  -> drag; release -> drop
  RIGHT-CLICK : index+middle+ring up, hold 1.4s -> right click (dwell)
  PALM FREEZE : all 4 fingers open       -> cursor locked
  SCROLL      : index+middle (peace sign), move up/down -> scroll
  SWIPE       : closed fist + fast swipe left/right -> browser back/fwd

Run:  python virtual_mouse.py
Quit: Q or ESC
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import ctypes
import time

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.0

# ── Config ────────────────────────────────────────────────
CAM_W, CAM_H   = 640, 480
MARGIN         = 0.15    # active-area crop fraction each edge
SMOOTH_MIN     = 0.30    # EMA alpha at rest (precision)
SMOOTH_MAX     = 0.82    # EMA alpha at speed (responsiveness)
VEL_MAX        = 0.035   # normalised velocity → SMOOTH_MAX

PINCH_CLOSE    = 0.060   # thumb-index dist = pinching
PINCH_OPEN     = 0.095   # thumb-index dist = released (hysteresis)
DRAG_HOLD_S    = 0.50    # pinch hold time before drag starts
CLICK_COOL     = 0.45    # min seconds between clicks
DBL_WIN        = 0.42    # double-click window (seconds)

RCLICK_HOLD_S  = 1.40    # 3-finger dwell seconds for right-click

SCROLL_DY_MIN  = 0.025   # anchor-departure threshold (anchor-based scroll)
SCROLL_COOL    = 0.08    # seconds between scroll ticks (~12 Hz)

SWIPE_VEL      = 0.018   # wrist x-velocity threshold for swipe
SWIPE_COOL     = 1.50    # seconds between swipes

FONT = cv2.FONT_HERSHEY_SIMPLEX
SW, SH = pyautogui.size()

# ── MediaPipe ─────────────────────────────────────────────
MP    = mp.solutions.hands
MPD   = mp.solutions.drawing_utils
hands = MP.Hands(max_num_hands=1,
                 min_detection_confidence=0.78,
                 min_tracking_confidence=0.78)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

# ── State ─────────────────────────────────────────────────
cx, cy       = SW // 2, SH // 2
alpha        = SMOOTH_MIN
prev_tx      = None
prev_ty      = None

# Pinch / Click / Drag  state machine
pstate       = "READY"   # READY | HOLD | DRAG
pstart       = 0.0
last_click   = 0.0
last_release = 0.0

# Right-click dwell
rc_start     = 0.0
rc_fired     = False

# Scroll (anchor-based)
scroll_anchor = None
last_scroll   = 0.0

# Swipe
prev_wx      = None
last_swipe   = 0.0

# Flash overlay
flash_txt    = ""
flash_col    = (0, 255, 200)
flash_exp    = 0.0

frame_times  = []


# ── Helpers ───────────────────────────────────────────────
def fingers_up(lm):
    """Return (index_up, middle_up, ring_up, pinky_up)."""
    return tuple(lm[t].y < lm[p].y
                 for t, p in [(8, 6), (12, 10), (16, 14), (20, 18)])


def pdist(lm):
    """Normalised thumb-index distance."""
    return ((lm[4].x - lm[8].x) ** 2 + (lm[4].y - lm[8].y) ** 2) ** 0.5


def map_xy(v, lo, hi, mx):
    return int(np.clip((v - lo) / (hi - lo) * mx, 0, mx - 1))


def scroll_native(direction, notches, x, y):
    """Native WM_MOUSEWHEEL — works in ALL Windows apps."""
    ctypes.windll.user32.SetCursorPos(int(x), int(y))
    ctypes.windll.user32.mouse_event(
        0x0800, 0, 0, ctypes.c_int(direction * notches * 120), 0)


def flash(txt, col=(0, 255, 200)):
    global flash_txt, flash_col, flash_exp
    flash_txt, flash_col, flash_exp = txt, col, time.time() + 1.1
    print(f"  [{txt}]")


def move_cursor(tx, ty):
    global cx, cy
    cx = int(alpha * tx + (1 - alpha) * cx)
    cy = int(alpha * ty + (1 - alpha) * cy)
    pyautogui.moveTo(cx, cy)


print(f"GestureFlow Virtual Mouse  |  screen {SW}x{SH}  |  Q / ESC = quit")

# ── Main loop ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    now   = time.time()

    # FPS tracking
    frame_times = [t for t in frame_times if now - t < 1.0]
    frame_times.append(now)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands.process(rgb)
    rgb.flags.writeable = True

    gesture = "NO HAND"
    iu = mu = ru = pu = False
    pd = 0.0

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        MPD.draw_landmarks(frame, res.multi_hand_landmarks[0],
                           MP.HAND_CONNECTIONS)

        iu, mu, ru, pu = fingers_up(lm)
        pd      = pdist(lm)
        tx      = map_xy(lm[8].x, MARGIN, 1 - MARGIN, SW)
        ty      = map_xy(lm[8].y, MARGIN, 1 - MARGIN, SH)
        wrist_x = lm[0].x

        # ── Velocity-adaptive EMA alpha ────────────────
        if prev_tx is not None:
            vel   = ((lm[8].x - prev_tx) ** 2 + (lm[8].y - prev_ty) ** 2) ** 0.5
            t     = min(vel / VEL_MAX, 1.0)
            alpha = SMOOTH_MIN + t * (SMOOTH_MAX - SMOOTH_MIN)
        prev_tx, prev_ty = lm[8].x, lm[8].y

        # ══════════════════════════════════════════════
        # GESTURE PRIORITY ORDER
        # ══════════════════════════════════════════════

        # ── 1. PALM FREEZE — all 4 fingers up ─────────
        if iu and mu and ru and pu:
            gesture       = "PALM FREEZE"
            rc_start      = 0.0
            rc_fired      = False
            scroll_anchor = None
            prev_wx       = None
            if pstate == "DRAG":           # auto-drop on freeze
                pyautogui.mouseUp()
                pstate = "READY"
                flash("DROP (freeze)", (0, 220, 130))

        # ── 2. RIGHT-CLICK DWELL — IMR up, pinky down ─
        elif iu and mu and ru and not pu:
            gesture       = "RIGHT-CLICK..."
            scroll_anchor = None
            if rc_start == 0.0:
                rc_start = now
                rc_fired = False
            held = now - rc_start
            if held >= RCLICK_HOLD_S and not rc_fired:
                pyautogui.rightClick()
                flash("RIGHT CLICK", (255, 80, 80))
                rc_fired = True
                rc_start = 0.0
            move_cursor(tx, ty)

        # ── 3. SCROLL — IM up, RP down ────────────────
        elif iu and mu and not ru and not pu:
            gesture  = "SCROLL"
            rc_start = 0.0
            rc_fired = False
            prev_wx  = None
            sy       = lm[8].y
            # Anchor-based: set anchor on pose entry, advance after each tick
            if scroll_anchor is None:
                scroll_anchor = sy          # first frame — establish anchor
            else:
                dy = scroll_anchor - sy     # +ve = hand moved UP = scroll up
                if abs(dy) > SCROLL_DY_MIN and (now - last_scroll) > SCROLL_COOL:
                    direction     = 1 if dy > 0 else -1
                    notches       = max(2, min(15, int(abs(dy) * 200)))
                    scroll_native(direction, notches, cx, cy)
                    last_scroll   = now
                    scroll_anchor = sy      # advance anchor
                    flash(f"SCROLL {'UP' if direction > 0 else 'DOWN'}",
                          (180, 80, 255))

        # ── 4. SWIPE — closed fist + wrist velocity ───
        elif not iu and not mu and not ru and not pu:
            gesture       = "FIST"
            rc_start      = 0.0
            rc_fired      = False
            scroll_anchor = None
            if prev_wx is not None:
                wv = wrist_x - prev_wx      # +ve = hand moved right (mirrored)
                if abs(wv) > SWIPE_VEL and (now - last_swipe) > SWIPE_COOL:
                    if wv > 0:              # moved right = forward
                        pyautogui.hotkey('alt', 'right')
                        flash("SWIPE FORWARD >>", (255, 200, 0))
                    else:                   # moved left = back
                        pyautogui.hotkey('alt', 'left')
                        flash("<< SWIPE BACK", (255, 200, 0))
                    last_swipe = now
            prev_wx = wrist_x

        # ── 5. MOVE / CLICK / DRAG — index up ─────────
        elif iu and not mu:
            rc_start      = 0.0
            rc_fired      = False
            scroll_anchor = None
            prev_wx       = None

            pinching = pd < PINCH_CLOSE
            released = pd > PINCH_OPEN

            if pstate == "READY":
                if pinching:
                    pstate  = "HOLD"
                    pstart  = now
                    gesture = "PINCH..."
                else:
                    gesture = "MOVE"
                    move_cursor(tx, ty)

            elif pstate == "HOLD":
                held = now - pstart
                if pinching and held >= DRAG_HOLD_S:
                    pstate = "DRAG"
                    pyautogui.mouseDown()
                    flash("DRAG START", (0, 200, 255))
                    gesture = "DRAG..."
                elif released:
                    pstate = "READY"
                    if (now - last_click) > CLICK_COOL:
                        if (now - last_release) < DBL_WIN:
                            pyautogui.doubleClick()
                            flash("DOUBLE CLICK", (0, 255, 200))
                            last_release = 0.0
                        else:
                            pyautogui.click()
                            flash("CLICK", (0, 255, 200))
                            last_release = now
                        last_click = now
                    gesture = "CLICK"
                else:
                    gesture = f"HOLD {held:.1f}s"
                    move_cursor(tx, ty)

            elif pstate == "DRAG":
                if released:
                    pyautogui.mouseUp()
                    pstate = "READY"
                    flash("DROP!", (0, 220, 130))
                    gesture = "DROP"
                else:
                    gesture = "DRAGGING"
                    move_cursor(tx, ty)

        # ── 6. IDLE — any other pose ───────────────────
        else:
            gesture       = "IDLE"
            rc_start      = 0.0
            rc_fired      = False
            scroll_anchor = None
            prev_wx       = None
            if pstate == "DRAG":           # auto-drop on pose change
                pyautogui.mouseUp()
                pstate = "READY"
                flash("DROP (pose change)", (0, 220, 130))

    else:
        # No hand detected
        if pstate == "DRAG":
            pyautogui.mouseUp()
            pstate = "READY"
        rc_start      = 0.0
        rc_fired      = False
        scroll_anchor = None

    # ══════════════════════════════════════════════════
    # HUD OVERLAY
    # ══════════════════════════════════════════════════
    cv2.rectangle(frame, (0, 0), (w, 140), (12, 12, 12), -1)

    # Gesture colour map
    GCOL = {
        "MOVE":           (0,   255, 100),
        "PALM":           (0,   220, 255),
        "SCROLL":         (200,  80, 255),
        "FIST":           (255, 200,   0),
        "DRAGGING":       (0,   180, 255),
        "RIGHT-CLICK...": (255,  80,  80),
        "CLICK":          (0,   255, 200),
        "DROP":           (0,   220, 130),
        "HOLD":           (0,   160, 255),
        "PINCH...":       (0,   160, 255),
        "DRAG...":        (0,   180, 255),
    }
    first_word = gesture.split()[0] if gesture else ""
    gcol = GCOL.get(first_word, (150, 150, 150))
    cv2.putText(frame, f"Gesture: {gesture}", (10, 32),
                FONT, 0.70, gcol, 2, cv2.LINE_AA)

    # Flash action text
    if now < flash_exp:
        cv2.putText(frame, flash_txt, (10, 68),
                    FONT, 0.85, flash_col, 2, cv2.LINE_AA)

    # Pinch distance bar
    if res.multi_hand_landmarks:
        BAR   = 220
        bfill = int(min(pd / 0.20, 1.0) * BAR)
        cx_ln = int(PINCH_CLOSE / 0.20 * BAR)
        op_ln = int(PINCH_OPEN  / 0.20 * BAR)
        cv2.rectangle(frame, (10, 82), (10 + BAR, 93), (40, 40, 40), -1)
        bc = (0, 220, 80)  if pd < PINCH_CLOSE else \
             (0, 160, 255) if pstate == "DRAG"  else (100, 100, 180)
        cv2.rectangle(frame, (10, 82), (10 + bfill, 93), bc, -1)
        cv2.line(frame, (10 + cx_ln, 80), (10 + cx_ln, 95), (0, 220, 255), 2)
        cv2.line(frame, (10 + op_ln, 80), (10 + op_ln, 95), (60,  180,  60), 1)
        cv2.putText(frame, f"Pinch:{pd:.3f}  DRAG>{DRAG_HOLD_S}s",
                    (10, 110), FONT, 0.36, (150, 150, 150), 1, cv2.LINE_AA)

    # Right-click dwell progress bar
    if rc_start > 0 and now < rc_start + RCLICK_HOLD_S + 0.1:
        prog = min((now - rc_start) / RCLICK_HOLD_S, 1.0)
        pw   = int(prog * 160)
        cv2.rectangle(frame, (10, 118), (10 + 160, 128), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 118), (10 + pw,  128), (80,  80, 255), -1)
        cv2.putText(frame, "RC dwell", (178, 127),
                    FONT, 0.35, (180, 80, 255), 1, cv2.LINE_AA)

    # Finger circles  I M R P
    for i, (lbl, st) in enumerate(zip("IMRP", [iu, mu, ru, pu])):
        col = (0, 220, 80) if st else (0, 50, 180)
        px  = w - 178 + i * 44
        cv2.circle(frame, (px, 28), 14, col, -1)
        cv2.circle(frame, (px, 28), 14, (255, 255, 255), 1)
        cv2.putText(frame, lbl, (px - 5, 33), FONT, 0.46, (255, 255, 255), 1)

    # FPS + alpha + cursor coords
    fps = len(frame_times)
    cv2.putText(frame, f"FPS:{fps}  a:{alpha:.2f}", (w - 150, 55),
                FONT, 0.40, (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"({cx},{cy})", (w - 120, 75),
                FONT, 0.38, (130, 130, 130), 1, cv2.LINE_AA)

    # Gesture legend strip (bottom)
    cv2.rectangle(frame, (0, h - 54), (w, h), (10, 10, 10), -1)
    legend = "I=MOVE  Pinch=CLICK  Hold=DRAG  IMR(hold)=RCLICK  IMRP=FREEZE  IM=SCROLL  Fist=SWIPE"
    cv2.putText(frame, legend, (5, h - 35), FONT, 0.30, (130, 130, 130), 1, cv2.LINE_AA)
    cv2.putText(frame, "GestureFlow Virtual Mouse  v2.0", (5, h - 18),
                FONT, 0.35, (60, 180, 60), 1, cv2.LINE_AA)

    cv2.imshow("GestureFlow Virtual Mouse  [Q=quit]", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
        break

# ── Cleanup ───────────────────────────────────────────────
if pstate == "DRAG":
    pyautogui.mouseUp()
cap.release()
hands.close()
cv2.destroyAllWindows()
print("GestureFlow stopped.")
