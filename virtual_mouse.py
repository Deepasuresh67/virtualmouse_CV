"""
virtual_mouse.py  —  Virtual Mouse (pinch-click edition)
Run:  python virtual_mouse.py

Gestures
--------
  MOVE        :  index finger up only        → cursor follows fingertip
  CLICK       :  pinch thumb+index, release  → left click (fires on release)
  DOUBLE-CLICK:  two pinch-releases < 0.5 s  → double click
  SCROLL UP   :  index+middle up, move up    → scroll up
  SCROLL DOWN :  index+middle up, move down  → scroll down
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import ctypes
import time

# ── pyautogui setup ───────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.0

# ── constants ─────────────────────────────────────────────
CAM_W, CAM_H    = 640, 480
MARGIN          = 0.15        # crop fraction each edge for coord mapping
SMOOTH          = 0.45        # EMA alpha  (0=frozen, 1=raw)

PINCH_CLOSE     = 0.06        # normalised dist → "pinching"
PINCH_OPEN      = 0.10        # normalised dist → "released"
CLICK_COOLDOWN  = 0.50        # seconds between clicks
DBL_WINDOW      = 0.45        # seconds between releases → double click

SCROLL_SPEED    = 8           # pyautogui scroll units / tick (increased)
SCROLL_COOL     = 0.10        # seconds between scroll ticks
SCROLL_MIN_DY   = 0.004       # min frame-to-frame delta to scroll (lowered)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── screen ────────────────────────────────────────────────
SW, SH = pyautogui.size()

# ── mediapipe ─────────────────────────────────────────────
mh    = mp.solutions.hands
mdraw = mp.solutions.drawing_utils
hands = mh.Hands(max_num_hands=1,
                 min_detection_confidence=0.75,
                 min_tracking_confidence=0.75)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

# ── state ─────────────────────────────────────────────────
cx, cy          = SW // 2, SH // 2   # smoothed cursor
pinch_active    = False               # True = currently pinching
last_release    = 0.0                 # time of last pinch release
last_click      = 0.0                 # debounce
last_scroll     = 0.0
prev_y          = None                # for frame-delta scroll
action_txt      = ""
action_expire   = 0.0
frame_times     = []


def tip_pip_dist(lm, tip, pip):
    """Euclidean distance between two landmarks (normalised)."""
    return ((lm[tip].x - lm[pip].x)**2 +
            (lm[tip].y - lm[pip].y)**2) ** 0.5


def fingers_up(lm):
    """Return (index_up, middle_up, ring_up, pinky_up)."""
    pairs = [(8,6), (12,10), (16,14), (20,18)]
    return tuple(lm[t].y < lm[p].y for t, p in pairs)


def scroll_wheel(direction, amount, x, y):
    """
    Send a native Windows WM_MOUSEWHEEL event.
    direction: +1 = scroll UP, -1 = scroll DOWN
    amount: number of 'notches' (120 units each)
    Bypasses pyautogui.scroll() which is unreliable for negative values on Windows.
    """
    MOUSEEVENTF_WHEEL = 0x0800
    WHEEL_DELTA = 120
    # Move cursor to target position first
    ctypes.windll.user32.SetCursorPos(int(x), int(y))
    delta = ctypes.c_int(direction * amount * WHEEL_DELTA)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, delta, 0)


def map_xy(v, lo, hi, out):
    return int(np.clip((v - lo) / (hi - lo) * out, 0, out - 1))


def show_action(txt):
    global action_txt, action_expire
    action_txt    = txt
    action_expire = time.time() + 1.0
    print(f"[{txt}]  @ ({cx},{cy})")


print(f"Virtual Mouse started  |  screen {SW}x{SH}  |  Q to quit")

# ── main loop ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    now   = time.time()

    frame_times = [t for t in frame_times if now - t < 1.0]
    frame_times.append(now)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands.process(rgb)
    rgb.flags.writeable = True

    gesture  = "NO HAND"
    pinch_d  = 0.0
    iu = mu = ru = pu = False

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        mdraw.draw_landmarks(frame, res.multi_hand_landmarks[0],
                             mh.HAND_CONNECTIONS)

        iu, mu, ru, pu = fingers_up(lm)
        pinch_d = tip_pip_dist(lm, 4, 8)   # thumb tip ↔ index tip

        # ── SCROLL  (index + middle up) ───────────────────
        if iu and mu and not ru and not pu:
            gesture = "SCROLL"
            cy_tip  = lm[8].y

            if prev_y is not None:
                dy = cy_tip - prev_y          # +ve = hand moved DOWN
                if abs(dy) > SCROLL_MIN_DY and (now - last_scroll) > SCROLL_COOL:
                    direction = -1 if dy > 0 else 1   # hand DOWN = scroll down
                    scroll_wheel(direction, SCROLL_SPEED, cx, cy)
                    last_scroll = now
                    show_action(f"SCROLL {'UP' if direction > 0 else 'DOWN'}")

            prev_y      = cy_tip
            pinch_active = False            # reset pinch while scrolling

        else:
            prev_y = None                   # reset scroll tracker

            # ── PINCH CLICK  ──────────────────────────────
            # Detect on RELEASE edge so held pinch doesn't
            # re-fire every frame.
            currently_pinching = pinch_d < PINCH_CLOSE

            if currently_pinching:
                pinch_active = True
                gesture = f"PINCH ({pinch_d:.3f})"

                # Move cursor to index tip while pinching too
                tx = map_xy(lm[8].x, MARGIN, 1-MARGIN, SW)
                ty = map_xy(lm[8].y, MARGIN, 1-MARGIN, SH)
                cx = int(SMOOTH * tx + (1-SMOOTH) * cx)
                cy = int(SMOOTH * ty + (1-SMOOTH) * cy)
                pyautogui.moveTo(cx, cy)

            elif pinch_active and pinch_d > PINCH_OPEN:
                # ── RELEASE → fire click ──────────────────
                pinch_active = False
                if (now - last_click) > CLICK_COOLDOWN:
                    if (now - last_release) < DBL_WINDOW:
                        pyautogui.doubleClick()
                        show_action("DOUBLE CLICK")
                        last_release = 0.0          # reset so 3rd won't chain
                    else:
                        pyautogui.click()
                        show_action("CLICK")
                        last_release = now
                    last_click = now
                gesture = "CLICK"

            # ── MOVE  (only index up, not pinching) ───────
            elif iu and not mu:
                gesture = "MOVE"
                tx = map_xy(lm[8].x, MARGIN, 1-MARGIN, SW)
                ty = map_xy(lm[8].y, MARGIN, 1-MARGIN, SH)
                cx = int(SMOOTH * tx + (1-SMOOTH) * cx)
                cy = int(SMOOTH * ty + (1-SMOOTH) * cy)
                pyautogui.moveTo(cx, cy)

            else:
                gesture = "IDLE"

    # ── HUD ───────────────────────────────────────────────
    # Dark header bar
    cv2.rectangle(frame, (0,0), (w,115), (15,15,15), -1)

    # Gesture name
    GCOL = {"MOVE":(0,255,100), "SCROLL":(200,80,255),
            "IDLE":(90,90,90),  "NO HAND":(60,60,60), "CLICK":(0,200,255)}
    gcol = GCOL.get(gesture.split()[0], (200,200,200))
    cv2.putText(frame, f"Gesture: {gesture}", (10,32),
                FONT, 0.70, gcol, 2, cv2.LINE_AA)

    # Action flash
    if now < action_expire:
        cv2.putText(frame, action_txt, (10,68),
                    FONT, 0.85, (0,255,200), 2, cv2.LINE_AA)

    # Pinch bar
    if res.multi_hand_landmarks:
        BAR = 200
        filled  = int(min(pinch_d / 0.20, 1.0) * BAR)
        close_x = int(PINCH_CLOSE / 0.20 * BAR)
        open_x  = int(PINCH_OPEN  / 0.20 * BAR)
        cv2.rectangle(frame, (10,82), (10+BAR,94), (40,40,40), -1)
        barcol = (0,220,80) if pinch_d < PINCH_CLOSE else (100,100,180)
        cv2.rectangle(frame, (10,82), (10+filled,94), barcol, -1)
        cv2.line(frame, (10+close_x,80), (10+close_x,96), (0,220,255), 2)  # CLOSE line
        cv2.line(frame, (10+open_x, 80), (10+open_x, 96), (60,180,60),  1) # OPEN  line
        cv2.putText(frame, f"Pinch:{pinch_d:.3f}  close<{PINCH_CLOSE} open>{PINCH_OPEN}",
                    (10,110), FONT, 0.36, (160,160,160), 1, cv2.LINE_AA)

    # Finger circles  I M R P
    for i,(lbl,st) in enumerate(zip("IMRP",[iu,mu,ru,pu])):
        col = (0,220,80) if st else (0,50,180)
        px  = w - 170 + i*42
        cv2.circle(frame, (px,28), 14, col, -1)
        cv2.circle(frame, (px,28), 14, (255,255,255), 1)
        cv2.putText(frame, lbl, (px-5,33), FONT, 0.46, (255,255,255), 1)

    # FPS
    fps = len(frame_times)
    cv2.putText(frame, f"FPS:{fps}", (w-70,68), FONT, 0.50, (0,255,180), 1)
    cv2.putText(frame, f"({cx},{cy})", (w-110,90), FONT, 0.38, (130,130,130), 1)

    cv2.imshow("Virtual Mouse  [Q=quit]", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print("Stopped.")
