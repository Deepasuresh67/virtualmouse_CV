"""
config.py — Central configuration for the Virtual Mouse System.
All tunable parameters live here to avoid magic numbers scattered across modules.
"""

# ──────────────────────────────────────────────
#  Camera & Display
# ──────────────────────────────────────────────
CAMERA_INDEX = 0          # Webcam index (0 = default)
FRAME_WIDTH  = 1280       # Capture width  (px)
FRAME_HEIGHT = 720        # Capture height (px)
FLIP_FRAME   = True       # Mirror the frame (natural feel)

# ──────────────────────────────────────────────
#  MediaPipe Hand Tracking
# ──────────────────────────────────────────────
MAX_HANDS              = 1
DETECTION_CONFIDENCE   = 0.75
TRACKING_CONFIDENCE    = 0.75

# ──────────────────────────────────────────────
#  Spatial Interaction Zones  (fractions of frame)
# ──────────────────────────────────────────────
# TOP zone  : y < TOP_ZONE_THRESH
# BOTTOM zone: y > BOTTOM_ZONE_THRESH
# LEFT zone : x < LEFT_ZONE_THRESH
# RIGHT zone: x > RIGHT_ZONE_THRESH
# CENTER    : everything else
TOP_ZONE_THRESH    = 0.25   # top 25 % (raised from 20 so scroll zone is easier to reach)
BOTTOM_ZONE_THRESH = 0.80   # bottom 20 %
LEFT_ZONE_THRESH   = 0.20   # left  20 %
RIGHT_ZONE_THRESH  = 0.80   # right 20 %
ZONE_HYSTERESIS    = 0.03   # dead-band to prevent jitter at zone edges

# ──────────────────────────────────────────────
#  Cursor / Coordinate Mapping
# ──────────────────────────────────────────────
# The "active area" of the camera used to map to full screen.
# Shrinking this region gives more screen travel per cm of finger movement.
ACTIVE_AREA_MARGIN = 0.15   # fraction cropped from each edge

# ──────────────────────────────────────────────
#  Adaptive Smoothing
# ──────────────────────────────────────────────
# alpha is the EMA weight for the NEW position.
# High alpha → responsive (less smoothing)
# Low  alpha → smooth   (more smoothing / precision)
#
# FIX: MIN was 0.10 — caused cursor to barely move for slow hands.
#      Raised to 0.35 so even slow motion gives visible cursor travel.
SMOOTH_ALPHA_MIN    = 0.35   # maximum smoothing (very slow movement)
SMOOTH_ALPHA_MAX    = 0.80   # minimum smoothing (very fast movement)
SMOOTH_VELOCITY_MAX = 60     # px/frame at which alpha reaches MAX (lowered for faster response)

# ──────────────────────────────────────────────
#  Gesture Detection Thresholds
# ──────────────────────────────────────────────
# --- General ---
PINCH_DISTANCE_THRESHOLD   = 0.065  # normalised index-thumb dist for pinch
MIDDLE_PINCH_THRESHOLD     = 0.065  # normalised middle-thumb dist for right-click

# --- Click debounce & timing ---
# DEBOUNCE: after a click fires, ignore new clicks for this long (prevents double-fire)
CLICK_DEBOUNCE_S           = 0.40   # seconds
# MAX HOLD: pinch held longer than this = drag intent, not click
CLICK_MAX_HOLD_S           = 1.20   # seconds

# --- Double-click ---
# Time window after first click release within which a 2nd pinch = double-click
DOUBLE_CLICK_WINDOW_MS     = 600    # ms

# --- Cancel gesture (micro-oscillation) ---
OSCILLATION_FRAMES         = 8     # number of frames to analyse
OSCILLATION_STD_THRESHOLD  = 0.012 # std-dev of tip position triggers cancel

# --- Scroll ---
# SENSITIVITY: number of pyautogui scroll units per gesture tick
SCROLL_SENSITIVITY         = 5      # increased from 3 for more visible scroll
# DEAD_ZONE: minimum normalised y-movement before scroll fires (prevents drift)
SCROLL_DEAD_ZONE           = 0.010  # lowered from 0.015 — fires on smaller movements

# ──────────────────────────────────────────────
#  Confidence System
# ──────────────────────────────────────────────
GESTURE_STABILITY_FRAMES   = 3     # frames a gesture must be stable before exec (lowered for faster click response)
CONFIDENCE_THRESHOLD       = 0.80  # fraction of stable frames required

# ──────────────────────────────────────────────
#  Visual Feedback / HUD
# ──────────────────────────────────────────────
HUD_FONT_SCALE     = 0.65
HUD_THICKNESS      = 2
SKELETON_COLOR     = (0, 255, 180)   # BGR  cyan-green
LANDMARK_COLOR     = (255, 255, 255) # BGR  white
TIP_COLOR          = (0, 200, 255)   # BGR  yellow-orange

# Zone badge colours (BGR)
ZONE_COLORS = {
    "CENTER" : (50,  200,  50),
    "TOP"    : (200,  50, 200),
    "BOTTOM" : (100, 100, 200),
    "LEFT"   : (200, 150,  50),
    "RIGHT"  : (50,  150, 200),
}

# Confidence bar colours
CONF_LOW_COLOR  = (0,  50, 220)   # red
CONF_MID_COLOR  = (0, 180, 220)   # yellow
CONF_HIGH_COLOR = (0, 220,  80)   # green
