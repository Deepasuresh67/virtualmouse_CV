# GestureFlow Virtual Mouse 🖱️✋

> **Control your computer with hand gestures — no mouse, no touchpad, just your webcam.**

A real-time, vision-based virtual mouse built with **MediaPipe Hands**, **OpenCV**, and **PyAutoGUI**. Detects 21 hand landmarks at 30+ FPS on a standard CPU webcam and translates hand gestures into full mouse control — including move, click, drag & drop, scroll, and more.

---

## Demo

```
☝️  Index finger  →  Move cursor
🤏  Pinch         →  Click
🤏⏱️  Hold pinch   →  Drag & Drop
✌️  Peace sign    →  Scroll Up / Down
🖐️  Open palm     →  Freeze cursor
✌️🤙  3 fingers   →  Right-click (dwell)
✊  Fist swipe    →  Browser Back / Forward
```

---

## Features

### 🎯 8 Gesture Actions

| Gesture | Hand Pose | Action |
|---|---|---|
| **Move** | ☝️ Index finger only | Cursor follows fingertip |
| **Click** | 🤏 Pinch → release | Left click (fires on release) |
| **Double Click** | 🤏🤏 Two quick pinches | Double click |
| **Drag & Drop** | 🤏 Hold pinch > 0.5s + move | Drag items, release to drop |
| **Right-Click** | ✌️🤙 Index+Middle+Ring up, hold 1.4s | Context menu |
| **Palm Freeze** | 🖐️ All fingers open | Cursor locks — rest your hand |
| **Scroll** | ✌️ Peace sign + move up/down | Scroll any window |
| **Swipe Navigate** | ✊ Fist + fast left/right | Browser back / forward |

### ⚙️ Technical Innovations

- **Velocity-Adaptive EMA Smoothing** — Cursor smoothing factor changes dynamically per frame based on hand movement speed. Slow = precision mode; fast = responsive mode.
- **Release-Edge Click Detection** — Click fires when you *open* your fingers, not when you close them. Eliminates all false fires from held pinch.
- **Pinch Hysteresis** — Two separate thresholds (close: 0.06, open: 0.095) prevent rapid toggling at the boundary.
- **Dwell-Time Right-Click** — Requires holding the 3-finger pose for 1.4 seconds with a live progress bar. Zero false positives.
- **Palm Freeze** — Open palm locks the cursor, letting you rest your hand without unwanted movement.
- **Native WM_MOUSEWHEEL Scroll** — Uses Windows `ctypes` API directly instead of PyAutoGUI, making scroll work in *all* applications in both directions.
- **Wrist-Velocity Swipe** — Swipe detection uses wrist landmark (lm[0]) velocity instead of fingertips — more stable during fast movements.

---

## Requirements

- Python 3.9+
- Standard RGB webcam
- Windows OS (for native scroll API)

### Dependencies

```
mediapipe>=0.10.0
opencv-python>=4.8.0
pyautogui>=0.9.54
numpy>=1.24.0
screeninfo>=0.8.1
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Deepasuresh67/virtualmouse_CV.git
cd virtualmouse_CV

# Install dependencies
pip install -r requirements.txt

# Run the virtual mouse
python virtual_mouse.py
```

Press **Q** or **ESC** to quit.

---

## How to Use Each Gesture

### ☝️ Move Cursor
Raise only your **index finger**. The cursor smoothly follows your fingertip across the screen.

### 🤏 Click
Bring your **thumb tip and index tip together** (pinch), then open them. The click fires on the *release*.

### 🤏🤏 Double Click
Do the pinch-and-release motion **twice quickly** (within 0.42 seconds).

### 🤏⏱️ Drag & Drop
Pinch and **hold** for 0.5 seconds — you'll see "DRAG START" flash. Move your hand to drag. Open fingers to drop.

### ✌️🤙 Right-Click
Raise **index + middle + ring** fingers. Hold the pose — a purple progress bar fills up. Right-click fires after **1.4 seconds**. Pinky must stay down.

### 🖐️ Palm Freeze
Raise **all 4 fingers** (full open palm). The cursor freezes. Lower your hand to resume.

### ✌️ Scroll
Raise **index + middle** fingers (peace sign). Move your hand **up** to scroll up, **down** to scroll down. Speed scales with movement.

### ✊ Swipe Navigate
Make a **closed fist** and swipe quickly:
- **Left** → Browser Back (Alt+Left)
- **Right** → Browser Forward (Alt+Right)

---

## On-Screen HUD Guide

```
┌─────────────────────────────────────────────────┐
│ Gesture: MOVE                    [ I ][ M ][ R ][ P ] │
│ CLICK                           FPS:28 a:0.55        │
│ Pinch:0.045 ████░░░░│  DRAG>0.5s   (1240,680)        │
│ ░░░░░░░░░░░░░░░  RC dwell                            │
├─────────────────────────────────────────────────┤
│ I=MOVE Pinch=CLICK Hold=DRAG IMR=RCLICK IMRP=FREEZE  │
│ GestureFlow Virtual Mouse v1.0                        │
└─────────────────────────────────────────────────┘
```

| HUD Element | Meaning |
|---|---|
| **Gesture label** | Current detected gesture (colour-coded) |
| **Flash text** | Last action fired (shown for 1 second) |
| **Pinch bar** | Thumb-index distance — cyan line = click threshold |
| **RC dwell bar** | Right-click progress — fills over 1.4s |
| **I M R P circles** | Green = finger up, Blue = finger down |
| **FPS / alpha** | Frames per second + current smoothing factor |
| **Cursor coords** | Current screen position |

---

## Project Structure

```
cv_proj/
├── virtual_mouse.py      ← Main entry point (self-contained, run this)
├── main.py               ← Modular pipeline entry point
├── config.py             ← All tunable parameters
├── hand_tracker.py       ← MediaPipe wrapper + landmark extraction
├── gesture_engine.py     ← Intent-aware gesture classifier
├── cursor_controller.py  ← EMA smoothing + mouse dispatch
├── zone_manager.py       ← Spatial interaction zones
├── visual_feedback.py    ← HUD overlay renderer
└── requirements.txt      ← Python dependencies
```

---

## Configuration

Edit `virtual_mouse.py` constants to tune for your setup:

```python
PINCH_CLOSE   = 0.060   # ↓ smaller = need tighter pinch to click
PINCH_OPEN    = 0.095   # ↑ larger = need wider open to release
DRAG_HOLD_S   = 0.50    # seconds to hold before drag starts
RCLICK_HOLD_S = 1.40    # seconds to hold 3-finger pose for right-click
SMOOTH_MIN    = 0.30    # cursor smoothing at slow speed (0=frozen, 1=raw)
SMOOTH_MAX    = 0.82    # cursor smoothing at fast speed
SWIPE_VEL     = 0.018   # wrist velocity threshold for swipe trigger
```

---

## Methodology

1. **Capture** — OpenCV reads webcam frames at 640×480, mirrored for natural feel
2. **Detect** — MediaPipe Hands extracts 21 3D landmarks per frame
3. **Classify** — Rule-based gesture classifier with priority ordering:
   - Palm Freeze > Right-Click Dwell > Scroll > Swipe > Move/Click/Drag
4. **Smooth** — Velocity-adaptive EMA maps normalised coordinates to screen pixels
5. **Act** — PyAutoGUI + native Windows ctypes dispatch mouse events

---

## Known Limitations

- Optimised for **Windows** (native scroll API uses Win32)
- Works best in **good lighting** (avoid backlit environments)
- Designed for **right-handed** index-finger dominant use
- Drag requires intentional hold — accidental drags are unlikely

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| MediaPipe | ≥0.10.0 | Hand landmark detection |
| OpenCV | ≥4.8.0 | Video capture + HUD rendering |
| PyAutoGUI | ≥0.9.54 | Mouse movement + click dispatch |
| NumPy | ≥1.24.0 | Coordinate math + EMA |
| ctypes | built-in | Native Windows scroll events |

---

## License

MIT License — free to use, modify, and distribute.

---

*Built with MediaPipe + OpenCV + Python*
