# Running on a Raspberry Pi 5

This is the deployment target for the live demo. Everything in the repo
that runs on a laptop also runs on a Pi 5 with the same commands — these
are the steps to get a fresh Pi from "first boot" to "live saliency
heatmap window on an attached display."

> **Tested on:** Raspberry Pi 5 (8 GB), Raspberry Pi OS 64-bit (Bookworm),
> USB UVC webcam, HDMI display. The exact same pipeline runs on the team
> macbooks; only the install path is different.

---

## 1. First-time setup (one-off, ~20 min)

```bash
# System dependencies
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y \
  git python3 python3-venv python3-pip \
  ffmpeg libavcodec-extra \
  v4l-utils \
  libgl1 libglib2.0-0          # OpenCV runtime deps
```

Pi 5 specifically — verify the camera is reachable and at what resolution:

```bash
v4l2-ctl --list-devices
v4l2-ctl --list-formats-ext -d /dev/video0
```

If a USB webcam is plugged in, you should see something like
`/dev/video0` listed. If you see only the CSI camera mount points, plug
the webcam in and re-run.

## 2. Clone + venv

```bash
git clone https://github.com/SaiAmartya/saliency-surveillance-codec.git
cd saliency-surveillance-codec

# venv keeps system Python clean; --system-site-packages is NOT needed
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
```

## 3. Install Python dependencies

The full `requirements.txt` works on Pi 5 as-is. Two notes:

- **PyTorch on Pi 5**: the default `torch` wheel installs fine on aarch64
  (Bookworm). YOLOv8n inference runs on CPU at about 6–10 fps on the Pi 5
  at 640×480 input — enough for our demo where we don't need every frame.
- **opencv-contrib-python** brings the `cv2.saliency` module we need for
  spectral residual. Pin it; the base `opencv-python` package does not
  include this module. `ultralytics` pulls in `opencv-python` anyway, and
  whichever of the two installs last owns `cv2/`. If contrib loses, the
  spectral backend falls back to a NumPy port on its own; reinstall
  `opencv-contrib-python` only if you need the `finegrained` backend.

```bash
pip install -r requirements.txt
```

If torch installation hangs or fails (occasionally happens on slow SD
cards), retry with `--no-cache-dir`. On first run, YOLO will
auto-download `yolov8n.pt` (~6 MB) — make sure the Pi has internet for
that one moment.

## 4. Smoke test (30 seconds)

```bash
# Verify the saliency stack — should print numbers, no crash
python -c "
import sys; sys.path.insert(0, '.')
from src.saliency import SaliencyEstimator
import numpy as np
est = SaliencyEstimator(backend='yolo+spectral')
fake = (np.random.rand(360, 640, 3) * 255).astype('uint8')
print('saliency:', est.predict(fake).shape, 'OK')
"
```

If that prints `saliency: (360, 640) OK`, the install is good.

## 5. The three demos worth running

### A. Live saliency heatmap window (the headline visual)

```bash
python scripts/live_demo.py
```

Opens a window with the live webcam feed overlaid with the YOLO+spectral
saliency heatmap and a real-time "usefulness" score from the gate. Press
`q` to quit. On Pi 5 you'll get ~5–10 fps depending on lighting — fine
for showing what the saliency picks up on.

### B. Webcam → compressed comparison (the 4-panel PNG)

```bash
# 2-second capture, encode both ways, render a 4-panel PNG comparison
python scripts/photo_demo.py --codec bgfg --duration 5 --saliency yolo+spectral --show
```

Output: `results/photo_demo/comparison.png`. The `--show` flag pops it
open in the Pi's default image viewer (Image Viewer or feh).

**Tip:** stand up and walk across the camera during the capture.
Static-room captures don't show off the bg/fg codec because there's no
foreground for the saliency map to find.

### C. The pre-built VIRAT comparison video

The headline result is a 1080p VIRAT clip with all 4 panels playing in
lockstep. That mp4 was rendered ahead of time on a laptop and lives at:

```
results/comparisons/clip_14_crf22.mp4    # bar interior, falling subject
results/comparisons/headline_*.mp4       # any pre-baked headline videos
```

If they aren't on the Pi (they shouldn't be — they're gitignored
because of size), copy them over from a laptop first:

```bash
# from the laptop, push the demo videos to the pi
scp results/comparisons/*.mp4 pi@<pi-address>:~/saliency-surveillance-codec/results/comparisons/
```

To play one on the Pi, just double-click the file or:

```bash
xdg-open results/comparisons/clip_14_crf22.mp4
```

---

## 6. Optional: bake fresh demo videos on the Pi

The Pi can build any demo video locally, it just takes longer than a
laptop. Encoding a 24-second VIRAT clip at CRF 22 takes ~3 min on Pi 5
versus ~30 s on an M3 macbook. If you do want to bake fresh demos on the
Pi:

```bash
# pre-built demo videos (matches §8 of docs/demo_scripts.md)
python scripts/compare_clip.py --clip clip_14 --crf 22 --out results/comparisons/headline_fall.mp4
python scripts/compare_clip.py --clip clip_02 --crf 22 --out results/comparisons/headline_grab.mp4
python scripts/compare_clip.py --clip clip_04 --crf 22 --out results/comparisons/headline_pool.mp4
```

The clips themselves (`data/real/clip_*.mp4`) need to be on the Pi too —
they're gitignored. Copy from a laptop:

```bash
scp -r data/ pi@<pi-address>:~/saliency-surveillance-codec/
```

---

## 7. Pi-specific gotchas

- **Camera permissions**: the user needs to be in the `video` group.
  `sudo usermod -aG video $USER`, then log out and back in.
- **PyTorch threadpool**: on a 4-core Pi 5, set `torch.set_num_threads(4)`
  at the top of any heavy script if you see CPU underutilization. Not
  required for `live_demo.py` (it's already optimized).
- **Avoid the Pi camera module** for this demo — the pipeline expects a
  UVC webcam. The libcamera path works but adds complexity we don't need
  for a short demo.
- **No `cv2.imshow` over SSH**: if you SSH to the Pi without X
  forwarding, `live_demo.py` will fail to create the window. Either run
  it from a terminal directly on the Pi (with monitor attached), or
  SSH with `-X` and accept the latency.
- **`ultralytics` first run**: the first YOLO call downloads
  `yolov8n.pt` (~6 MB). Make sure the Pi has internet at first launch;
  after that, it's offline-capable.
- **The "100+ fps" saliency figure refers to the
  spectral-residual component of saliency**, which is the only piece
  that hits that throughput on a Pi 5. The full YOLO+spectral+motion
  stack runs at 5–10 fps on the Pi.

---

## 8. Demo-day checklist

Bring to the demo:

- The Pi 5 (in case it has the demo wired up)
- USB webcam (with a known-good USB cable)
- HDMI cable + small monitor for the live saliency window
- A laptop as a backup running the same code
- Pre-baked demo videos on a USB drive (in case the Pi misbehaves)

The night before:

1. Boot the Pi from a known-good SD card
2. Run `python scripts/live_demo.py` and confirm the heatmap renders
3. Run `python scripts/photo_demo.py --duration 5 --codec bgfg --show` and confirm a comparison PNG drops onto the desktop
4. Open one pre-baked comparison mp4 and confirm video playback works

If all four pass, you're set.

---

## 9. Comparing Pi 5 vs M-series macbook

For the writeup, here are the rough numbers we've measured on each. The
codec is identical; only encode wall-clock differs.

| Stage | M3 Pro macbook | Pi 5 (8 GB, CPU only) |
|---|---|---|
| Spectral saliency map | 110 fps | 80–100 fps |
| YOLOv8n detect (640×480) | 60 fps (MPS) | 7 fps |
| `live_demo.py` overall | 35 fps | 5–10 fps |
| Encode a 5 s 640×360 clip @ CRF 22 (bgfg) | ~3 s | ~15 s |
| Encode a 24 s 720p VIRAT clip @ CRF 22 (bgfg) | ~30 s | ~3 min |

The Pi numbers are perfectly fine for a surveillance camera (security
cams typically record at 10–15 fps anyway), but encoding time is what
you'll notice. Pre-bake demo videos.
