# Raspberry Pi demo

Standalone bundle for the saliency-aware compression pipeline.
It ships two demos:

1. **Compression Pipeline Demo** — a pre-computed playback that runs
   end-to-end in ~25 seconds on a Raspberry Pi 4 or 5. Shows the headline
   compression result without re-running the (slow) encoder on the Pi.
2. **Live Saliency Demo** — opens the camera and runs the *real* saliency
   pipeline live (same backends, fusion, smoothing and mask shaping the
   codec uses), overlaying the saliency heatmap on the feed in real time.

## What's in here

- `gui.py` — graphical front end with both demos in tabs (recommended for
  live demos). Requires `python3-tk`.
- `demo_pi.py` — pure-stdlib console replay of the compression pipeline.
  Launch this if you want to run from the terminal, or if tkinter isn't
  installed.
- `scripts/live_demo.py` — the live saliency demo (opened by the GUI's
  "Live Saliency Demo" tab; can also be run directly from the terminal).
- `src/` — the saliency/codec package the live demo imports.
- `yolov8n.pt` — YOLOv8n weights, used only by the `yolo` / `yolo+spectral`
  live-demo backends (kept here so they load offline).
- `requirements-live.txt` — Python deps for the live demo (the compression
  playback needs none of these).
- `clip_virat_crf22.mp4` (~43 MB) — the pre-computed 2-row demo grid:

  | layout    | left              | middle              | right        |
  |-----------|-------------------|---------------------|--------------|
  | **top**   | original          | baseline H.265      | *(black borders on each outer side)* |
  | **bottom**| saliency mask     | static background   | ours_bgfg    |

  Top row says *"here's the input and the dumb-baseline H.265"*. Bottom row
  says *"here's how we do better: identify what matters with the saliency
  mask, lock everything else to this one static background image, and the
  result is the file in the bottom-right."* Every panel is labelled with
  its on-disk MB and ours_bgfg shows its size delta vs baseline.

## Setup

Keep the folder structure intact — `gui.py`, `demo_pi.py`, the `scripts/`
and `src/` folders, and `yolov8n.pt` all need to stay where they are
relative to each other.

### Compression Pipeline Demo

No Python dependencies beyond the standard library are required. You only
need a video player for the final "open" step. On Raspberry Pi OS Desktop,
any of these work and the demo picks the first one available:

- `vlc` (recommended; usually pre-installed on Pi OS Desktop)
- `mpv`
- `ffplay`
- `xdg-open` (falls through to whatever is registered as the default video
  handler)

Install VLC if it's missing:

```bash
sudo apt install vlc
```

If you want the `gui.py` front end, also ensure tkinter is present (usually
preinstalled on Pi OS Desktop):

```bash
sudo apt install python3-tk
```

### Live Saliency Demo

This one runs real computer vision, so it needs a few Python packages and
a working camera. The lightweight tier is enough on a Pi:

```bash
pip install -r requirements-live.txt
```

That installs NumPy + opencv-contrib-python, which enables the **spectral**
and **finegrained** saliency backends (CPU-only, real-time on a Pi —
recommended). The `yolo` and `yolo+spectral` backends additionally need
PyTorch + Ultralytics; those lines are commented out in
`requirements-live.txt` because they're heavy on a Pi and run YOLO on the
CPU. Uncomment them only if you want semantic (person/vehicle/bag)
saliency. The `yolov8n.pt` weights ship in this folder so they load offline.

## Run

**Recommended — GUI:**

```bash
python3 gui.py
```

A window opens with two tabs:

- **Live Saliency Demo** — set the camera index, pick a saliency backend,
  set the number of background-calibration frames, toggle rolling-median
  auto-recalibration, then click "Launch Live Demo Window". A separate
  OpenCV window opens: step out of frame during the calibration banner,
  step back in to see the live saliency heatmap. Keys in that window —
  `q` quit, `r` recalibrate, `s` snapshot.
- **Compression Pipeline Demo** — click "Run Compression Pipeline Demo".
  The log streams the pipeline stages in real time and the comparison
  video opens in your default player when finished.

**Console-only fallbacks:**

```bash
python3 demo_pi.py                  # compression pipeline replay
python3 scripts/live_demo.py        # live saliency demo (add --saliency spectral on a Pi)
```

Use these if tkinter isn't installed or you prefer the terminal. Run
`python3 scripts/live_demo.py --help` to see every flag (camera, saliency
backend, bg-samples, mask-only, auto-recal).

## Numbers shown by the demo

Real outputs of the reference run on a 75-second 1080p VIRAT surveillance
clip (~158 MB raw):

| Track          | Size     | sal-PSNR (dB) |
|----------------|----------|----------------|
| original       | 158 MB   | —              |
| baseline H.265 | 33.9 MB  | 37.44          |
| ours_bgfg      |  6.5 MB  | 34.76          |

Headline: **~81% size reduction vs baseline H.265 at the same CRF**,
with a 2.7 dB sal-PSNR delta on the salient regions we promised to keep.

## Re-pointing the demo at a different cached clip

If you want to swap the demo for a different clip, edit the constants at
the top of `demo_pi.py`:

- `COMPARISON_VIDEO` — path to the new demo grid mp4 (keep it next to
  the script unless you change this)
- `BASELINE_SIZE_MB`, `OURS_SIZE_MB`, `BASELINE_SAL_PSNR_DB`,
  `OURS_SAL_PSNR_DB`, `SOURCE_*`, `COMPARE_*` — set to the real measured
  numbers from your `ablation_bgfg.py` / `compare_clip.py --internals` run

`gui.py` calls into `demo_pi.py`, so updating the constants there is
sufficient for both entry points.

## Pacing knob

`SPEED_MULTIPLIER` in `demo_pi.py` (default `1.0`) scales every sleep
uniformly. Set to `0.5` to halve the runtime; `2.0` to slow it down
for a longer talk track.
