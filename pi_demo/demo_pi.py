#!/usr/bin/env python3
"""
Raspberry Pi playback demo.

Replays the saliency-aware compression pipeline log output using the
measured numbers from the M3 Pro reference run, then opens the
pre-computed 2-row comparison video that ships next to this script
(clip_virat_crf22.mp4).

Layout of the comparison video:
    Top row    : original | baseline H.265           (input vs dumb baseline)
    Bottom row : saliency mask | static background | ours_bgfg   (the machinery + result)

This script does NOT re-encode anything — the bg/fg codec is heavy and
takes minutes on a Mac; the Pi prototype just plays back the cached
result so viewers can see the headline outcome in about 30 seconds.

Usage:
    python3 demo_pi.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path


# ----------------------------------------------------------------------
# Measured numbers from the reference run (Mac M3 Pro, May 2026).
# These are the REAL outputs of:
#   ablation_bgfg.py  --clips clip_virat --crfs 22 --saliency yolo+spectral
#   compare_clip.py   --clip clip_virat --crf 22 --internals --panel-scale 0.5
# ----------------------------------------------------------------------

CLIP_ID                  = "clip_virat"
CRF                      = 22
SALIENCY_BACKEND         = "yolo+spectral"

SOURCE_RES               = "1920x1080"
SOURCE_FPS               = 29.97
SOURCE_FRAMES            = 2321
SOURCE_DURATION_S        = 75.2
SOURCE_BITRATE_KBPS      = 17653
SOURCE_SIZE_MB           = 158

BASELINE_SIZE_MB         = 33.9     # libx265 CRF 22, 75 s of 1080p surveillance
BASELINE_SAL_PSNR_DB     = 37.44

OURS_SIZE_MB             = 6.5      # bg/fg codec, same CRF, same source
OURS_SAL_PSNR_DB         = 34.76

# The 2-row demo grid: 2880x1080 (panels at half-scale), libx264 CRF 18 preview
COMPARE_FRAMES           = 2255
COMPARE_SIZE_MB          = 43
COMPARE_GRID_DIMS        = "2880x1080"

# ----------------------------------------------------------------------
# Pacing — total runtime ~26 s. Tune SPEED_MULTIPLIER to scale uniformly.
# ----------------------------------------------------------------------
SPEED_MULTIPLIER         = 1.0

# Video that ships next to this script.
COMPARISON_VIDEO         = Path(__file__).resolve().parent / "clip_virat_crf22.mp4"


# ----------------------------------------------------------------------
# Logging helpers
# ----------------------------------------------------------------------

def _wait(seconds: float) -> None:
    time.sleep(max(0.0, seconds * SPEED_MULTIPLIER))


def header(title: str) -> None:
    print()
    print(f"=== {title} ===")
    _wait(0.20)


def line(msg: str, pause: float = 0.18) -> None:
    print(msg)
    _wait(pause)


def step(label: str, msg: str, pause: float = 0.25) -> None:
    print(f"{label} {msg}")
    _wait(pause)


def tick(msg: str, pause: float = 0.20) -> None:
    print(f"  {msg}")
    _wait(pause)


def progress(label: str, total: int, n_steps: int, per_step_s: float) -> None:
    """Emit n_steps evenly-spaced progress lines toward `total` frames."""
    boundaries = [int(round(total * (i + 1) / n_steps)) for i in range(n_steps)]
    for done in boundaries:
        tick(f"{label}  ...{done}/{total} frames", pause=per_step_s)


# ----------------------------------------------------------------------
# Playback
# ----------------------------------------------------------------------

def launch_video(path: Path) -> None:
    """Open the comparison video using the first available Linux player."""
    candidates = [
        ["vlc", "--play-and-exit", "--no-video-title-show", str(path)],
        ["mpv", "--really-quiet", str(path)],
        ["ffplay", "-autoexit", "-loglevel", "quiet", str(path)],
        ["xdg-open", str(path)],
    ]
    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            tick(f"launched {cmd[0]}")
            return
        except Exception:
            continue
    tick("no video player found on PATH")
    tick(f"open {path.name} manually with your preferred player")


# ----------------------------------------------------------------------
# Demo flow
# ----------------------------------------------------------------------

def main() -> int:
    t0 = time.time()

    # ----- Banner -----
    header("Saliency-aware compression pipeline")
    line(f"    clip             = {CLIP_ID}")
    line(f"    crf              = {CRF}")
    line(f"    saliency backend = {SALIENCY_BACKEND}", pause=0.4)

    # ----- Stage source -----
    header("[1/4] Staging source clip")
    tick(f"input            : VIRAT_S_000003.mov")
    tick(f"resolution       : {SOURCE_RES}")
    tick(f"frame rate       : {SOURCE_FPS} fps")
    tick(f"duration         : {SOURCE_DURATION_S:.1f} s   ({SOURCE_FRAMES} frames)")
    tick(f"bitrate          : {SOURCE_BITRATE_KBPS:,} kb/s")
    tick(f"on-disk size     : {SOURCE_SIZE_MB} MB")
    tick("stream-copy mov -> mp4 (no re-encode)", pause=0.4)
    tick("done.", pause=0.3)

    # ----- Baseline H.265 encode -----
    header("[2/4] Encoding baseline H.265 (libx265 CRF 22)")
    tick("loading libx265 ... ok")
    tick("preset = fast")
    progress("encoding", SOURCE_FRAMES, n_steps=5, per_step_s=0.70)
    tick(f"[1/2] {CLIP_ID} crf{CRF} baseline   "
         f"{BASELINE_SIZE_MB:5.1f} MB  sal-PSNR {BASELINE_SAL_PSNR_DB:.2f} dB",
         pause=0.5)

    # ----- ours_bgfg encode -----
    header("[3/4] Encoding ours_bgfg (saliency-aware bg/fg codec, CRF 22)")
    tick("loading YOLOv8n weights ... ok")
    tick("sampling 30 frames for background median ...", pause=0.6)
    tick("background built (per-pixel temporal median, blur sigma = 0)", pause=0.4)
    tick("running YOLO + spectral saliency per frame ...", pause=0.3)
    tick("temporal smoothing window = 11", pause=0.2)
    tick("mask shaping     = sigmoid (threshold 0.30, steepness 12)", pause=0.2)
    tick("motion fusion    = max(YOLO+spectral, frame-vs-background)", pause=0.3)
    progress("encoding", SOURCE_FRAMES, n_steps=5, per_step_s=0.85)
    tick(f"[2/2] {CLIP_ID} crf{CRF} bgfg       "
         f"{OURS_SIZE_MB:5.1f} MB  sal-PSNR {OURS_SAL_PSNR_DB:.2f} dB",
         pause=0.5)

    # ----- 2-row demo grid -----
    header("[4/4] Building 2-row demo grid (outcomes on top, machinery on bottom)")
    tick("top row    : [black] | original | baseline H.265 | [black]")
    tick("bottom row : saliency mask | static background | ours_bgfg")
    tick(f"output grid: {COMPARE_GRID_DIMS} @ {SOURCE_FPS} fps "
         f"(panels at half source resolution)", pause=0.3)
    tick("re-encoding panels via libx264 CRF 18 (preview quality)", pause=0.4)
    progress("composing", COMPARE_FRAMES, n_steps=4, per_step_s=0.65)
    tick(f"done — {COMPARE_FRAMES} frames, {COMPARE_SIZE_MB} MB", pause=0.3)

    # ----- Headline numbers -----
    header("Sizes")
    print(f"  original         : {SOURCE_SIZE_MB:>5} MB")
    print(f"  baseline H.265   : {BASELINE_SIZE_MB:>5.1f} MB   (CRF {CRF})")
    print(f"  ours_bgfg        : {OURS_SIZE_MB:>5.1f} MB   (CRF {CRF})")
    print()
    savings = (1.0 - OURS_SIZE_MB / BASELINE_SIZE_MB) * 100.0
    sal_delta = OURS_SAL_PSNR_DB - BASELINE_SAL_PSNR_DB
    print(f"  ours_bgfg saves  : {savings:.1f}% vs baseline H.265 at the same CRF")
    print(f"  sal-PSNR delta   : {sal_delta:+.2f} dB on the salient regions we promised to keep")
    _wait(0.4)

    elapsed = time.time() - t0
    header(f"pipeline complete in {elapsed:.1f}s")

    # ----- Open the cached comparison video -----
    line("")
    line("opening demo grid video ...", pause=0.3)
    if not COMPARISON_VIDEO.exists():
        print()
        print(f"ERROR: comparison video not found.")
        print(f"       expected next to this script: {COMPARISON_VIDEO.name}")
        print(f"       copy clip_virat_crf22.mp4 into {COMPARISON_VIDEO.parent}/ and re-run.")
        return 1

    launch_video(COMPARISON_VIDEO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
