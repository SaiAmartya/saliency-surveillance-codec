"""
Simulate realistic long-form surveillance footage and measure compression
savings on the bg/fg codec.

The existing 5–10 second action-heavy clips are the WORST case for our codec
because every frame has motion. Real surveillance is the opposite: 95%+ of
frames show an empty scene, with brief bursts of action. This script builds a
synthetic clip of length T that interleaves:

    [N seconds idle background] + [action burst] + [M seconds idle background]
    [...optionally repeated...]

…and runs both baseline H.265 and ours_bgfg on it so you can quote a
realistic surveillance-density compression number.

The "idle background" is the temporal-median image from the action clip,
written N seconds long with tiny per-pixel noise so H.265 doesn't completely
optimize it to nothing (matching a real CCTV stream where sensor noise is
always present).

Usage:
    # 5-minute "hour-like" sim with one 10s action burst in the middle
    python scripts/simulate_hour.py --clip clip_14 --minutes 5

    # Customise: 10 minutes total, 3 action bursts, low CRF for high quality
    python scripts/simulate_hour.py --clip clip_14 --minutes 10 --bursts 3 --crf 28

    # Just print the projected savings without encoding
    python scripts/simulate_hour.py --clip clip_14 --minutes 60 --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.bg_fg_codec import BgFgCodec, BgFgConfig, compute_background_median
from src.compress import encode_uniform
from src.metrics import saliency_weighted_psnr
from src.saliency import SaliencyEstimator

DATA = ROOT / "data/real"
OUT_DIR = ROOT / "results/simulate_hour"


def synthesize_long_clip(
    action_path: Path,
    minutes: float,
    bursts: int,
    out_path: Path,
    noise_sigma: float = 1.5,
) -> tuple[int, int, float]:
    """Write a long mp4 = [idle bg] interleaved with `bursts` copies of the action clip.

    Returns (n_frames, n_action_frames, fps).
    """
    # Build background reference
    bg = compute_background_median(str(action_path), n_samples=30)
    bg_h, bg_w = bg.shape[:2]

    # Probe action clip
    cap = cv2.VideoCapture(str(action_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_action = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    total_frames = int(round(minutes * 60 * fps))
    n_action_total = n_action * bursts
    n_idle = max(0, total_frames - n_action_total)
    # Distribute action bursts evenly through the timeline
    burst_positions = np.linspace(0, total_frames - n_action - 1, bursts, dtype=int).tolist()

    # Open writer (pipe to ffmpeg rawvideo so we don't fight cv2 codec defaults)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{bg_w}x{bg_h}", "-r", f"{fps}",
        "-i", "-",
        # Encode the *source* as visually-lossless mezzanine so we measure the
        # bgfg/baseline trade fairly. CRF 14 is near-lossless.
        "-c:v", "libx264", "-preset", "fast", "-crf", "14",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None

    rng = np.random.default_rng(0)
    bg_int16 = bg.astype(np.int16)

    # Pre-read all action frames so we can splice them on demand
    cap = cv2.VideoCapture(str(action_path))
    action_frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if (f.shape[1], f.shape[0]) != (bg_w, bg_h):
            f = cv2.resize(f, (bg_w, bg_h))
        action_frames.append(f)
    cap.release()

    burst_set = set()
    for start in burst_positions:
        for i in range(n_action):
            burst_set.add(start + i)
    burst_lookup = {}
    for start in burst_positions:
        for i in range(n_action):
            burst_lookup[start + i] = i

    print(f"  writing {total_frames} frames "
          f"({minutes:.1f}min @ {fps:.0f}fps): "
          f"{n_action_total} action + {n_idle} idle  ({bursts} bursts)")
    for n in range(total_frames):
        if n in burst_set:
            f = action_frames[burst_lookup[n]]
        else:
            # Idle = background + per-pixel sensor noise (so it's not literally constant)
            noise = rng.normal(0, noise_sigma, bg.shape).astype(np.int16)
            f = np.clip(bg_int16 + noise, 0, 255).astype(np.uint8)
        proc.stdin.write(f.tobytes())
        if n % (int(fps) * 30) == 0 and n > 0:
            print(f"    ...{n}/{total_frames} ({n/fps:.0f}s)")

    proc.stdin.close()
    proc.wait()
    return total_frames, n_action_total, fps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="clip_14",
                        help="Action clip to splice into the idle timeline")
    parser.add_argument("--minutes", type=float, default=5.0,
                        help="Total duration of the synthetic clip")
    parser.add_argument("--bursts", type=int, default=1,
                        help="How many copies of the action clip to splice in")
    parser.add_argument("--crf", type=int, default=28)
    parser.add_argument("--saliency", default="spectral",
                        choices=["spectral", "finegrained", "yolo", "yolo+spectral"],
                        help="Saliency backend (spectral is safe if no YOLO)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just print the projected savings math, don't encode")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_path = DATA / f"{args.clip}.mp4"
    if not src_path.exists():
        raise SystemExit(f"Missing: {src_path}")

    sim_src = OUT_DIR / f"sim_{args.clip}_{int(args.minutes)}min.mp4"
    sim_base = OUT_DIR / f"sim_{args.clip}_{int(args.minutes)}min_baseline_crf{args.crf}.mp4"
    sim_bgfg = OUT_DIR / f"sim_{args.clip}_{int(args.minutes)}min_bgfg_crf{args.crf}.mp4"

    if args.dry_run:
        # Use the per-frame averages from the existing ablation to project
        # Idle frame size ≈ 25% of baseline minimum, action frame ≈ 73% of baseline
        cap = cv2.VideoCapture(str(src_path))
        n_action = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        total = int(args.minutes * 60 * fps)
        n_action_total = n_action * args.bursts
        n_idle = total - n_action_total
        idle_frac = n_idle / total
        action_frac = n_action_total / total
        projected = idle_frac * 0.25 + action_frac * 0.73
        print(f"\n=== Dry-run projection — {args.minutes:.0f}min sim, {args.bursts} burst(s) ===")
        print(f"  idle frames    : {n_idle}  ({idle_frac*100:.2f}%)")
        print(f"  action frames  : {n_action_total}  ({action_frac*100:.2f}%)")
        print(f"  projected size : {projected*100:.1f}% of baseline  "
              f"({(1-projected)*100:.1f}% smaller)")
        return

    print(f"=== Synthesizing {args.minutes}-min surveillance simulation ===")
    print(f"  action clip: {args.clip}  ({args.bursts} burst(s) spliced in)")
    n_total, n_action_total, fps = synthesize_long_clip(
        src_path, args.minutes, args.bursts, sim_src,
    )
    print(f"  source written: {sim_src.stat().st_size/1024/1024:.1f} MB")

    print(f"\n  encoding baseline H.265 at CRF {args.crf}...")
    encode_uniform(str(sim_src), str(sim_base), crf=args.crf)
    base_size = sim_base.stat().st_size

    print(f"  encoding ours_bgfg at CRF {args.crf} (saliency={args.saliency})...")
    cfg = BgFgConfig(saliency_backend=args.saliency, crf=args.crf)
    BgFgCodec(cfg).encode(str(sim_src), str(sim_bgfg))
    bgfg_size = sim_bgfg.stat().st_size

    # Per-second numbers (the headline)
    duration_s = n_total / fps
    base_bps = base_size * 8 / duration_s
    bgfg_bps = bgfg_size * 8 / duration_s

    print("\n=== RESULT ===")
    print(f"  duration         : {duration_s:.0f}s ({duration_s/60:.1f}min, "
          f"{n_action_total/n_total*100:.1f}% action)")
    print(f"  baseline H.265   : {base_size/1024/1024:7.2f} MB   ({base_bps/1000:.1f} kbps)")
    print(f"  ours_bgfg        : {bgfg_size/1024/1024:7.2f} MB   ({bgfg_bps/1000:.1f} kbps)")
    print(f"  savings          : {(1-bgfg_size/base_size)*100:+.1f}%")
    print(f"\n  source:   {sim_src}")
    print(f"  baseline: {sim_base}")
    print(f"  bgfg:     {sim_bgfg}")


if __name__ == "__main__":
    main()
