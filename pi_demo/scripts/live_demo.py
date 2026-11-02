"""
Live webcam demo — saliency mask at the same quality as the compression pipeline.

Opens the default camera, runs the FULL bg/fg saliency pipeline on every frame
(yolo + spectral + motion-vs-background, sigmoid mask shaping, temporal
smoothing), and overlays the JET-coloured mask on the live feed. Shows the
gate's usefulness score in a HUD.

This matches the saliency-mask panel produced by `compare_clip.py --internals`
on a recorded clip — same backends, same fusion, same smoothing window, same
sigmoid threshold/steepness, same 0.25 floor, same alpha-blend weights.

Why this differs from the previous version:
    The old live_demo used `backend='spectral'` with a 5-frame smoother and no
    motion mask. Spectral residual is novelty-driven and lights up any textured
    region (TVs, posters, edges of windows) — not "the things you care about".
    The codec's saliency mask is much better because it ALSO uses YOLOv8n
    semantic detection (knows people / vehicles / bags / animals) AND a motion
    mask from background subtraction. Per-pixel max fuses all three so anything
    flagged by any signal counts as salient.

Calibration:
    Motion-vs-background requires a "background image" — for a recorded clip
    we take the temporal median of 30 evenly-spaced samples. For a live camera
    we capture N frames at startup (`--bg-samples`, default 60 = ~2s @ 30fps)
    with you out of frame, then use that as the initial background.

Auto-recalibration:
    Cameras don't sit still: you move around, lighting drifts, and if the
    initial calibration captured you in an unusual position (close to the
    camera, partially blocking the scene), the background image goes stale
    fast — the motion mask starts firing on huge sections of the room.

    To fix this we maintain a rolling buffer of recent frame thumbnails and
    rebuild the background every AUTO_RECAL_INTERVAL_S seconds as the
    per-pixel median of that buffer. Same trick the codec uses on a recorded
    clip — the median rejects anything moving (subjects appear in different
    positions across the window, so they don't survive), while static scene
    geometry passes through unchanged. Disable with --no-auto-recal.

Keys (focus the OpenCV window):
    q   quit
    r   manually recalibrate background (step out of frame first)
    s   save the current frame to results/live_demo_snap.png

Usage:
    python scripts/live_demo.py
    python scripts/live_demo.py --camera 1
    python scripts/live_demo.py --saliency yolo            # YOLO only, fastest
    python scripts/live_demo.py --bg-samples 90            # 3s calibration
    python scripts/live_demo.py --mask-only                # just the heatmap
    python scripts/live_demo.py --no-auto-recal            # static bg, manual 'r' only
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.gate import FootageGate
from src.saliency import SaliencyEstimator, TemporalSmoother, describe_device
from src.bg_fg_codec import BgFgCodec, BgFgConfig
from src.compress import _build_alpha


# Match the compression pipeline exactly — every constant below mirrors
# BgFgCodec / compare_clip.py so the live overlay is identical to what
# the codec sees per frame on a recorded clip.
SMOOTH_WINDOW       = 11
MASK_THRESHOLD      = 0.30
MASK_STEEPNESS      = 12.0
MASK_FLOOR          = 0.25
OVERLAY_FRAME_W     = 0.55
OVERLAY_HEAT_W      = 0.45

# Rolling-median auto-recalibration. Mirrors the codec's compute_background_median()
# trick but on a sliding window of live frames. Time-based (not frame-count) so it
# works whether the live FPS is 30 or 7.5 — the buffer accumulates whatever frames
# arrive in the wall-clock window, the median rejects motion regardless of count.
AUTO_RECAL_INTERVAL_S   = 4.0     # rebuild background every 4s of wall time
AUTO_RECAL_MIN_SAMPLES  = 15      # don't recompute until at least 15 frames buffered
AUTO_RECAL_BUFFER_CAP   = 180     # hard cap on buffered thumbs (~6s at 30fps)
AUTO_RECAL_THUMB_PX     = 320     # max long-edge dimension of buffered thumbs (memory)

SNAP_PATH           = ROOT / "results" / "live_demo_snap.png"


# ----------------------------------------------------------------------
# Background calibration
# ----------------------------------------------------------------------

def calibrate_background(cap: cv2.VideoCapture, n_samples: int, window_name: str) -> np.ndarray | None:
    """Capture N frames with a countdown overlay; return their per-pixel median."""
    samples: list[np.ndarray] = []
    print(f"Calibrating background — step OUT of frame for ~{n_samples / 30:.1f}s...")
    while len(samples) < n_samples:
        ok, frame = cap.read()
        if not ok:
            continue
        samples.append(frame)

        # On-screen countdown
        h, w = frame.shape[:2]
        secs_left = max(0.0, (n_samples - len(samples)) / 30.0)
        msg1 = "CALIBRATING BACKGROUND"
        msg2 = f"step out of frame  -  {secs_left:.1f}s left"
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0)
        cv2.putText(frame, msg1, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
        cv2.putText(frame, msg2, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.imshow(window_name, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            return None

    background = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
    print(f"Background captured ({n_samples} frames). Now step into the frame!")
    return background


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--saliency", default="yolo+spectral",
                        choices=["spectral", "finegrained", "yolo", "yolo+spectral"],
                        help="Saliency backend (default yolo+spectral — same as the codec)")
    parser.add_argument("--bg-samples", type=int, default=60,
                        help="Frames to capture for background calibration (default 60 ~= 2s)")
    parser.add_argument("--mask-only", action="store_true",
                        help="Show just the heatmap (no original-frame blend)")
    parser.add_argument("--no-auto-recal", action="store_true",
                        help="Disable rolling-median background auto-recalibration "
                             "(use the initial calibration for the whole session). "
                             "Press 'r' to recalibrate manually instead.")
    args = parser.parse_args()

    # Loud device banner BEFORE the camera even opens — if YOLO is going to
    # crawl on CPU the user should know up front, not 30 frames in.
    dev = describe_device(saliency_backend=args.saliency)
    print(f"\n=== Saliency live demo ===")
    print(f"  {dev['banner']}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    # Warm up auto-exposure / white balance
    for _ in range(15):
        cap.read()

    window_name = f"Saliency live demo — saliency = {args.saliency}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    background = calibrate_background(cap, args.bg_samples, window_name)
    if background is None:
        cap.release()
        cv2.destroyAllWindows()
        return 0

    # Same components the codec uses. We share the BgFgCodec instance only for
    # its `_motion_mask` helper (so the formula stays in lockstep with the
    # codec). The standalone SaliencyEstimator is the one we actually call.
    saliency = SaliencyEstimator(backend=args.saliency)
    motion_helper = BgFgCodec(BgFgConfig(saliency_backend=args.saliency))
    smoother = TemporalSmoother(window=SMOOTH_WINDOW)
    gate = FootageGate(threshold=0.35, enable_person=False)

    # Resize background to the camera's frame size if they differ
    ok, probe = cap.read()
    if ok and background.shape[:2] != probe.shape[:2]:
        background = cv2.resize(background, (probe.shape[1], probe.shape[0]))

    auto_recal = not args.no_auto_recal
    print("Live demo running. Press 'q' to quit, 'r' to recalibrate, 's' to snap.")
    if auto_recal:
        print(f"Auto-recalibration: rolling-median background, "
              f"every {AUTO_RECAL_INTERVAL_S:.0f}s.")

    last_log = time.time()
    fps_acc = 0
    fps_disp = 0.0

    # Rolling-median auto-recal state. Buffer + dimensions are lazily sized
    # on the first frame so they match the real camera resolution.
    thumb_buffer: deque = deque(maxlen=AUTO_RECAL_BUFFER_CAP)
    thumb_dims: tuple[int, int] | None = None
    last_recal_t = time.time()
    recal_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # === The actual codec saliency pipeline ===
        sal_raw = saliency.predict(frame)
        motion = motion_helper._motion_mask(frame, background)
        sal_raw = np.maximum(sal_raw, motion)
        sal_s = smoother.smooth(sal_raw)
        alpha = _build_alpha(sal_s, mode="sigmoid",
                             threshold=MASK_THRESHOLD, steepness=MASK_STEEPNESS)
        alpha = np.maximum(alpha, MASK_FLOOR)

        # === Rolling-median auto-recalibration ===
        # Append a downscaled thumbnail of every incoming frame to a sliding
        # buffer. Every AUTO_RECAL_INTERVAL_S of wall time, replace the
        # background with the per-pixel median of that buffer. The median
        # rejects subjects (they appear in different positions across the
        # window) while static geometry survives — same trick the codec
        # uses on a recorded clip, just done online over recent history.
        if auto_recal:
            if thumb_dims is None:
                fh, fw = frame.shape[:2]
                scale = AUTO_RECAL_THUMB_PX / float(max(fw, fh))
                tw = max(2, int(fw * scale)) & ~1
                th = max(2, int(fh * scale)) & ~1
                thumb_dims = (tw, th)
            thumb_buffer.append(cv2.resize(frame, thumb_dims))

            if (time.time() - last_recal_t) >= AUTO_RECAL_INTERVAL_S \
               and len(thumb_buffer) >= AUTO_RECAL_MIN_SAMPLES:
                median_thumb = np.median(np.stack(thumb_buffer, axis=0),
                                         axis=0).astype(np.uint8)
                background = cv2.resize(median_thumb,
                                        (frame.shape[1], frame.shape[0]))
                smoother = TemporalSmoother(window=SMOOTH_WINDOW)  # reset
                last_recal_t = time.time()
                recal_count += 1
                # Quiet logging: first few then every 10th to avoid spam
                if recal_count <= 3 or recal_count % 10 == 0:
                    print(f"  [auto-recal] background refreshed "
                          f"(count={recal_count}, window={len(thumb_buffer)} frames)")

        # JET overlay (same weights as compare_clip)
        alpha_u8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(alpha_u8, cv2.COLORMAP_JET)
        if args.mask_only:
            display = heat
        else:
            display = cv2.addWeighted(frame, OVERLAY_FRAME_W, heat, OVERLAY_HEAT_W, 0.0)

        # === HUD ===
        score = gate.step(frame)
        color = (0, 220, 0) if score.triggered else (130, 130, 130)
        status = "RECORDING" if score.triggered else "idle"
        cv2.putText(display, f"{status}   useful={score.usefulness:.2f}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(display, f"saliency={args.saliency}   smoother={SMOOTH_WINDOW}f",
                    (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        cv2.putText(display, f"motion={score.motion:.2f}  flow={score.flow_magnitude:.1f}  fps={fps_disp:.1f}",
                    (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if auto_recal:
            next_in = max(0.0, AUTO_RECAL_INTERVAL_S - (time.time() - last_recal_t))
            cv2.putText(display,
                        f"auto-recal: {recal_count}  next in {next_in:.1f}s  "
                        f"(buffer={len(thumb_buffer)} thumbs)",
                        (12, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)
        cv2.putText(display, "q quit  /  r recalibrate  /  s snap",
                    (12, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(window_name, display)

        # FPS tracker
        fps_acc += 1
        now = time.time()
        if now - last_log >= 1.0:
            fps_disp = fps_acc / (now - last_log)
            fps_acc = 0
            last_log = now

        # === Key handling ===
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            background = calibrate_background(cap, args.bg_samples, window_name)
            if background is None:
                break
            if background.shape[:2] != frame.shape[:2]:
                background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
            smoother = TemporalSmoother(window=SMOOTH_WINDOW)   # reset history
            thumb_buffer.clear()                                 # drop stale thumbs
            last_recal_t = time.time()                           # restart auto-recal cooldown
        if key == ord("s"):
            SNAP_PATH.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(SNAP_PATH), display)
            print(f"Saved snap to {SNAP_PATH}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
