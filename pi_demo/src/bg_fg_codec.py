"""
Background/foreground decomposition codec for stationary surveillance cameras.

This is the headline innovation: surveillance cameras don't move, so 95% of
every frame is identical to the previous one. A general-purpose codec (H.265)
still has to spend bits modelling that redundancy with motion vectors and
residuals. We exploit the stationary-camera assumption explicitly:

  1. Pass 1 — build a "background reference" image from the temporal median
              of N evenly-spaced frames. Robust to short-lived foreground.
  2. Pass 2 — for each frame, replace non-salient pixels with the matching
              background pixel:
                   stabilised = mask * frame + (1 - mask) * background
              Mask comes from any SaliencyEstimator backend; we ship sigmoid
              shaping so the transition is smooth (no visible seams).
  3. H.265 — encode the stabilised stream with the existing pipeline. Because
              non-salient pixels are now BYTE-IDENTICAL across frames, H.265's
              inter-frame prediction reduces those regions to near-zero bits.
              All the bitrate flows to the foreground.

The decoded output looks like the original frame except non-salient regions
are pinned to the (sharp) background reference instead of carrying their own
noisy textures. Visually this is often *better* than baseline H.265 at the
same bitrate, because the static background never picks up compression noise.

Public API:
    BgFgCodec(saliency_backend="yolo+spectral", crf=28, ...).encode(in, out)
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from .saliency import SaliencyEstimator, TemporalSmoother
from .compress import DEFAULT_KEYINT_S, FramePipeEncoder, H265EncoderConfig, _build_alpha

logger = logging.getLogger(__name__)


@dataclass
class BgFgConfig:
    # Saliency
    saliency_backend: str = "yolo+spectral"
    saliency_yolo_kwargs: Optional[dict] = None
    smooth_window: int = 11             # temporal smoothing window for the mask
    mask_mode: str = "sigmoid"
    mask_threshold: float = 0.30
    mask_steepness: float = 12.0
    mask_floor: float = 0.25            # min blend weight on the frame — safety net so
                                        # subjects YOLO/spectral missed never fully vanish.
                                        # Trades a bit of size win for "the person is always there".

    # Motion saliency (NEW). The temporal-median background gives us a *free*
    # foreground signal: any pixel whose value differs from the background is
    # by definition moving foreground. We combine this with the chosen saliency
    # backend via per-pixel max, so motion is ALWAYS preserved even when YOLO
    # misses or spectral residual is weak. This is the key fix for the
    # "person disappeared into the bar" failure mode.
    use_motion_saliency: bool = True
    motion_threshold: int = 18          # min per-channel delta in [0,255] to count as motion
    motion_ramp: float = 35.0           # delta -> [0,1] ramp slope; smaller = more sensitive
    motion_dilate: int = 7              # px to dilate the motion mask so it covers full body

    # Background reference
    bg_sample_count: int = 30           # static mode: frames sampled across the clip for median
    bg_blur_sigma: float = 0.0          # post-median Gaussian smoothing (0 = none)

    # Rolling-median background (May 2026). Set bg_mode="rolling" to enable.
    # Mirrors the trick proved out in scripts/live_demo.py: instead of a single
    # static median over the entire clip, recompute the background every
    # bg_recalibration_interval_s using a centred window of nearby samples. The
    # codec then uses the *right* background for each segment of frames. Strong
    # win when lighting drifts, the camera is bumped, or a subject lingers in
    # the same area long enough to leak into a clip-wide median.
    #
    # Sidecar: rolling mode encodes the sequence of backgrounds as a tiny 1-fps
    # mp4 (consecutive backgrounds barely differ, so inter-frame prediction
    # collapses them) instead of saving one PNG per segment. Honest accounting
    # without blowing up total_bytes.
    bg_mode: str = "static"                          # "static" | "rolling"
    bg_recalibration_interval_s: float = 4.0         # rolling: recompute every Ns
    bg_rolling_window_s: float = 6.0                 # rolling: ± window of samples per median
    bg_rolling_sample_stride_s: float = 0.5          # rolling: walk-the-clip sample stride
    bg_rolling_thumb_long_edge_px: int = 320         # rolling: thumb size in the buffer
    bg_sidecar_crf: int = 28                         # rolling: CRF for the bg sidecar mp4

    # Encoder
    codec: str = "libx265"
    crf: int = 28
    preset: str = "fast"
    keyint_s: Optional[float] = DEFAULT_KEYINT_S     # max GOP; see compress.DEFAULT_KEYINT_S


def _sample_frame_indices(n_total: int, k: int) -> list[int]:
    """k evenly-spaced indices in [0, n_total). Falls back to all frames if k>=n."""
    if k >= n_total:
        return list(range(n_total))
    return np.linspace(0, n_total - 1, k, dtype=int).tolist()


def compute_background_median(input_path: str, n_samples: int = 30, blur_sigma: float = 0.0) -> np.ndarray:
    """Per-pixel temporal median over evenly-spaced frames. Returns HxWx3 uint8 BGR."""
    print(f"  [bg] sampling {n_samples} frames for static background median...", flush=True)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_frame_indices(n_total, n_samples)

    stack = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            stack.append(frame)
    cap.release()

    if not stack:
        raise RuntimeError(f"No frames read from {input_path}")

    print(f"  [bg] computing per-pixel median over {len(stack)} samples...", flush=True)
    arr = np.stack(stack, axis=0)  # NxHxWx3 uint8
    # median over the time axis; uint8 promoted via np to avoid overflow
    bg = np.median(arr, axis=0).astype(np.uint8)
    if blur_sigma and blur_sigma > 0:
        k = max(3, int(blur_sigma * 6) | 1)  # odd kernel
        bg = cv2.GaussianBlur(bg, (k, k), blur_sigma)
    print(f"  [bg] static background ready ({bg.shape[1]}x{bg.shape[0]})", flush=True)
    return bg


# ---------- Rolling-median background schedule ----------
#
# Two shapes are accepted everywhere downstream:
#   - np.ndarray                                  → single static background
#   - list[tuple[int, int, np.ndarray]]           → schedule of (start_frame, end_frame, bg)
# `bg_for_frame(source, frame_idx)` resolves either shape to the right bg.

BgSource = "np.ndarray | list[tuple[int, int, np.ndarray]]"


def bg_for_frame(source, frame_idx: int) -> np.ndarray:
    """Pick the background image that applies to a given frame index."""
    if isinstance(source, np.ndarray):
        return source
    for start, end, bg in source:
        if start <= frame_idx < end:
            return bg
    return source[-1][2]  # past the last segment → use the last segment's bg


def build_rolling_backgrounds(
    input_path: str,
    recalibration_interval_s: float = 4.0,
    rolling_window_s: float = 6.0,
    sample_stride_s: float = 0.5,
    thumb_long_edge_px: int = 320,
) -> list[tuple[int, int, np.ndarray]]:
    """Walk the clip once, build a schedule of per-segment backgrounds.

    Algorithm:
      1. Open the video once, read frame-by-frame, and every `sample_stride_s`
         of wall time capture a downscaled thumbnail (long edge ≤ thumb_long_edge_px).
      2. Slice the clip into segments of length `recalibration_interval_s`.
      3. For each segment, take all thumbs whose timestamp falls in a ± window
         (window_s/2) centred on the segment midpoint, compute the per-pixel
         median across that window, upscale back to source resolution.
      4. Return [(start_frame, end_frame, bg_full_res), ...].

    The centred window means each segment's background sees both past AND future
    samples — the codec is offline, so we can take advantage. Subjects that move
    around are rejected (they're in different positions across the window);
    subjects that linger in one place for more than half the window will leak
    in (same trade as the codec's static median, but localised in time).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stride_frames = max(1, int(round(sample_stride_s * fps)))
    scale = thumb_long_edge_px / float(max(src_w, src_h))
    tw = max(2, int(src_w * scale)) & ~1
    th = max(2, int(src_h * scale)) & ~1

    # Pass 1: collect (timestamp_s, thumb)
    print(
        f"  [bg-rolling] pass 1/2: walking {n_total} frames, sampling every "
        f"{stride_frames} ({sample_stride_s:.1f}s) → thumb {tw}x{th}...",
        flush=True,
    )
    progress_every = max(1, n_total // 10) if n_total > 0 else 200
    samples: list[tuple[float, np.ndarray]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride_frames == 0:
            samples.append((idx / fps, cv2.resize(frame, (tw, th))))
        idx += 1
        if n_total > 0 and idx % progress_every == 0:
            print(f"    walked {idx}/{n_total} frames  ({len(samples)} samples buffered)", flush=True)
    cap.release()
    if not samples:
        raise RuntimeError(f"No samples collected from {input_path}")
    print(f"  [bg-rolling] pass 1 done: {len(samples)} sample thumbs collected", flush=True)

    total_duration_s = (n_total / fps) if n_total > 0 else (samples[-1][0] + sample_stride_s)
    n_segments = max(1, int(np.ceil(total_duration_s / recalibration_interval_s)))
    half_window = rolling_window_s / 2.0
    print(
        f"  [bg-rolling] pass 2/2: building {n_segments} per-segment medians "
        f"(every {recalibration_interval_s:.1f}s, ±{half_window:.1f}s window)...",
        flush=True,
    )

    schedule: list[tuple[int, int, np.ndarray]] = []
    seg_progress_every = max(1, n_segments // 10)
    for i in range(n_segments):
        seg_start_s = i * recalibration_interval_s
        seg_end_s = (i + 1) * recalibration_interval_s
        center_s = seg_start_s + recalibration_interval_s / 2.0
        window_lo, window_hi = center_s - half_window, center_s + half_window

        window_thumbs = [t for (ts, t) in samples if window_lo <= ts <= window_hi]
        if not window_thumbs:
            # Fall back to nearest available thumb if the window underflows
            window_thumbs = [min(samples, key=lambda kv: abs(kv[0] - center_s))[1]]

        median_thumb = np.median(np.stack(window_thumbs, axis=0), axis=0).astype(np.uint8)
        bg_full = cv2.resize(median_thumb, (src_w, src_h))

        seg_start_frame = max(0, int(round(seg_start_s * fps)))
        seg_end_frame = int(round(seg_end_s * fps)) if i < n_segments - 1 else n_total
        schedule.append((seg_start_frame, seg_end_frame, bg_full))
        if (i + 1) % seg_progress_every == 0 or (i + 1) == n_segments:
            print(f"    built segment {i+1}/{n_segments}  "
                  f"(t={seg_start_s:.1f}-{seg_end_s:.1f}s, "
                  f"{len(window_thumbs)} thumbs in window)", flush=True)

    print(
        f"  [bg-rolling] schedule ready: {len(schedule)} segments × "
        f"{src_w}x{src_h}  (sidecar will be a 1-fps mp4 of these)",
        flush=True,
    )
    return schedule


class BgFgCodec:
    """Background/foreground codec built on top of the existing H.265 pipeline.

    The output mp4 is a valid H.265 stream — decoders don't need to know we
    pre-stabilised the input. Numbers reported (PSNR / SSIM / bytes) are
    apples-to-apples against the baseline H.265 encode of the same clip.
    """

    def __init__(self, cfg: Optional[BgFgConfig] = None) -> None:
        self.cfg = cfg or BgFgConfig()
        self.saliency = SaliencyEstimator(
            backend=self.cfg.saliency_backend,
            yolo_kwargs=self.cfg.saliency_yolo_kwargs,
        )
        self.smoother = TemporalSmoother(window=self.cfg.smooth_window)

        encoder_cfg = H265EncoderConfig(
            codec=self.cfg.codec,
            crf=self.cfg.crf,
            preset=self.cfg.preset,
            keyint_s=self.cfg.keyint_s,
        )
        self._encoder = FramePipeEncoder(encoder_cfg)

    # ---------- Core preprocessing ----------

    def _motion_mask(self, frame: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Per-pixel motion signal from |frame - background|. Returns HxW float [0,1].

        Anything that differs from the static background is foreground by definition.
        This catches subjects the semantic saliency missed (dark figures on dark
        backgrounds, low-confidence YOLO detections, etc.).
        """
        diff_bgr = np.abs(frame.astype(np.int16) - background.astype(np.int16))
        # max across BGR channels — a change in any channel counts
        diff = diff_bgr.max(axis=2).astype(np.float32)
        # Subtract threshold, then ramp to [0,1]
        ramp = (diff - self.cfg.motion_threshold) / max(self.cfg.motion_ramp, 1.0)
        m = np.clip(ramp, 0.0, 1.0)
        # Dilate so the mask covers the full body, not just edges
        if self.cfg.motion_dilate > 0:
            k = max(3, self.cfg.motion_dilate)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            # Convert to uint8 for dilation, back to float
            m_u8 = (m * 255).astype(np.uint8)
            m_u8 = cv2.dilate(m_u8, kernel)
            # Gentle blur the boundary so the mask edge is smooth
            m_u8 = cv2.GaussianBlur(m_u8, (k | 1, k | 1), 0)
            m = m_u8.astype(np.float32) / 255.0
        return m

    def _blend_frame(self, frame: np.ndarray, background: np.ndarray, sal: np.ndarray) -> np.ndarray:
        """Replace low-saliency pixels with background. Returns HxWx3 uint8."""
        if sal.shape != frame.shape[:2]:
            sal = cv2.resize(sal, (frame.shape[1], frame.shape[0]))
        alpha = _build_alpha(
            sal, self.cfg.mask_mode, self.cfg.mask_threshold, self.cfg.mask_steepness
        )
        if self.cfg.mask_floor > 0:
            alpha = np.maximum(alpha, self.cfg.mask_floor)
        alpha3 = alpha[..., None]
        out = alpha3 * frame.astype(np.float32) + (1.0 - alpha3) * background.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _stabilised_iter(self, input_path: str, bg_source):
        """Yield blended (foreground-on-background) frames in stream order.

        `bg_source` is either:
          - np.ndarray            → single static background (resized if needed)
          - list of segment tuples → rolling schedule; `bg_for_frame` picks the
                                     right segment for each frame index.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {input_path}")
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # If we got a single static bg, resize once up front. The rolling
        # schedule's backgrounds are already at source resolution.
        if isinstance(bg_source, np.ndarray) and bg_source.shape[:2] != (h, w):
            bg_source = cv2.resize(bg_source, (w, h))

        # Aim for ~20 progress lines across the whole encode so the GUI log
        # streams visibly without spamming. Floor at 30 frames so very short
        # clips still tick at least a few times.
        progress_every = max(30, n_total // 20) if n_total > 0 else 60
        print(
            f"  [encode] streaming {n_total} frames @ {fps:.2f} fps through "
            f"saliency + bg/fg blend → libx265 ...",
            flush=True,
        )
        t_start = time.time()

        self.smoother.reset()
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            bg = bg_for_frame(bg_source, frame_idx)
            sal_raw = self.saliency.predict(frame)
            # Combine with motion-from-background-difference: per-pixel max.
            # This is the key fix — anything that's moving is foreground even
            # if YOLO/spectral didn't fire on it.
            if self.cfg.use_motion_saliency:
                motion = self._motion_mask(frame, bg)
                sal_raw = np.maximum(sal_raw, motion)
            sal = self.smoother.smooth(sal_raw)
            yield self._blend_frame(frame, bg, sal)
            frame_idx += 1
            if frame_idx % progress_every == 0:
                elapsed = time.time() - t_start
                cur_fps = frame_idx / max(elapsed, 1e-6)
                eta = (n_total - frame_idx) / max(cur_fps, 1e-6) if n_total else 0
                pct = (100.0 * frame_idx / n_total) if n_total else 0
                print(
                    f"    encoded {frame_idx}/{n_total} frames  "
                    f"({pct:5.1f}%, {cur_fps:5.1f} fps, ETA {eta:5.1f}s)",
                    flush=True,
                )

        cap.release()
        elapsed = time.time() - t_start
        cur_fps = frame_idx / max(elapsed, 1e-6)
        print(
            f"  [encode] done — {frame_idx} frames in {elapsed:.1f}s "
            f"({cur_fps:.1f} fps avg)",
            flush=True,
        )

    # ---------- Public encode ----------

    def encode(self, input_path: str, output_path: str) -> dict:
        """Two-pass encode: build background(s), then emit stabilised frames to H.265.

        Returns: {out_path, bytes, n_frames, background_path, background_bytes,
                  total_bytes, bg_mode, ... (rolling: n_bg_segments)}.

        Sidecar packaging:
          - bg_mode='static'  : single PNG saved alongside (`<out>_bg.png`).
          - bg_mode='rolling' : the schedule's backgrounds get encoded as a
                                tiny 1-fps mp4 (`<out>_bgseq.mp4`). Since
                                consecutive backgrounds barely differ,
                                libx265 inter-frame prediction collapses the
                                whole sequence into ~1–3 MB total instead of
                                the N × PNG bloat you'd get otherwise.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Build background(s) according to mode.
        if self.cfg.bg_mode == "rolling":
            print(
                f"  [BgFgCodec] mode=rolling, recalibrate every "
                f"{self.cfg.bg_recalibration_interval_s:.1f}s, ±"
                f"{self.cfg.bg_rolling_window_s/2:.1f}s median window",
                flush=True,
            )
            bg_source = build_rolling_backgrounds(
                input_path,
                recalibration_interval_s=self.cfg.bg_recalibration_interval_s,
                rolling_window_s=self.cfg.bg_rolling_window_s,
                sample_stride_s=self.cfg.bg_rolling_sample_stride_s,
                thumb_long_edge_px=self.cfg.bg_rolling_thumb_long_edge_px,
            )
            # Sidecar: encode bg sequence as a tiny 1-fps mp4 (highly compressible
            # because consecutive backgrounds are nearly identical).
            bg_path = str(Path(output_path).with_suffix("")) + "_bgseq.mp4"
            sidecar_enc = FramePipeEncoder(H265EncoderConfig(
                codec=self.cfg.codec, crf=self.cfg.bg_sidecar_crf, preset="medium",
                keyint_s=self.cfg.keyint_s,
            ))
            sidecar_enc.encode(
                (bg for _, _, bg in bg_source),
                bg_path, fps=1.0, size=(w, h),
            )
            n_segments = len(bg_source)
        elif self.cfg.bg_mode == "static":
            print(
                f"  [BgFgCodec] mode=static, single clip-wide median of "
                f"{self.cfg.bg_sample_count} samples",
                flush=True,
            )
            bg_source = compute_background_median(
                input_path,
                n_samples=self.cfg.bg_sample_count,
                blur_sigma=self.cfg.bg_blur_sigma,
            )
            bg_path = str(Path(output_path).with_suffix("")) + "_bg.png"
            cv2.imwrite(bg_path, bg_source, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            n_segments = 1
        else:
            raise ValueError(f"Unknown bg_mode: {self.cfg.bg_mode!r}. Use 'static' or 'rolling'.")

        print(f"  [BgFgCodec] encoding stabilised stream → {output_path}", flush=True)
        stats = self._encoder.encode(
            self._stabilised_iter(input_path, bg_source),
            output_path, fps=fps, size=(w, h),
        )
        stats["bg_mode"] = self.cfg.bg_mode
        stats["n_bg_segments"] = n_segments
        stats["background_path"] = bg_path
        stats["background_bytes"] = os.path.getsize(bg_path)
        # Combined "wire bytes" the receiver has to download.
        stats["total_bytes"] = stats["bytes"] + stats["background_bytes"]
        return stats


# ---------- Convenience CLI ----------

if __name__ == "__main__":
    import argparse, json, sys
    from .metrics import video_metrics

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True, help="Output mp4 path")
    parser.add_argument("--saliency", default="yolo+spectral",
                        choices=["spectral", "finegrained", "yolo", "yolo+spectral"])
    parser.add_argument("--crf", type=int, default=28)
    parser.add_argument("--bg-samples", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--steepness", type=float, default=12.0)
    parser.add_argument("--smooth", type=int, default=7)
    parser.add_argument("--bg-mode", default="static", choices=["static", "rolling"],
                        help="static (single clip-wide median, default) or "
                             "rolling (recompute background every N seconds).")
    parser.add_argument("--bg-recal-interval", type=float, default=4.0,
                        help="Rolling mode: recalibration interval in seconds (default 4.0).")
    parser.add_argument("--bg-window", type=float, default=6.0,
                        help="Rolling mode: ± median window in seconds (default 6.0).")
    parser.add_argument("--keyint", type=float, default=DEFAULT_KEYINT_S,
                        help=f"Max seconds between I-frames (default {DEFAULT_KEYINT_S:g}; "
                             "0 = encoder default of 250 frames).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = BgFgConfig(
        saliency_backend=args.saliency,
        crf=args.crf,
        bg_sample_count=args.bg_samples,
        mask_threshold=args.threshold,
        mask_steepness=args.steepness,
        smooth_window=args.smooth,
        bg_mode=args.bg_mode,
        bg_recalibration_interval_s=args.bg_recal_interval,
        bg_rolling_window_s=args.bg_window,
        keyint_s=args.keyint or None,
    )
    codec = BgFgCodec(cfg)
    stats = codec.encode(args.input, args.out)

    metrics = video_metrics(args.input, args.out, every=5)
    raw = os.path.getsize(args.input)
    print(json.dumps({
        **stats,
        "raw_input_bytes": raw,
        "psnr_mean": metrics.get("psnr_mean"),
        "ssim_mean": metrics.get("ssim_mean"),
        "saving_vs_raw_input_pct": round((1 - stats["total_bytes"] / raw) * 100, 1),
    }, indent=2))
