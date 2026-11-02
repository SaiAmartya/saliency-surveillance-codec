"""
H.265 encoding helpers.

After the May 2026 cleanup that removed the legacy `ours_sigmoid` pre-blur
pipeline, this module only provides:

  - `_build_alpha`        : saliency-mask shaping function (sigmoid / binary /
                            alpha). Used by `bg_fg_codec.BgFgCodec._blend_frame`
                            and the saliency-overlay panel in `compare_clip.py`
                            so both renders stay visually consistent with the
                            mask the encoder actually sees.
  - `H265EncoderConfig`   : codec / crf / preset / max-GOP bundle.
  - `FramePipeEncoder`    : raw BGR frames in → H.265 mp4 out via an ffmpeg
                            stdin pipe. Used by `BgFgCodec` as its underlying
                            stream encoder.
  - `encode_uniform`      : transcode an existing video file to H.265 at the
                            chosen CRF. Used by `ablation_bgfg.py`,
                            `photo_demo.py`, `bench_external.py`, etc., as the
                            apples-to-apples baseline.

`SaliencyCompressor` / `CompressorConfig` / `_tier_c_blend` (the Tier C
pre-blur path) were removed alongside `src/pipeline.py` and
`scripts/run_pipeline.py`. The bg/fg codec is the active encoder now.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Saliency-mask shaping (kept after the cleanup — bg_fg_codec + compare_clip use it)
# ---------------------------------------------------------------------------

def _build_alpha(sal: np.ndarray, mode: str, threshold: float, steepness: float) -> np.ndarray:
    """Convert a continuous saliency map into a per-pixel sharp/blur weight.
    Output: HxW float32 in [0,1]. 1 = keep frame fully, 0 = fall back to background."""
    if mode == "alpha":
        return np.clip(sal, 0.0, 1.0).astype(np.float32)
    if mode == "binary":
        return (sal >= threshold).astype(np.float32)
    if mode == "sigmoid":
        # Sharp logistic transition centred at threshold. Saturates near 0/1
        # so salient interior is fully kept and non-salient interior fully
        # replaced, but the boundary is smooth (no hard edges that would
        # create artefacts).
        return 1.0 / (1.0 + np.exp(-steepness * (sal - threshold))).astype(np.float32)
    raise ValueError(f"Unknown mask_mode: {mode!r}")


# ---------------------------------------------------------------------------
# H.265 encoder used by bg_fg_codec
# ---------------------------------------------------------------------------

# Max seconds between I-frames. x265's default is 250 frames (~8 s at 30 fps),
# so a stationary camera re-sends its whole background as an intra frame every
# few seconds — and once the background is pinned, those I-frames are most of
# the bg/fg stream's bytes. The encoder still cuts a new I-frame on a scene
# change (scenecut stays on), so a long cap only stretches the GOP while the
# scene is static: the "dynamic GOP" that surveillance encoders such as Axis
# Zipstream and Hikvision H.265+ rely on. The cost is coarser seeking.
DEFAULT_KEYINT_S = 20.0


@dataclass
class H265EncoderConfig:
    codec: str = "libx265"        # libx265 or libx264
    crf: int = 28                 # baseline quality (higher = smaller)
    preset: str = "fast"
    keyint_s: Optional[float] = DEFAULT_KEYINT_S   # None = encoder default GOP


def _codec_args(codec: str, preset: str, crf: int, keyint_s: Optional[float], fps: float) -> list[str]:
    """Shared encoder flags, so the baseline and bg/fg encodes stay apples-to-apples."""
    args = ["-c:v", codec, "-preset", preset, "-crf", f"{crf}", "-pix_fmt", "yuv420p"]
    if keyint_s:
        args += ["-g", str(max(1, round(keyint_s * fps)))]
    # macOS QuickTime requires the hvc1 codec tag for H.265-in-MP4 playback;
    # ffmpeg defaults to hev1 which QuickTime refuses. Same bitstream either
    # way — just a metadata flag in the container. Apply only for libx265
    # (libx264 / other codecs would reject this tag).
    if codec in ("libx265", "hevc"):
        args += ["-tag:v", "hvc1"]
    return args


class FramePipeEncoder:
    """Pipe a stream of raw BGR frames to ffmpeg for H.265 encoding.

    Used by `BgFgCodec` to take the already-stabilised foreground-on-background
    frames and emit a valid H.265 mp4.
    """

    def __init__(self, cfg: Optional[H265EncoderConfig] = None) -> None:
        self.cfg = cfg or H265EncoderConfig()

    def encode(
        self,
        frames_iter: Iterable[np.ndarray],
        out_path: str,
        fps: float,
        size: tuple[int, int],
    ) -> dict:
        w, h = size
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", f"{fps}",
            "-i", "-",
            *_codec_args(self.cfg.codec, self.cfg.preset, self.cfg.crf, self.cfg.keyint_s, fps),
            out_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None and proc.stderr is not None

        n = 0
        try:
            for f in frames_iter:
                if f.shape[1] != w or f.shape[0] != h:
                    f = cv2.resize(f, (w, h))
                proc.stdin.write(f.tobytes())
                n += 1
        except BrokenPipeError:
            pass  # ffmpeg exited early; its stderr (raised below) says why
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            err = proc.stderr.read().decode(errors="replace").strip()
            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with code {proc.returncode}: {err[-800:]}")

        return {
            "out_path": out_path,
            "n_frames": n,
            "bytes": os.path.getsize(out_path) if os.path.exists(out_path) else 0,
        }


# ---------------------------------------------------------------------------
# Baseline transcode (used by ablation_bgfg, photo_demo, bench_external, ...)
# ---------------------------------------------------------------------------

def encode_uniform(
    input_path: str,
    out_path: str,
    crf: int = 28,
    codec: str = "libx265",
    preset: str = "fast",
    keyint_s: Optional[float] = DEFAULT_KEYINT_S,
) -> dict:
    """Baseline: re-encode an existing video uniformly, no saliency awareness.

    Uses the same encoder flags (GOP cap included) as `FramePipeEncoder`, and
    drops audio (`-an`): the bg/fg encode is video-only, so a baseline that
    kept the source's audio track would inflate the reported savings.
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", input_path, "-an",
        *_codec_args(codec, preset, crf, keyint_s, fps),
        out_path,
    ]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}: {err[-800:]}")
    return {"out_path": out_path, "bytes": os.path.getsize(out_path)}


if __name__ == "__main__":
    # Smoke test: pipe 30 noisy frames through the H.265 encoder.
    rng = np.random.default_rng(0)
    frames = [(rng.random((240, 320, 3)) * 255).astype(np.uint8) for _ in range(30)]
    enc = FramePipeEncoder()
    stats = enc.encode(iter(frames), "/tmp/compress_smoke.mp4", fps=10.0, size=(320, 240))
    print("compress.py smoke:", stats)
