"""
Per-frame saliency estimation.

Backends:
  - "spectral"    : OpenCV StaticSaliencySpectralResidual (Hou & Zhang 2007).
                    Classical, no training, runs at 100+ fps on a laptop CPU.
                    Falls back to an equivalent NumPy port when the contrib
                    `cv2.saliency` module is missing.
  - "finegrained" : OpenCV StaticSaliencyFineGrained. Edge/contour-aware.
  - "yolo"        : Semantic saliency from YOLOv8n detections — Gaussian-blurred
                    soft masks around boxes weighted by confidence. Picks up
                    people, vehicles, bags, animals, etc. *Recommended for
                    surveillance*: spectral residual is novelty-driven (it
                    fires on cluttered backgrounds and high-contrast static
                    objects), whereas the things we actually care about in a
                    security feed are semantic — humans and what they carry.
  - "yolo+spectral": Combine YOLO (semantic) and spectral residual (low-level).
                     Takes the per-pixel max so we keep both. Robust fallback
                     when YOLO finds nothing in a frame.
  - "model"       : hook for a pretrained saliency CNN (TASED-Net/UNISAL).
                    Stub; not wired up.

Public API:
  SaliencyEstimator(backend="spectral")
  .predict(frame_bgr: np.ndarray) -> np.ndarray          # HxW float32 in [0,1]
  .predict_batch(frames) -> np.ndarray                    # NxHxW float32

Temporal smoothing (optional) is provided by TemporalSmoother — used by the
pipeline to avoid flickering QP maps between adjacent frames.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Iterable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# COCO class IDs that matter for home/perimeter surveillance.
# Defaults below cover people, common vehicles, bags, animals, and
# objects often associated with intrusion/incident events.
SURVEILLANCE_COCO_CLASSES: tuple[int, ...] = (
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    24,  # backpack
    26,  # handbag
    28,  # suitcase
    39,  # bottle (often weapons proxy / object of interest)
    43,  # knife
    63,  # laptop
    67,  # cell phone
    73,  # book
)


class YoloSaliency:
    """Turn YOLOv8n detections into a smooth per-pixel saliency map.

    For each detection box: place a 2D Gaussian centered at the box, with
    sigma proportional to the box's diagonal, scaled by detection confidence.
    Sum across boxes, normalise to [0,1]. Returns a flat low-intensity map if
    no detections fire — never all-zeros, so downstream blur stays bounded.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        classes: Optional[Iterable[int]] = SURVEILLANCE_COCO_CLASSES,
        conf_threshold: float = 0.10,
        sigma_scale: float = 0.55,
        infer_every: int = 1,
        fallback_floor: float = 0.05,
        strict: bool = True,
        device: str = "auto",
    ) -> None:
        """
        Args:
            weights: Path or name of YOLO checkpoint (yolov8n.pt is the default).
            classes: COCO class IDs to keep. None = keep everything.
            conf_threshold: Drop detections below this confidence.
            sigma_scale: Gaussian sigma = sigma_scale * sqrt(box_w * box_h).
                         Larger = softer mask edges around each object.
            infer_every: Run inference only every N frames; reuse the previous
                         mask in between. infer_every=1 is full per-frame.
            fallback_floor: Value of the flat map returned when no detections
                            fire (so the compressor doesn't over-blur).
            strict: If True (default), raise RuntimeError if YOLO cannot load.
                    Set False only if you actively want the silent flat-map
                    fallback — usually you don't.
            device: Inference device for YOLO. "auto" (default) picks MPS on
                    Apple Silicon, CUDA if available, else CPU. Pass an explicit
                    string ("mps", "cuda", "cpu", "cuda:0") to override.
                    Ultralytics does NOT auto-select MPS — passing "auto" here
                    is the difference between ~5 fps and ~30 fps on M-series.
        """
        self.weights = weights
        self.classes = set(classes) if classes is not None else None
        self.conf_threshold = conf_threshold
        self.sigma_scale = sigma_scale
        self.infer_every = max(1, int(infer_every))
        self.fallback_floor = fallback_floor
        self.strict = strict
        self.device = self._resolve_device(device)

        self._model = None
        self._load_attempted = False
        self._available = False
        self._frame_idx = 0
        self._last_map: Optional[np.ndarray] = None

        # Eager load in strict mode so failures surface at construction time,
        # not silently 1000 times during a 30-minute ablation run.
        if self.strict:
            self._load()

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to a concrete device string."""
        if device != "auto":
            return device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"


    def _load(self) -> bool:
        if self._load_attempted:
            return self._available
        self._load_attempted = True
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
            # Move model weights onto the resolved device once at load time so
            # every subsequent predict() doesn't re-pay the device-transfer cost.
            try:
                self._model.to(self.device)
            except Exception as e:
                logger.warning(
                    f"YoloSaliency: could not move model to device "
                    f"{self.device!r} ({e}); inference will pick its own."
                )
            self._available = True
            logger.info(
                f"YoloSaliency: loaded {self.weights} on device={self.device!r}"
            )
        except ImportError as e:
            msg = (
                f"YoloSaliency: ultralytics is not installed in this Python "
                f"environment. Install it with: pip install ultralytics  "
                f"(underlying error: {e})"
            )
            if self.strict:
                raise RuntimeError(msg) from e
            logger.warning(msg + " — falling back to flat map.")
            self._available = False
        except Exception as e:
            msg = (
                f"YoloSaliency: failed to load weights {self.weights!r} "
                f"(underlying error: {e})"
            )
            if self.strict:
                raise RuntimeError(msg) from e
            logger.warning(msg + " — falling back to flat map.")
            self._available = False
        return self._available

    @staticmethod
    def _gaussian_blob(
        h: int, w: int, cx: float, cy: float, sigma: float, amp: float
    ) -> np.ndarray:
        """Add an isotropic Gaussian centered at (cx, cy) with given sigma+amp.

        Returns the blob alone — caller is expected to sum into an accumulator.
        The Gaussian is separable, so this is the outer product of two 1-D
        profiles: O(H+W) exp() calls instead of O(H*W) plus two full-frame
        coordinate grids. Same values (to float32 rounding), ~70x faster per
        detection at 1080p (26 ms -> 0.4 ms on an M-series laptop), which
        matters on the Pi where this runs for every box on every frame.
        """
        two_s2 = 2.0 * max(sigma, 1.0) ** 2
        gx = np.exp(-((np.arange(w, dtype=np.float32) - cx) ** 2) / two_s2)
        gy = np.exp(-((np.arange(h, dtype=np.float32) - cy) ** 2) / two_s2)
        return (amp * gy)[:, None] * gx[None, :]

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]

        # Reuse last map if we're inside an infer_every window
        if self._last_map is not None and (self._frame_idx % self.infer_every) != 0:
            self._frame_idx += 1
            return self._last_map.copy()

        self._frame_idx += 1

        if not self._load():
            self._last_map = np.full((h, w), self.fallback_floor, dtype=np.float32)
            return self._last_map.copy()

        try:
            results = self._model.predict(
                frame_bgr,
                verbose=False,
                conf=self.conf_threshold,
                device=self.device,
            )
        except Exception as e:
            logger.debug(f"YoloSaliency: inference error {e}; using flat map")
            self._last_map = np.full((h, w), self.fallback_floor, dtype=np.float32)
            return self._last_map.copy()

        sal = np.zeros((h, w), dtype=np.float32)
        n_kept = 0
        for r in results or []:
            for box in getattr(r, "boxes", []) or []:
                cls_id = int(box.cls.item())
                if self.classes is not None and cls_id not in self.classes:
                    continue
                conf = float(box.conf.item())
                # xyxy in original image coords
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten().tolist()
                bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                sigma = self.sigma_scale * float(np.sqrt(bw * bh))
                sal += self._gaussian_blob(h, w, cx, cy, sigma, amp=conf)
                n_kept += 1

        if n_kept == 0:
            self._last_map = np.full((h, w), self.fallback_floor, dtype=np.float32)
            return self._last_map.copy()

        # Normalise to [0, 1]
        sal = sal / max(sal.max(), 1e-8)
        # Clamp the floor so non-salient regions never go to literal 0,
        # which would let the compressor turn them into a flat blur.
        sal = np.maximum(sal, self.fallback_floor)
        self._last_map = sal
        return sal.copy()


def describe_device(device: str = "auto", saliency_backend: str = "yolo+spectral") -> dict:
    """Resolve a device string and describe it in human-readable form.

    Returns a dict with:
      - resolved     : 'mps' | 'cuda' | 'cuda:0' | 'cpu' (concrete device YOLO will use)
      - accelerated  : True iff the resolved device is GPU-class (mps/cuda)
      - yolo_in_use  : True iff the chosen saliency backend will actually call YOLO
      - label        : human-readable string ("MPS (Apple Silicon GPU via Metal)",
                       "CUDA: NVIDIA RTX 4090", "CPU (no GPU available)")
      - banner       : a one-liner ready to print

    The label is honest about the backend choice: if the user picked a CPU-only
    saliency backend (spectral / finegrained), `yolo_in_use=False` and the
    banner says so — no point implying GPU use when there's no neural inference.
    """
    resolved = YoloSaliency._resolve_device(device)
    accelerated = resolved.startswith("mps") or resolved.startswith("cuda")
    yolo_in_use = "yolo" in saliency_backend

    if resolved.startswith("mps"):
        label = "MPS (Apple Silicon GPU via Metal)"
    elif resolved.startswith("cuda"):
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            label = f"CUDA — {name}"
        except Exception:
            label = f"CUDA ({resolved})"
    else:
        label = "CPU (no GPU accelerator available or selected)"

    if not yolo_in_use:
        banner = (f"hardware accel   : N/A — saliency backend {saliency_backend!r} "
                  f"runs on CPU (no neural inference)")
    elif accelerated:
        banner = f"hardware accel   : {label}  [YOLO accelerated]"
    else:
        banner = (f"hardware accel   : {label}  "
                  f"[WARNING — YOLO running on CPU, expect ~5-10× slowdown]")

    return {
        "resolved": resolved,
        "accelerated": accelerated,
        "yolo_in_use": yolo_in_use,
        "label": label,
        "banner": banner,
    }


class SpectralResidualSaliency:
    """NumPy port of OpenCV's StaticSaliencySpectralResidual (Hou & Zhang 2007).

    Same algorithm and constants as the contrib implementation (64x64 grey,
    3x3 box-filtered log amplitude, 5x5 sigma-8 Gaussian on the squared
    reconstruction); matches it to within 1e-3 per pixel. Exists because
    `cv2.saliency` silently disappears when `opencv-python` (an ultralytics
    dependency) is installed after `opencv-contrib-python` — both packages
    own the same `cv2/` directory and the last one installed wins.
    """

    def __init__(self, size: int = 64) -> None:
        self.size = size

    def computeSaliency(self, frame_bgr: np.ndarray) -> tuple[bool, np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        small = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_LINEAR_EXACT)
        spectrum = np.fft.fft2(small.astype(np.float64))
        log_amp = np.log(np.abs(spectrum) + 1e-12)
        residual = log_amp - cv2.blur(log_amp, (3, 3))
        recon = np.fft.ifft2(np.exp(residual + 1j * np.angle(spectrum)))
        sal = cv2.GaussianBlur(np.abs(recon), (5, 5), 8)
        sal = sal * sal
        sal = (sal / max(float(sal.max()), 1e-12)).astype(np.float32)
        h, w = frame_bgr.shape[:2]
        return True, cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)


def spectral_residual_saliency():
    """OpenCV's contrib spectral-residual saliency if present, else the NumPy port."""
    if hasattr(getattr(cv2, "saliency", None), "StaticSaliencySpectralResidual_create"):
        return cv2.saliency.StaticSaliencySpectralResidual_create()
    logger.warning(
        "cv2.saliency is unavailable (opencv-python is shadowing "
        "opencv-contrib-python); using the NumPy spectral-residual port."
    )
    return SpectralResidualSaliency()


class SaliencyEstimator:
    def __init__(
        self,
        backend: str = "spectral",
        yolo_kwargs: Optional[dict] = None,
    ) -> None:
        self.backend = backend
        yolo_kwargs = yolo_kwargs or {}

        if backend == "spectral":
            self._impl = spectral_residual_saliency()
            self._spectral = self._impl
            self._yolo = None
        elif backend == "finegrained":
            try:
                self._impl = cv2.saliency.StaticSaliencyFineGrained_create()
            except AttributeError as e:
                raise RuntimeError(
                    "cv2.saliency not available. Reinstall opencv-contrib-python "
                    "after anything that pulls in opencv-python."
                ) from e
            self._spectral = None
            self._yolo = None
        elif backend == "yolo":
            self._impl = None
            self._spectral = None
            self._yolo = YoloSaliency(**yolo_kwargs)
        elif backend == "yolo+spectral":
            self._impl = None
            self._spectral = spectral_residual_saliency()
            self._yolo = YoloSaliency(**yolo_kwargs)
        elif backend == "model":
            raise NotImplementedError(
                "Model-based saliency backend is not wired up yet. "
                "Use backend='spectral', 'finegrained', or 'yolo' for now."
            )
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    @staticmethod
    def _normalise(sal: np.ndarray) -> np.ndarray:
        mn, mx = float(sal.min()), float(sal.max())
        if mx - mn < 1e-8:
            return np.full_like(sal, 0.5, dtype=np.float32)
        return ((sal - mn) / (mx - mn)).astype(np.float32)

    def _spectral_predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        success, sal = self._spectral.computeSaliency(frame_bgr)
        if not success:
            logger.warning("Spectral saliency failed; flat map returned.")
            return np.full(frame_bgr.shape[:2], 0.5, dtype=np.float32)
        return self._normalise(sal.astype(np.float32))

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return HxW saliency map in [0,1] float32. Matches frame's HxW."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR frame, got shape {frame_bgr.shape}")

        if self.backend in ("spectral", "finegrained"):
            success, sal = self._impl.computeSaliency(frame_bgr)
            if not success:
                logger.warning("Saliency compute failed; returning flat map.")
                return np.full(frame_bgr.shape[:2], 0.5, dtype=np.float32)
            return self._normalise(sal.astype(np.float32))

        if self.backend == "yolo":
            return self._yolo.predict(frame_bgr)

        if self.backend == "yolo+spectral":
            yolo_map = self._yolo.predict(frame_bgr)
            spec_map = self._spectral_predict(frame_bgr)
            # Per-pixel max keeps strong signals from either source. We
            # de-weight spectral so YOLO dominates when it has confidence.
            combined = np.maximum(yolo_map, 0.6 * spec_map)
            return self._normalise(combined)

        raise RuntimeError(f"Unhandled backend {self.backend!r}")

    def predict_batch(self, frames: np.ndarray) -> np.ndarray:
        return np.stack([self.predict(f) for f in frames], axis=0)


class TemporalSmoother:
    """Moving-average smoother for saliency maps, to suppress flicker."""

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._buf: Deque[np.ndarray] = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    def smooth(self, sal: np.ndarray) -> np.ndarray:
        self._buf.append(sal.astype(np.float32))
        return np.mean(np.stack(list(self._buf), axis=0), axis=0)


def saliency_to_qp_map(
    sal: np.ndarray,
    qp_baseline: int = 28,
    qp_delta: int = 6,
    block_size: int = 16,
) -> np.ndarray:
    """Convert saliency map (HxW, [0,1]) to a per-block QP map suitable for ROI encoding.

    Low saliency -> higher QP (coarser); high saliency -> lower QP (crisper).
    Returns an int32 grid of shape (H//block_size, W//block_size).
    """
    h, w = sal.shape
    bh, bw = h // block_size, w // block_size
    sal_small = cv2.resize(sal, (bw, bh), interpolation=cv2.INTER_AREA)
    qp = qp_baseline - qp_delta * sal_small  # high saliency -> low QP
    return np.clip(np.round(qp), 18, 40).astype(np.int32)


if __name__ == "__main__":
    # Smoke test
    logging.basicConfig(level=logging.INFO)
    est = SaliencyEstimator(backend="spectral")
    smoother = TemporalSmoother(window=5)

    for i in range(5):
        frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
        sal = est.predict(frame)
        sal_s = smoother.smooth(sal)
        qp = saliency_to_qp_map(sal_s)
        print(f"frame {i}: sal range=[{sal.min():.3f}, {sal.max():.3f}] "
              f"qp range=[{qp.min()}, {qp.max()}] qp_shape={qp.shape}")
    print("saliency.py smoke test passed.")
