# Research brief: saliency-based / ROI video compression

> **Note:** This is a research brief gathered during planning. URLs and specific numbers should be verified against primary sources before citing in the analysis document.

## State of the art (2022–2025)

The field has converged on a few patterns:

- **HEVC/H.265 ROI extensions** remain the industry standard. x265 supports per-region quantization parameter (QP) maps via its API.
- **Perceptual quality-adaptive coding** combines just-noticeable-difference (JND) theory with attention maps. Low-saliency regions get QP+3 to +5 (aggressive compression); high-saliency regions get QP−2 to QP (crisper). Reported gains: 25–40% bitrate savings vs uniform encoding without visible quality loss.
- **Eye-tracking datasets** are the ground truth for saliency:
  - **DHF1K** — dynamic human fixation, ~1000 videos, 600p, gaze annotations.
  - **SALICON** — Saliency in Context, ~20K images, mouse-click derived.
  - **Hollywood / UCF-Sports** — action video with eye-tracking.

Lightweight saliency models dominate recent papers because real-time inference is the constraint that matters in deployment.

## Pretrained saliency / attention models for real-time inference

| Model | Approx size | CPU speed | Notes |
|---|---|---|---|
| **TASED-Net** (temporal attention saliency) | ~15 MB | 30–50 fps | Pretrained, PyTorch, no temporal dependency required for static scenes. Common pick. |
| **BASNet** (boundary-aware) | ~100 MB | 15–25 fps | Cleaner object boundaries; good for "is this a person" type masks. |
| **ViNet / ViT-based saliency** | varies | 20+ fps | Can leverage distilled ViT encoders. Fewer public implementations. |
| **SALICON-pretrained ResNet18 + decoder** | ~50 MB | 50+ fps | Easy to retrain in 2–3 hours on a GPU; DIY-friendly. |
| **OpenCV `cv2.saliency.StaticSaliencySpectralResidual`** | n/a | 100+ fps | Classical (Hou & Zhang 2007). What we ship in `src/saliency.py` for Day 1. |

For a 7-day build the pragmatic order is: spectral-residual baseline → TASED-Net upgrade if time permits.

## Pipeline integration

```
video frame → saliency model → smooth temporally → resize to CTU grid → QP map → encoder
```

Encoder options, in descending fidelity-to-the-idea / ascending ease:

1. **Real ROI in libx265** — per-block QP via the C API; gives the principled answer; some yak-shaving with bindings.
2. **Per-frame average QP via x264 `--qp-file`** — modulate frame QP by mean saliency. Loses spatial granularity but works end-to-end.
3. **Software pre-blur + uniform CRF encode** — blur non-salient regions before a vanilla ffmpeg encode. Strictly worse than real ROI but bulletproof in 30 lines. **This is what the Day-1 pipeline ships.**

Useful Python libraries: `ffmpeg-python`, `opencv-python` (incl. `opencv-contrib-python` for `cv2.saliency`), `torch` + saliency model, `numpy`/`scipy`, optionally `av` (PyAV) for direct codec control.

## Evaluation

| Metric | Python package | Effort | Notes |
|---|---|---|---|
| **PSNR** | `skimage.metrics.peak_signal_noise_ratio` | trivial | Perceptually weak baseline. |
| **MS-SSIM** | `skimage.metrics.structural_similarity` (multiscale) | low | HVS-grounded; better than PSNR. |
| **LPIPS** | `lpips` pip package | low | Deep perceptual metric; needs torch; AlexNet variant is fine. |
| **VMAF** | `libvmaf` CLI + wrapper | medium | Netflix standard; very accurate; a few clips is enough for the report. |
| **PSNR-HVS-M** | research code, not pip | medium | Add only if time allows. |

For the demo: PSNR + MS-SSIM (cheap) + LPIPS (if the laptop has GPU); run VMAF on 2–3 representative clips for the analysis document.
