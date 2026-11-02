# Original Design Plan
## A Perceptually-Guided, Sensor-Gated Compression Pipeline for Home Security Cameras

**Author:** Sai Amartya

---

## 1. Executive summary

More than 95% of home-security footage is never watched. It still costs storage, energy, bandwidth, and data-center capacity. The problem breaks into three connected questions: how to compress video using principles of human perception, how to automatically identify which footage is actually "useful," and what algorithms make such a system scalable, privacy-conscious, and precise on edge hardware.

This project answers all three with one coherent system: a real-time video pipeline that (a) uses a lightweight multi-sensor gate — motion, optical flow, on-device object detection, and audio-event detection — to decide when footage is "useful," and (b) applies *saliency-aware perceptual compression* to every frame so that bits go where a human would actually look. The demo runs entirely in software on a laptop, uses only open-source and pretrained components, and ships with a measurable bitrate saving versus a standard H.265 baseline on real surveillance footage.

A one-page "why not a pure neural codec?" analysis — grounded in Sebastian Lague's "AI Compression is 300× Better" video and the underlying literature — explains why the hybrid approach is the realistic answer for edge deployment today, without having to ship an unrealistic system.

---

## 2. How the design maps to the problem

Each of the three questions has a dedicated subsystem:

| Question | Our answer |
|---|---|
| Compress video using principles of human perceptual science | Saliency-aware region-adaptive quantization, layered on H.265 |
| Automatically identify "useful" footage using sensor data | Multi-modal fusion gate: motion + optical flow + object detection + audio events |
| Algorithms & edge-processing techniques that are scalable, privacy-conscious, precise | Classical codec + tiny ML models, all on-device, no raw video leaves the camera |

The architecture is designed so each block in the pipeline maps directly to one of them.

---

## 3. The core technical idea

### 3.1 One-sentence pitch
A home-security camera that only records at full fidelity when its sensors agree something interesting is happening, and even then compresses each frame by the rules of human vision rather than uniformly — cutting storage by roughly an order of magnitude over naive recording, at near-zero extra compute cost.

### 3.2 System architecture

```
        ┌──────────────┐
        │ Camera input │  (webcam or public surveillance clips)
        └──────┬───────┘
               │
               ▼
     ┌──────────────────────┐
     │  A. Useful-footage   │  <- MOG2 / optical flow (always on, cheap)
     │     gating layer     │  <- YOLOv8n person/vehicle detector (triggered)
     │                      │  <- YAMNet audio event detector (triggered)
     │                      │  -> usefulness score [0..1]
     └──────────┬───────────┘
                │
       ┌────────┴────────┐
       │                 │
  score ≥ τ            score < τ
       │                 │
       ▼                 ▼
 ┌──────────────┐  ┌────────────────┐
 │ B. Full-     │  │ D. Timelapse / │
 │    fidelity  │  │    sparse KF   │
 │    record    │  │                │
 └──────┬───────┘  └────────┬───────┘
        │                   │
        ▼                   ▼
 ┌─────────────────────────────────┐
 │ C. Saliency-aware perceptual    │
 │    compression core             │
 │   (runs on EVERY stored frame)  │
 │                                 │
 │   tiny saliency CNN ──▶ QP map  │
 │   H.265 ROI encode (libx265)    │
 └─────────────┬───────────────────┘
               │
               ▼
         compressed .mp4
         + event log (JSON)
```

### 3.3 What makes each block scientific, not ad-hoc

- **The gating layer** is a sensor-fusion classifier. Every signal is cheap enough to run continuously on a laptop CPU. A weighted sum + threshold is the baseline; a tiny logistic regression over fusion features is the stretch goal. This is the "useful footage" answer.
- **The compression core** is where the human-visual-system science lives. Saliency is a proxy for gaze. Gaze is where distortion hurts. Bits go there; bits elsewhere are wasted. This is grounded in Itti & Koch 1998 (saliency), Campbell & Robson 1968 (CSF), Mannos & Sakrison 1974 (JND), and the HEVC perceptual-coding literature — real citations, not hand-wavy.
- **The "what didn't we do"** section — pure neural codecs — gets its own page. We summarise the state of the art (DVC, ELIC, VCT), the reasons they don't deploy to edge (compute, memory, no silicon acceleration), and explain that our hybrid *takes the perceptual-weighting philosophy of neural codecs and realises it on classical rails*. That framing directly addresses the inspiration video.

---

## 4. Scientific foundations (the HVS story)

The five principles below are each citeable from primary sources:

1. **Foveation.** The human fovea has ~10× the spatial resolution of peripheral retina. Visual acuity falls off predictably with eccentricity. *Wandell 1995 (Foundations of Vision); Geisler & Perry 1998 (foveated multiresolution pyramids).*

2. **Contrast Sensitivity Function (CSF).** The eye is most sensitive to mid spatial frequencies (~4 cpd), less to very low or very high frequencies. JPEG's quantization matrix was literally derived from the CSF. *Campbell & Robson 1968; Watson & Ahumada 2005.*

3. **Just-Noticeable-Difference (JND).** There is a distortion threshold below which error is invisible. Modern codecs tune quantization to stay under it. *Mannos & Sakrison 1974; Yang et al. 2005.*

4. **Spatial & temporal masking.** Distortion hides in high-activity regions (textured areas, edges, motion). HEVC and AV1 use local-activity metrics to modulate QP accordingly. *Winkler 2005; Sullivan et al. 2012 (HEVC overview).*

5. **Saliency & attention.** Attention maps correlate strongly with gaze data (DHF1K, SALICON datasets). Distortion in attended regions is more noticeable. *Itti, Koch & Niebur 1998; Borji & Itti 2013.*

Our compression strategy uses (5) as the spatial *mask* and (3)(4) to decide how aggressively to quantize outside the mask. The evaluation uses HVS-grounded metrics — MS-SSIM, PSNR-HVS-M — rather than raw PSNR.

---

## 5. Subsystem details

### 5.1 Useful-footage gating layer

A four-signal fusion pipeline. All components are open-source and run on laptop CPU:

| Signal | Library / model | Approx. CPU cost | What it's good at |
|---|---|---|---|
| Frame differencing / MOG2 background subtraction | OpenCV `cv2.createBackgroundSubtractorMOG2` | <10 ms / frame at 480p | Any pixel-level change; runs always-on |
| Sparse optical flow (Lucas-Kanade) | OpenCV | ~5–10 ms / frame | Distinguishes real motion from lighting flicker |
| Tiny object detector (person / vehicle / package) | YOLOv8n (Ultralytics), INT8 | ~40–60 ms / frame on laptop CPU | Semantic "is this a person" gate; triggered only by motion |
| Audio event detection | YAMNet (TensorFlow, pretrained on AudioSet) | ~50 ms / 1 s audio window | Glass break, dog bark, doorbell, shouting |

**Fusion rule (baseline):** a weighted sum of normalised confidences with a tunable threshold `τ`. Written as:

```
usefulness = 0.25·motion + 0.25·flow + 0.30·person + 0.20·audio_event
record_full_fidelity = usefulness > τ
```

**Stretch goal:** train a 5-feature logistic regression on a small hand-labeled dataset of ~30 min of home-security clips to learn the weights and threshold rather than hand-tuning them. This is what makes the gate scalable and precise — we're not just using a constant, we're learning the fusion policy.

**Privacy:** all inference happens on-device. The event log contains only semantic labels and timestamps (`{"t": 1716000000, "event": "person", "conf": 0.82}`), never raw frames. This is the industry pattern already used by Wyze, Eufy, Amcrest — we just make it explicit in the analysis.

### 5.2 Saliency-aware perceptual compression core

**Step 1 — saliency inference.** Use a lightweight pretrained saliency model. TASED-Net is the obvious first pick (small, PyTorch, good documentation), with a BASNet fallback if inference is too slow. Both output a 0–255 saliency heatmap per frame. Smooth temporally with a 5-frame moving average to avoid flicker between consecutive frames' QP maps.

**Step 2 — saliency → QP map.** Normalize the saliency map, resize it to match the codec's coding-tree-unit (CTU) grid (16×16 for H.264, 64×64 for H.265), then map:
```
QP_block = QP_baseline − Δ · saliency_block_normalised
```
With `QP_baseline = 28` and `Δ = 6`, high-saliency regions get QP=22 (crisp), low-saliency get QP=34 (aggressive). The range is tunable and will be ablated in the evaluation.

**Step 3 — encode with per-region QP.** Three possible routes, in descending order of fidelity-to-the-idea / ascending order of ease:

- **(a) Real ROI encoding in x265.** libx265 accepts per-block QP offsets via its C API. Harder to wire up, but faithful to the pitch. A Python wrapper like `python-libx265` can bridge it; expect some yak-shaving.
- **(b) Per-frame average QP in x264.** ffmpeg has `--qp-file` for per-frame QP. We modulate average QP based on frame-level saliency mean. Loses spatial granularity but still shows the principle end-to-end. *Recommended day-1 baseline.*
- **(c) Software fallback: saliency-weighted spatial blur + standard encode.** If neither codec integration works in time, pre-blur non-salient regions before a uniform-QP encode. This is strictly worse than true ROI but is bulletproof in under 30 lines of OpenCV.

Plan: build (c) first (end-to-end in a day), then move to (b), then push to (a) only if time allows. The analysis document discusses (a) as the principled approach regardless.

### 5.3 The neural-codec comparison (Option 1 graft)

A single page in the analysis, structured as:

- **What neural codecs achieve.** DVC (Lu et al. 2019), ELIC (He et al. 2022), VCT (2023). Typical gains: ~5–15% BD-rate improvement over HEVC in controlled benchmarks; HiFiC-style GANs achieve near-imperceptible quality at very low bitrates on single images.
- **Why they don't ship to the edge.** Four numeric reasons:
  - Inference latency: 100–2000 ms/frame on a laptop CPU vs ~33 ms for real-time H.265.
  - Model size: 50–300 MB of weights, vs tens of KB for codec tables.
  - Memory bandwidth: neural codecs need several GB/s; edge SBCs have a fraction of that.
  - Zero hardware acceleration: every phone and camera SoC has a dedicated H.264/H.265 block; none have a neural-codec block.
- **Our bridge.** We borrow the *idea* behind neural codecs — bits should be allocated to what humans actually perceive — and implement it on classical rails using a tiny saliency CNN guiding a production codec. We pay milliseconds, not seconds.

This section is what turns a "nice CS project" into one that engages with the research frontier. It's also the section most directly tied to the inspiration video.

---

## 6. Privacy-conscious design — the story we tell

Privacy is a first-class requirement for home cameras. Our answers:

1. **All inference is on-device.** Nothing leaves the camera except optionally the event log.
2. **Selective upload.** Only clips flagged as "useful" with high confidence are ever transmitted; silent hours stay local.
3. **Optional face blurring on decode.** Adds a post-processing step during playback using OpenCV face detection + Gaussian blur — demonstrable in the live demo.
4. **Event log is semantic, not photographic.** No raw pixel data is persisted in the log; only labels and timestamps.
5. **Federated-ready architecture.** Because the detector models are small and local, the system is compatible with federated fine-tuning (each home improves the model without sharing video). We note this as future work rather than implementing it.

Sustainability is tied in here too: less upload = less data-center energy = direct carbon reduction.

---

## 7. Technology stack

Everything is Python, all dependencies are pip-installable, all models are pretrained and openly licensed.

| Layer | Stack |
|---|---|
| Video I/O | `opencv-python`, `ffmpeg-python` (wrapper over system ffmpeg with libx264 / libx265) |
| Saliency | PyTorch + TASED-Net (pretrained weights), fallback: MobileNet-based saliency decoder |
| Object detection | Ultralytics `yolov8n.pt` (3.2 MB) |
| Audio events | TensorFlow Lite + YAMNet (`saved_model.pb`, ~3.7 MB) |
| Motion / flow | OpenCV MOG2, Lucas-Kanade |
| Metrics | `scikit-image` (PSNR, SSIM, MS-SSIM), `lpips` pip package, `libvmaf` CLI |
| Glue | Numpy, SciPy, small FastAPI/CLI wrapper so the demo is one command |
| Visualization | Matplotlib for metric plots; OpenCV overlays for live saliency heatmap during the demo |

---

## 8. Evaluation plan (how we prove it works)

Evaluation drives both the analysis writing and the demo visuals, so it's worth doing properly.

**Datasets.** Two sources:
- VIRAT or Avenue public surveillance datasets (freely available research datasets with realistic home / street footage).
- A handful of self-recorded clips with your webcam, labelled with ground-truth "useful / not useful" timestamps.

**Metrics:**
- **Compression metrics:** bitrate (Mbps), file size ratio vs H.265-at-QP-28 baseline.
- **Perceptual quality:** MS-SSIM, LPIPS, VMAF for a subset of clips.
- **Gate performance:** precision / recall / F1 of "useful footage" detection against your hand-labels.
- **System metrics:** end-to-end latency per frame, peak RAM, CPU utilization — proves the edge-feasibility claim.

**Comparisons:**
1. Baseline: uniform H.265 at QP=28.
2. Gate-only: record nothing outside useful windows, full-quality inside.
3. Saliency-only: saliency-aware QP on every frame, no gating.
4. **Ours: gate + saliency combined.**

Expected outcome: (4) beats (1) by 5–10× in storage and beats (2)(3) on the quality–storage Pareto frontier. Even if numbers come in modest, the comparative story still holds.

---

## 9. Risks and fallbacks

| Risk | Likelihood | Mitigation |
|---|---|---|
| TASED-Net inference too slow on your laptop | Medium | Fall back to a smaller model (MobileNet-based saliency), or drop to 5 fps saliency and interpolate between frames. |
| libx265 ROI API too fiddly to wire in time | High | First fallback is per-frame average QP in x264; second is software-side blur before uniform encode. Both answer the principle. |
| YAMNet TF dependency breaks the environment | Medium | Remove audio gate entirely; the gate still works on motion + person detection. Audio becomes "future work." |
| Metrics come in flatter than hoped (e.g., only 20% file-size saving) | Low–medium | Frame the win honestly. 20% is still huge at data-center scale. The reasoning matters more than a 10× number. |
| Live demo fails (webcam, audio, power) | Medium | Pre-recorded screencast on the laptop, ready to play in the same UI. Practice switching to it in under 10 seconds. |

---

## 10. Key references (partial — will be finalized in the analysis)

*Human visual system & perceptual compression:*
- Campbell, F. W., & Robson, J. G. (1968). Application of Fourier analysis to the visibility of gratings. *J. Physiol.*, 197(3).
- Mannos, J. L., & Sakrison, D. J. (1974). The effects of a visual fidelity criterion on the encoding of images. *IEEE TIT*, 20(4).
- Itti, L., Koch, C., & Niebur, E. (1998). A model of saliency-based visual attention for rapid scene analysis. *IEEE TPAMI*, 20(11).
- Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003). Multiscale structural similarity for image quality assessment. *Asilomar*.
- Yang, X. et al. (2005). Just-noticeable-distortion model and its applications in video coding. *Signal Processing: Image Communication*, 20(7).
- Sullivan, G. J. et al. (2012). Overview of the High Efficiency Video Coding (HEVC) standard. *IEEE TCSVT*, 22(12).
- Wandell, B. A. (1995). *Foundations of Vision*. Sinauer.

*Neural video compression:*
- Lu, G. et al. (2019). DVC: An end-to-end deep video compression framework. *CVPR*.
- Ballé, J. et al. (2018). Variational image compression with a scale hyperprior. *ICLR*.
- He, D. et al. (2022). ELIC: Efficient learned image compression. *CVPR*.
- CompressAI (InterDigital): https://github.com/InterDigitalInc/CompressAI

*Evaluation metrics:*
- Li, Z. et al. (2016). Toward a better quality metric for the video community (VMAF). *SMPTE Motion Imaging Journal*.
- Zhang, R. et al. (2018). The unreasonable effectiveness of deep features as a perceptual metric (LPIPS). *CVPR*.
- Chandler, D. M., & Hemami, S. S. (2007). VSNR / PSNR-HVS variants. *IEEE TIP*, 16(10).

*Detection models we'll actually use:*
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- YAMNet: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- TASED-Net (check current canonical repo before citing exact URL).

*Inspiration:*
- Lague, S. (2025). *AI Compression is 300x Better (but we don't use it).* YouTube. https://www.youtube.com/watch?v=i6l3535vRjA
