# Architecture & Pipeline Overview

A guide to how the system works end-to-end, written for team members who need
to explain it to others. Every section has a diagram + short notes on what
matters most.

> Render note: GitHub, VS Code, Notion, and Obsidian all render Mermaid
> diagrams natively. Just open this file.

---

## 1. Camera deployment runtime — what runs on the device

This is what happens on the camera once we ship. Frames come in from the
sensor, get gated, get analyzed for saliency, get encoded, and get stored or
uploaded.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#ffffff','primaryTextColor':'#111111','primaryBorderColor':'#71717a','lineColor':'#71717a','tertiaryColor':'#ffffff'}}}%%
flowchart TD
    A([Camera sensor<br/>raw frames]) --> B[Footage gate<br/>motion + flow + person]
    B -->|usefulness < threshold| C[Idle mode<br/>heavy compression / skip]
    B -->|usefulness ≥ threshold| F[BgFgCodec encoder<br/>saliency + bg/fg blend + H.265<br/>see diagram 2 for internals]
    F --> G([H.265 mp4<br/>storage or upload])
    C --> G

    style F fill:#fde2e2,stroke:#dc2626,color:#7f1d1d
```

**Notes for each step:**

- **Footage gate** (`src/gate.py`): cheap CPU check that decides if anything
  interesting is happening. Uses MOG2 background subtraction (motion), sparse
  optical flow (movement magnitude), and optional YOLOv8n person detection.
  Threshold tunable (default 0.25). Saves enormous power on long idle periods.
  The gate runs its own motion/flow signals and does **not** consume the
  codec's saliency map — the two pipelines are independent.
- **BgFgCodec encoder** (`src/bg_fg_codec.py`): our main innovation. Owns the
  saliency estimator (`src/saliency.py`, backend selectable — see diagram 3)
  and the temporal smoother internally; the camera runtime treats it as a
  single black box that takes raw frames in and emits H.265 bytes out. The
  per-frame data path (saliency → smoother → sigmoid mask → bg/fg blend →
  libx265) is broken out in diagram 2.
- **Idle mode**: when nothing interesting is happening, encode with very
  aggressive compression (or skip entirely). Saves storage + bandwidth.

---

## 2. BgFgCodec internals — the two-pass encoder

This is our headline innovation. It's a thin pre-processor in front of an
unmodified H.265 encoder. Two passes per clip.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#ffffff','primaryTextColor':'#111111','primaryBorderColor':'#71717a','lineColor':'#71717a','tertiaryColor':'#ffffff','clusterBkg':'#fafafa','clusterBorder':'#a1a1aa'}}}%%
flowchart TB
    subgraph P1[Pass 1 — Build background reference]
        S1[Sample N=30 frames<br/>evenly across clip] --> S2[Per-pixel temporal median]
        S2 --> BG([Background image<br/>HxWx3 uint8])
    end

    subgraph P2[Pass 2 — Saliency-blended foreground]
        F1[Each frame] --> F2[Saliency map<br/>yolo+spectral]
        F1 --> FM[Motion mask<br/>&#124;frame − bg&#124;]
        BG -.shared.-> FM
        F2 --> FX[Per-pixel max<br/>semantic ∪ motion]
        FM --> FX
        FX --> F3[Smooth in time<br/>~7–11 frame window]
        F3 --> F4[Sigmoid mask<br/>+ floor 0.25]
        BG -.shared.-> F5
        F4 --> F5[Blend:<br/>mask·frame + 1-mask·bg]
        F5 --> F6[Stabilised frame stream]
    end

    P1 --> P2
    P2 --> ENC[libx265 encoder<br/>CRF 28]
    ENC --> OUT([ours_bgfg.mp4])

    style P1 fill:#fef3c7,stroke:#d97706,color:#78350f
    style P2 fill:#e8f4fd,stroke:#3b82f6,color:#0c4a6e
    style ENC fill:#fde2e2,stroke:#dc2626,color:#7f1d1d
```

**Notes for each step:**

- **Sample 30 frames + per-pixel median**: builds a clean "what does the empty
  scene look like" reference. Temporal median is robust to short-lived
  foreground — a person walking through won't pollute the background because
  no pixel stays "non-background" for more than a fraction of the clip.
- **Saliency map**: per-frame map of "what to preserve." Backend is
  selectable — see diagram 3.
- **Motion mask**: per-pixel `|frame − background|`, thresholded and dilated.
  Anything that differs from the static background is by definition
  foreground, so this catches subjects the semantic saliency missed (dark
  figures on dark backgrounds, low-confidence YOLO detections, etc.).
- **Per-pixel max (semantic ∪ motion)**: combines the two via `max`, so a
  pixel is "important" if *either* signal fires. This is the fix for the
  "person dissolved into the bar" failure mode where YOLO alone went blank.
- **Temporal smoothing**: moving-average over ~7–11 recent maps. Prevents
  flicker — without this the mask jitters every frame and the codec wastes
  bits encoding the jitter.
- **Sigmoid-shaped mask + floor 0.25**: converts continuous saliency [0,1]
  into a soft blending weight. Steep transition centered at `threshold=0.3`,
  so salient interior is fully preserved, non-salient interior fully
  replaced, and the boundary is smooth (no visible seams). The 0.25 floor is
  a safety net: even "non-salient" pixels keep a quarter of the original
  frame, so anything saliency missed never *fully* vanishes.
- **Blend**: `stabilised = mask·frame + (1−mask)·background`. The salient
  regions keep their original pixels; non-salient regions are replaced with
  the matching background pixel at that exact location.
- **libx265 encode**: the magic step. Because non-salient pixels are now
  **byte-identical across frames**, H.265's inter-frame prediction reduces
  those regions to ~zero bits via skip flags. All bitrate flows to the
  foreground. The output is a normal H.265 mp4 — no custom decoder needed.

**Key insight:** we're not replacing H.265. We're feeding H.265 a stream that
plays to its strengths. Every byte the codec saves on the static background
is a byte it can spend on the action.

---

## 3. Saliency backend evolution — why three backends exist

The saliency map tells the codec "this is what to preserve." Getting this
right was where we had the biggest perceptual gains. We have three backends
now, each addressing a problem with the previous one.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#ffffff','primaryTextColor':'#111111','primaryBorderColor':'#71717a','lineColor':'#71717a','tertiaryColor':'#ffffff','clusterBkg':'#fafafa','clusterBorder':'#a1a1aa'}}}%%
flowchart LR
    subgraph V1[v1 - spectral residual]
        A1[Frame] --> A2[FFT log-magnitude<br/>spectrum]
        A2 --> A3[Subtract smoothed avg<br/>residual = novel]
        A3 --> A4[Inverse FFT]
        A4 --> A5([Saliency map])
    end

    subgraph V2[v2 - YOLO semantic]
        B1[Frame] --> B2[YOLOv8n detect<br/>person, vehicle, bag, etc.]
        B2 --> B3[For each box:<br/>Gaussian blob @ center]
        B3 --> B4[Sum + normalize]
        B4 --> B5([Saliency map])
    end

    subgraph V3[v3 - hybrid]
        C1[Frame] --> C2[YOLO map]
        C1 --> C3[Spectral map · 0.6]
        C2 --> C4[Per-pixel max]
        C3 --> C4
        C4 --> C5([Saliency map])
    end

    V1 -.problem.-> X1[Fires on edges/clutter,<br/>misses actual subjects]
    V2 -.problem.-> X2[Goes blank when YOLO<br/>misses or is uncertain]
    V3 -.solution.-> X3[Semantic + low-level fallback]

    style V1 fill:#f3f4f6,stroke:#6b7280,color:#1f2937
    style V2 fill:#e8f4fd,stroke:#3b82f6,color:#0c4a6e
    style V3 fill:#dcfce7,stroke:#16a34a,color:#14532d
    style X1 fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    style X2 fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    style X3 fill:#dcfce7,stroke:#16a34a,color:#14532d
```

**Notes for each backend:**

- **v1 — spectral residual** (Hou & Zhang 2007, OpenCV built-in):
  closed-form FFT trick. Subtracts the smoothed log-magnitude spectrum from
  the raw one — what's left is "the parts of this image that deviate from
  the typical 1/f natural-image statistics." Fast (100+ fps), no training.
  *Problem*: it's novelty-driven, not semantic. Fires on high-contrast edges,
  cluttered textures, and timestamps in the corner. Misses uniformly-lit
  humans walking across a clean floor. This is why clip_04 lost the pool
  sticks — the saliency map said the cue sticks weren't unusual enough.

- **v2 — YOLO semantic** (new — `YoloSaliency`): YOLOv8n detections drive
  Gaussian blobs centered at each detection box, with σ proportional to box
  size and amplitude equal to detection confidence. We restrict to a
  surveillance-relevant class subset: person, bicycle, car, motorcycle, bus,
  truck, bird, cat, dog, backpack, handbag, suitcase, bottle, knife, laptop,
  cell phone, book. Picks up what we actually care about.
  *Problem*: when YOLO is uncertain (poor lighting, occlusion, unfamiliar
  pose), the saliency map can collapse to near-empty, which then makes the
  codec replace large regions with background — visible artifacts.

- **v3 — yolo+spectral hybrid** (new — default): per-pixel max of the YOLO
  map and a scaled spectral residual map. Best of both worlds: YOLO drives
  the semantic foreground, spectral residual provides a low-level safety net
  for regions YOLO missed. This is the production default.

---

## 4. The full data path — a single clip's journey

Putting it all together: one mp4 file going through the entire system to
become a smaller mp4 file.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#ffffff','primaryTextColor':'#111111','primaryBorderColor':'#71717a','lineColor':'#71717a','tertiaryColor':'#ffffff'}}}%%
flowchart LR
    IN([clip_04.mp4<br/>10s, 300 frames]) --> A[Read frames]
    A --> B[BgFgCodec.encode]
    B --> P1[Pass 1: median bg]
    B --> P2[Pass 2: per-frame]
    P2 --> S[Saliency<br/>yolo+spectral]
    S --> M[Smoother +<br/>sigmoid mask]
    P1 --> BL[Blend]
    M --> BL
    BL --> ST[Stabilised<br/>frame stream]
    ST --> H[libx265<br/>encode]
    H --> OUT([ours_bgfg.mp4<br/>~64 KB<br/>43% smaller])

    EVAL[Metrics<br/>PSNR / SSIM / sal-PSNR] -.evaluation.-> OUT
    EVAL -.evaluation.-> IN

    style B fill:#fde2e2,stroke:#dc2626,color:#7f1d1d
    style S fill:#e8f4fd,stroke:#3b82f6,color:#0c4a6e
    style OUT fill:#dcfce7,stroke:#16a34a,color:#14532d
```

**Notes on the journey:**

- **Inputs**: any mp4 — surveillance clips, webcam captures, anything from a
  stationary camera. The pipeline doesn't know or care about the source.
- **Two passes**: pass 1 builds the background once per clip (~1s overhead),
  pass 2 processes every frame. Total time on a 10-second 640×360 clip:
  ~3-5 seconds on a laptop CPU.
- **Output**: a standards-compliant H.265 mp4. Plays in any modern video
  player. Decoders have no idea it was pre-processed.
- **Evaluation**: PSNR (pixel fidelity), SSIM (structural similarity),
  sal-PSNR (fidelity only in salient regions). The last one is the honest
  metric for our codec — see writeup section.

---

## 5. Why this beats baseline H.265 — the intuition

Baseline H.265 has no idea what's important. It spends bits on every pixel
of every frame, weighted by the codec's internal heuristics (which prefer
"interesting" regions but don't know what an "interesting" region IS).

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#ffffff','primaryTextColor':'#111111','primaryBorderColor':'#71717a','lineColor':'#71717a','tertiaryColor':'#ffffff','clusterBkg':'#fafafa','clusterBorder':'#a1a1aa'}}}%%
flowchart TB
    subgraph BASE[Baseline H.265]
        BF[Each frame] --> BE[Encode whole frame<br/>uniformly]
        BE --> BB([All pixels get bits<br/>112 KB])
    end

    subgraph OURS[ours_bgfg]
        OF[Each frame] --> OS[Mark salient regions]
        OS --> OR[Replace non-salient<br/>with background]
        OR --> OE[H.265 encodes stream]
        OE --> OB([Non-salient = 0 bits<br/>Salient = full bits<br/>64 KB])
    end

    BASE -.contrast.-> OURS

    style BASE fill:#f3f4f6,stroke:#6b7280,color:#1f2937
    style OURS fill:#dcfce7,stroke:#16a34a,color:#14532d
```

**The two-line pitch:**

> Baseline H.265 wastes bits encoding the parts of the scene that never
> change. We feed H.265 a version of the video where the background pixels
> are byte-identical across frames, so its inter-frame prediction collapses
> those regions to ~zero bits. All saved bits go to the foreground —
> 43% smaller files at comparable perceived quality on the regions that
> matter.

---

## 6. Quick reference — files & responsibilities

| Module | Job |
|---|---|
| `src/saliency.py` | All saliency backends (spectral, finegrained, yolo, yolo+spectral). Temporal smoothing. |
| `src/gate.py` | Footage gate (motion + flow + person). Decides if recording is worth it. |
| `src/compress.py` | `FramePipeEncoder` (raw BGR frames → H.265), `encode_uniform` (baseline transcode), `_build_alpha` (saliency-mask shaping). |
| `src/bg_fg_codec.py` | The bg/fg decomposition codec (`ours_bgfg`). Our headline innovation. |
| `src/metrics.py` | PSNR, SSIM, LPIPS, and **saliency-weighted PSNR** (the honest metric for our codec). |
| `src/neural_codec.py` | Single-frame autoencoder (the *naive learned-codec baseline* — not competitive). |
| `scripts/ablation_bgfg.py` | Head-to-head comparison: baseline H.265 vs ours_bgfg. |
| `scripts/qualitative_catalog_v2.py` | 4-panel comparison PDF for demo example selection. |
| `scripts/photo_demo.py` | Live webcam demo, baseline vs ours_bgfg side by side. |
| `docs/writeup_updates.md` | Drop-in sections for analysis.docx. |

> **Note:** the prior `ours_sigmoid` pre-blur pipeline (formerly `src/pipeline.py`, `scripts/run_pipeline.py`, `SaliencyCompressor`) was retired in May 2026 — the bg/fg codec dominates it on every measured CRF, so we removed it from the runnable surface. The historical comparison tables in `docs/writeup_updates.md` and `docs/results_summary.md` are frozen and no longer reproducible from the codebase as-is.
