# Saliency-Aware Surveillance Codec

A saliency-aware background/foreground codec for stationary surveillance cameras. It pins everything a viewer wouldn't look at to a learned background image, so a standard H.265 encoder spends its bits on people, vehicles, and anything that moves.

---

## The one-line pitch

Surveillance cameras don't move. We exploit that explicitly: replace non-salient pixels with a learned background image, then hand the stream to H.265. The codec's inter-frame prediction collapses the static regions to near-zero bits, and every saved bit moves to the foreground.

The output is a standards-compliant H.265 mp4. Any decoder plays it back without modification.

---

## What's in the repo

| File | What it is |
|---|---|
| [`analysis.docx`](analysis.docx) | The 5-page technical analysis: method, perceptual grounding, results, limitations. |
| [`docs/architecture.md`](docs/architecture.md) | System overview with mermaid diagrams. |
| [`docs/demo_scripts.md`](docs/demo_scripts.md) | Every copy-pasteable demo command, grouped by purpose. |
| [`docs/raspberry_pi_setup.md`](docs/raspberry_pi_setup.md) | Fresh-Pi-5 install and demo-day checklist. |
| [`docs/external_data.md`](docs/external_data.md) | How to source higher-resolution surveillance footage (VIRAT, Mixkit, Pexels). |
| [`docs/writeup_updates.md`](docs/writeup_updates.md) | Drop-in section content used to build the analysis. |


---

## Headline numbers

The clip we ship in the Pi demo: **VIRAT_S_000003**, 75 s of stationary 1080p surveillance footage (~158 MB raw).

| Setting | Baseline H.265 | Ours bgfg | Saved |
|---|---|---|---|
| VIRAT_S_000003 @ CRF 22 | 33.9 MB | **6.5 MB** | **81%** |

Sal-PSNR (saliency-weighted PSNR, error counted only inside the regions we promised to keep): **34.76 dB** for ours vs **37.44 dB** for the baseline — a 2.7 dB drop on the salient pixels, in exchange for an 81% file-size reduction. To the eye, the subject quality is indistinguishable on a side-by-side playback (see `pi_demo/clip_virat_crf22.mp4`).

For broader context, the 20-clip aggregate over the Kaggle CCTV Action-Recognition dataset (13 action classes, 640×360 clips):

| Setting | Baseline H.265 | Ours bgfg | Saved |
|---|---|---|---|
| 20-clip aggregate @ CRF 22 | 256.4 KB | 213.4 KB | **17%** |
| 20-clip aggregate @ CRF 28 | 134.2 KB | 118.4 KB | **12%** |

The per-clip win scales with how much of the frame is static background, so the codec gets *better* as cameras move to higher resolution and wider angles — the direction surveillance is already going. The VIRAT 1080p clip is the clearest demonstration of that scaling. See [`docs/architecture.md`](docs/architecture.md) for the why and [`docs/demo_scripts.md`](docs/demo_scripts.md) for how to reproduce.

These numbers were measured with x265's default 250-frame GOP. The 20 s GOP cap (step 5 below) makes long static clips smaller still: on a 960×540 copy of the same 75 s VIRAT clip it shrinks the bg/fg file by a further 24–26% at the same CRF (PSNR within 0.05 dB), taking the saving over baseline H.265 at CRF 22 from 51% to 58% — and from 77% to 81% with simulated sensor noise added. Clips shorter than ~8 s, like the Kaggle set, are unaffected.

---

## Quick start

```bash
git clone https://github.com/SaiAmartya/saliency-surveillance-codec.git
cd saliency-surveillance-codec
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The three demos worth running first:

```bash
# 1. Live webcam saliency heatmap window
python scripts/live_demo.py

# 2. Webcam → 4-panel compressed comparison
python scripts/photo_demo.py --duration 5 --saliency yolo+spectral --show

# 3. Apples-to-apples on any mp4 (VIRAT clip, your own footage, etc.)
python scripts/bench_external.py --input path/to/clip.mp4 --saliency yolo+spectral --show
```

For the full reproducibility flow (encoding all 20 clips, regenerating the catalog and RD curve), see [`docs/demo_scripts.md`](docs/demo_scripts.md).

**Running on a Raspberry Pi 5 instead of a laptop?** See [`docs/raspberry_pi_setup.md`](docs/raspberry_pi_setup.md).

---

## Code map

```
saliency-surveillance-codec/
├── analysis.docx                   5-page technical analysis
├── README.md                       You're here
├── PLAN.md                         Original design plan
├── LICENSE                         MIT
├── requirements.txt
│
├── docs/
│   ├── architecture.md             System diagrams + how it works
│   ├── demo_scripts.md             Copy-pasteable demo commands
│   ├── raspberry_pi_setup.md       Pi 5 deployment guide
│   ├── external_data.md            How to source 1080p+ test footage
│   ├── writeup_updates.md          Drop-in analysis sections
│   ├── 00-overview.md              Orientation for new collaborators
│   └── research/                   Four research briefs feeding the analysis
│
├── src/
│   ├── saliency.py                 YOLO+spectral+motion saliency backends
│   ├── gate.py                     Useful-footage gate (motion + flow + YOLO)
│   ├── bg_fg_codec.py              ← The headline innovation: bg/fg decomposition codec
│   ├── compress.py                 FramePipeEncoder (raw frames → H.265), encode_uniform (baseline), _build_alpha (mask shaping)
│   ├── neural_codec.py             Tiny autoencoder (the naïve learned-codec baseline)
│   └── metrics.py                  PSNR / SSIM / LPIPS / saliency-weighted PSNR
│
├── scripts/
│   ├── live_demo.py                Real-time webcam saliency heatmap window
│   ├── photo_demo.py               Webcam capture → 4-panel comparison PNG
│   ├── compare_clip.py             Build a side-by-side video for any clip
│   ├── saliency_video.py           Render a saliency-overlay video for any clip
│   ├── bench_external.py           Run the codec on any mp4 (VIRAT, Pexels, etc.)
│   ├── ablation_bgfg.py            Head-to-head comparison across all 20 clips × 3 CRFs
│   ├── qualitative_catalog_v2.py   Build the multi-page comparison PDF
│   ├── simulate_hour.py            Synth long-form footage for the hour-long argument
│   ├── train_autoencoder.py        Trains the neural-codec baseline
│   ├── make_test_video.py          Synthetic test clip generator
│   └── (older ablation scripts: ablation_real.py, ablation_mask_modes.py, etc.)
│
└── pi_demo/                        Standalone Raspberry Pi playback bundle
    ├── demo_pi.py                  Replays the pipeline log in ~25s, opens the cached comparison
    ├── clip_virat_crf22.mp4        Pre-computed 3-panel side-by-side (~168 MB)
    └── README.md                   Setup + run instructions for the Pi
```

Every module exposes a small public API documented in its docstring. `bg_fg_codec.py` is the integration surface for the headline codec. The previous `ours_sigmoid` pre-blur pipeline (`src/pipeline.py`, `scripts/run_pipeline.py`) was retired in May 2026 — bg/fg dominated it on every measured CRF, so the runnable code was removed. The historical comparison tables in `docs/writeup_updates.md` and `docs/results_summary.md` are frozen and no longer reproducible.

---

## How the codec works (30-second version)

1. **Sample 30 evenly-spaced frames** from the clip. Take the per-pixel temporal median — that's the background image. Short-lived foreground (people walking past) is rejected by the median.
2. **For each frame**, compute a saliency map from three combined signals: YOLOv8n detections (semantic — people, vehicles, bags, animals), spectral residual (low-level novelty), and motion against the background image. Temporal-smooth across an 11-frame window.
3. **Blend each frame against the background** using a sigmoid-shaped saliency mask, with a 0.25 floor so the original frame is never *fully* replaced. The blended stream goes to an unmodified `libx265` at the target CRF.
4. **The output is a normal H.265 mp4.** Non-salient regions are byte-identical across consecutive frames, so the codec's inter-frame prediction emits skip flags rather than residuals. All the bits move to the foreground.
5. **Long GOP for a static scene.** Once the background is pinned, intra frames dominate the stream — each one re-sends the entire background. The max I-frame interval is raised from x265's default 250 frames (~8 s) to 20 s (`keyint_s` / `--keyint`); x265 still cuts an I-frame on a scene change, so this is the same "dynamic GOP" idea commercial surveillance encoders (Axis Zipstream, Hikvision H.265+) use. The baseline gets the identical setting, so comparisons stay apples-to-apples.

See [`docs/architecture.md`](docs/architecture.md) for the diagrams.

---

## Honest limitations

The codec assumes a **stationary camera**. Pan/tilt/zoom footage breaks the temporal-median background model, and handheld VIRAT clips lose roughly half the size advantage. When a subject **fills most of the frame** (close-ups, dense two-person scenes), there's barely any background to replace and savings shrink toward zero. When a subject **stands still for >50% of the clip duration**, the median absorbs them into the background — the codec then "sees" them as part of the scene.

These are scope restrictions, not bugs. Most home surveillance deployments are stationary cameras watching mostly-empty spaces, which is exactly where the codec wins. The analysis calls out all three failure modes explicitly.

We also report **saliency-weighted PSNR** (sal-PSNR), which only counts error inside regions our saliency map said matter. Plain PSNR penalises us for intentionally replacing background pixels — which is the *contract* of the codec, not a bug. See §7 of `analysis.docx` for the metric defense.

---

## Conventions

- **Python 3.10+**, type hints, docstrings, no prints in library code (use `logging`).
- **No data committed.** `data/` and most of `results/` are gitignored — see `.gitignore` for exactly what's kept.
- **No model weights committed.** YOLOv8n auto-downloads on first call (~6 MB).
- **Imports.** Modules under `src/` use relative imports (`from .gate import ...`). Scripts under `scripts/` add the repo root to `sys.path` and use absolute imports.
- **Smart quotes are ASCII** in source code, fancy in writeups.

---

## License

[MIT](LICENSE) — feel free to learn from this, fork it, or build on it.

---

## Inspiration

Sebastian Lague, *AI Compression is 300× Better (but we don't use it)* — https://www.youtube.com/watch?v=i6l3535vRjA. The video's framing — neural codecs are dramatically better in theory but impractical at the edge — is the hinge of the project, and the reason our trained autoencoder is included as the *naïve* learned-codec baseline rather than the actual contribution.
