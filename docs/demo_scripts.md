# Demo scripts — copy-paste reference

Every runnable demo, grouped by what it shows. All commands assume you've
`cd`'d into the project root and activated `.venv`:

```bash
cd path/to/saliency-surveillance-codec
source .venv/bin/activate
```

If you ever see `ModuleNotFoundError: ultralytics`, fix it with:
```bash
pip install ultralytics
```

---

## 0. Elevator-pitch demo — one command, biggest visual impact

If you have 30 seconds to show someone the result, run this:

```bash
python scripts/compare_clip.py --clip clip_14 --crf 22 --show
```

Builds a 4-panel mp4 of the bar-interior fall clip (original | saliency |
baseline 194 KB | ours_bgfg ~142 KB) and pops it open in QuickTime. They see
all four versions play in lockstep with size labels burned into each panel.
Subjects (including the falling person) are preserved throughout — verified
after the motion-aware saliency + mask-floor fix.

Output: `results/comparisons/clip_14_crf22.mp4`. Takes ~10 seconds.

---

## 1. Standalone module demos

Each shows one piece of the system in isolation. Great for explaining
*how* the system works, step by step.

### 1a. Live webcam saliency heatmap (existing — real-time)

```bash
python scripts/live_demo.py
```

Opens a window with your webcam feed overlaid with the YOLO+spectral
saliency heatmap and a live "usefulness" score. Press `q` to quit.
Use for "look, the saliency knows where you are in the frame."

### 1b. Saliency heatmap video for a specific clip

```bash
# Single-panel overlay video
python scripts/saliency_video.py --clip clip_14 --show

# Side-by-side: original | overlay
python scripts/saliency_video.py --clip clip_02 --side-by-side --show

# Compare backends on the same clip
python scripts/saliency_video.py --clip clip_04 --saliency spectral --out /tmp/sal_spectral.mp4
python scripts/saliency_video.py --clip clip_04 --saliency yolo --out /tmp/sal_yolo.mp4
python scripts/saliency_video.py --clip clip_04 --saliency yolo+spectral --out /tmp/sal_hybrid.mp4
```

Output: `results/saliency_videos/<clip>_saliency_<backend>.mp4`.

Use this to argue "spectral residual misses the subject, YOLO catches it."

### 1c. Background reference image for a clip

```bash
# Build the temporal-median background and save as PNG
python -c "
import sys; sys.path.insert(0,'.')
from src.bg_fg_codec import compute_background_median
import cv2
bg = compute_background_median('data/real/clip_14.mp4', n_samples=30)
cv2.imwrite('/tmp/bg_clip14.png', bg)
print('wrote /tmp/bg_clip14.png')
" && open /tmp/bg_clip14.png
```

Shows the "empty room" image we extract from a clip. Useful for
"this is what we're replacing the non-salient pixels with."

### 1d. Gate-trace plot (motion + flow + person scores over time)

```bash
# Already exists from prior work
ls results/figures/gate_trace*.png
open results/figures/gate_trace.png
open results/figures/gate_trace_real.png
```

Shows the footage gate scoring each frame's "usefulness" and triggering
recording when threshold is crossed.

---

## 2. Full-clip head-to-head comparison demos

Encode and compare on a specific clip. The output is a SINGLE playable
video with all panels stacked — press play once and see everything.

### 2a. The bar-interior fall (recommended demo)

```bash
# 4-panel: original | saliency | baseline | ours_bgfg at CRF 22 (high quality)
python scripts/compare_clip.py --clip clip_14 --crf 22 --show

# Same clip at CRF 28 (more aggressive compression)
python scripts/compare_clip.py --clip clip_14 --crf 28 --show
```

### 2b. The grab scene (security cam, strongest "captured the crime" narrative)

```bash
python scripts/compare_clip.py --clip clip_02 --crf 22 --show
```

### 2c. The pool hall hit (the original failure case — now fixed)

```bash
# CRF 22 — the redemption
python scripts/compare_clip.py --clip clip_04 --crf 22 --show

# CRF 34 — show the failure mode, then explain why we recommend CRF 22 for stationary scenes
python scripts/compare_clip.py --clip clip_04 --crf 34 --show
```

### 2d. Try any clip

```bash
# Replace clip_XX with anything from clip_01 .. clip_20
python scripts/compare_clip.py --clip clip_06 --crf 22 --show

# Drop the saliency panel for a cleaner 3-panel demo
python scripts/compare_clip.py --clip clip_13 --crf 22 --no-saliency --show
```

---

## 3. Live webcam demos (5-second capture, real-time)

### 3a. The 4-panel comparison from your laptop camera

```bash
# Default: 2-second capture, CRF 28
python scripts/photo_demo.py --saliency yolo+spectral --show

# 5-second capture for a longer demo with movement
python scripts/photo_demo.py --duration 5 --saliency yolo+spectral --show

# Higher quality
python scripts/photo_demo.py --crf 22 --duration 5 --saliency yolo+spectral --show
```

Output: `results/photo_demo/comparison.png` and the captured/encoded mp4s.

**Tip for live demo:** stand up and walk across the frame during the
capture. Static-room captures don't show off bg/fg because there's no
foreground to amortize against. Movement = visible savings.

### 3b. Live saliency heatmap (continuous, no capture)

```bash
python scripts/live_demo.py
```

Same as 1a. Show this *first* to demo the saliency, then switch to 3a for
the full encoded comparison.

---

## 4. Open pre-computed artifacts

Everything below was generated by past ablation runs. No re-encoding needed.

### 4a. The 180-page qualitative catalog

```bash
open results/figures/qualitative_catalog_v2.pdf
```

4-panel comparison for every clip × frame × CRF. Recommended demo pages:
- **page 14**: clip_02 grab, CRF 22 (security cam, strong narrative)
- **page 32**: clip_04 hit, CRF 22 (vindication of pool-hall failure)
- **page 122**: clip_14 fall, CRF 22 (best size win, cleanest visual)

### 4b. The RD curve

```bash
open results/figures/qualitative_catalog_v2.pdf
open results/ablation_bgfg/rd_curve.png
```

Sal-PSNR vs bytes, averaged across all 20 clips. Shows bgfg dominating
the H.265 baseline on the Pareto frontier.

### 4c. The numerical summary

```bash
cat results/ablation_bgfg/summary.md
open results/ablation_bgfg/summary.json   # raw numbers
```

Aggregate table by config × CRF. Drop straight into slides.

---

## 5. Reproduce / re-run the research

Use these only if you've changed parameters and want fresh numbers.

### 5a. Full ablation across all 20 clips (~25 min)

```bash
python scripts/ablation_bgfg.py --saliency yolo+spectral
```

### 5b. Smoke test (3 clips × 1 CRF, ~30 sec)

```bash
python scripts/ablation_bgfg.py --quick --saliency yolo+spectral
```

### 5c. Single clip across all CRFs (~30 sec)

```bash
python scripts/ablation_bgfg.py --clips clip_14 --saliency yolo+spectral
```

### 5d. Rebuild the catalog PDF (after re-running 5a)

```bash
python scripts/qualitative_catalog_v2.py --saliency yolo+spectral
```

---

## 6. Common questions + which command to run

| Question | Run this |
|---|---|
| "How does the saliency work?" | `python scripts/live_demo.py` (live heatmap on you) |
| "Show me the actual codec output" | `python scripts/compare_clip.py --clip clip_14 --crf 22 --show` |
| "What does the 'background' actually look like?" | `python -c "..."` from §1c |
| "Does this work in real time?" | `python scripts/photo_demo.py --codec both --duration 5 --show` |
| "What's your aggregate result?" | `cat results/ablation_bgfg/summary.md` |
| "Where does it fail?" | `python scripts/compare_clip.py --clip clip_03 --crf 22 --show` (handheld camera, bg model breaks) |
| "How is this different from your old approach?" | See `docs/writeup_updates.md` for the frozen head-to-head against the legacy `ours_sigmoid` pre-blur pipeline (retired May 2026; bg/fg dominated it on every CRF). |
| "Is YOLO actually doing something?" | `python scripts/saliency_video.py --clip clip_14 --saliency spectral --out /tmp/s.mp4` then same with `--saliency yolo+spectral` — compare side by side |

---

## 7. File / directory cheat sheet

| Where | What |
|---|---|
| `results/ablation_bgfg/encoded/` | Every clip × config × CRF, encoded mp4s |
| `results/ablation_bgfg/summary.md` | Aggregate metric table |
| `results/ablation_bgfg/rd_curve.png` | The RD curve plot |
| `results/figures/qualitative_catalog_v2.pdf` | 180-page comparison PDF |
| `results/comparisons/` | Single-clip 3/4-panel demo videos (compare_clip.py output) |
| `results/saliency_videos/` | Saliency-overlay videos (saliency_video.py output) |
| `results/photo_demo/comparison.png` | Latest webcam-demo result |
| `docs/architecture.md` | System overview with mermaid diagrams |
| `docs/writeup_updates.md` | Drop-in sections for analysis.docx |
| `docs/demo_scripts.md` | This file |

---

## 8. Pre-built demo assets (bake these ahead of a live demo)

Run these once now so the videos are ready and you don't need to wait
during the demo:

```bash
# Three headline demo videos — the ones you actually plan to show
python scripts/compare_clip.py --clip clip_14 --crf 22 --out results/comparisons/headline_fall.mp4
python scripts/compare_clip.py --clip clip_02 --crf 22 --out results/comparisons/headline_grab.mp4
python scripts/compare_clip.py --clip clip_04 --crf 22 --out results/comparisons/headline_pool.mp4

# Saliency-only video for the "how does saliency work" slide
python scripts/saliency_video.py --clip clip_14 --side-by-side --out results/comparisons/saliency_demo.mp4

# Background-reference still (the "empty room" extraction)
python -c "
import sys; sys.path.insert(0,'.')
from src.bg_fg_codec import compute_background_median
import cv2
for clip in ['clip_14', 'clip_02', 'clip_04']:
    bg = compute_background_median(f'data/real/{clip}.mp4', n_samples=30)
    cv2.imwrite(f'results/comparisons/{clip}_background.png', bg)
print('wrote 3 background refs')
"
```

After running, you have a pre-built `results/comparisons/` folder with
everything you need, no live encoding required during the presentation.
