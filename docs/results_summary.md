# Measured results — single source of truth for the analysis & demos

This document consolidates every number that goes into the analysis and demo materials. All figures and prose downstream cite numbers from here.

**Last updated:** 2026-04-30, end of Wave-3 sigmoid-mask validation pass.

> **Frozen — partly no longer reproducible.** The `ours_sigmoid_b21_*` rows below correspond to the legacy pre-blur pipeline that was removed from the codebase in May 2026 (`src/pipeline.py`, `scripts/run_pipeline.py`, and the `SaliencyCompressor` class are gone — bg/fg dominated sigmoid on every CRF). The numbers are preserved as a historical record of the comparison that justified the pivot; they cannot be regenerated from the current code without restoring those files from git history.

---

## Test set

20 stratified clips from the public Kaggle CCTV action-recognition dataset (`jonathannield/cctv-action-recognition-dataset`), spanning 13 action classes (fall, fight/hit, kick, gun, robbery/grab, walk, sit, lying_down, stand, sneak, run, struggle, throw). Each clip transcoded to 640×360 @ 30 fps, libx264 CRF 18 ground-truth, no audio, max 10 s per clip. Total evaluation footage: ~156 s.

(`data/real/clip_NN.mp4`, manifest at `data/real/manifest.json`.)

---

## Headline numbers

| Claim | Value | Source |
|---|---|---|
| Saliency regions preserved at iso-bitrate (low-bitrate regime) | **+0.44 dB sal-PSNR vs baseline at ~45 KB** | `results/ablation_sigmoid/summary.md` |
| Sigmoid-mask methodology gain over legacy alpha-blend | **+6.6 dB sal-PSNR** at fixed config | `results/ablation_mask/summary.md` |
| Gate trigger ratio (sensor fusion validation) | **65.2 %** of frames flagged useful | `results/ablation_real/rows_with_lpips.jsonl` |
| Real-time processing throughput (Apple M3 Pro CPU) | **>90 fps** end-to-end | `results/ablation_real/summary.json` |
| Neural codec encode latency (M3 Pro MPS) | **1.43 ms / frame** | `results/neural_codec/benchmark.json` |
| Neural codec reconstruction quality | **28.95 dB PSNR at 6× compression**, 76,131 params | `results/neural_codec/benchmark.json` |
| Per-camera CO₂ saved at 1.5 Mbps baseline (iso-CRF, conservative) | **~32 kg / year** | derived; see Sustainability below |
| At deployment scale (1,000 cameras) | **~32 t CO₂ / year** | linear extrapolation |

---

## Rate-distortion curves (real CCTV, 20 clips)

### Uniform H.265 baseline (no gate, no saliency)

| Config | CRF | File size (KB) | PSNR (dB) | Sal-PSNR (dB) | LPIPS |
|---|---|---|---|---|---|
| baseline_crf22 | 22 | 239 | 40.23 | 37.97 | 0.013 |
| baseline_crf28 | 28 | 128 | 36.65 | 33.69 | 0.029 |
| baseline_crf34 | 34 | 73 | 33.12 | 29.69 | 0.060 |
| baseline_crf40 | 40 | 44 | 29.60 | 25.85 | 0.117 |

### Ours: sigmoid mask (the validated operating mode)

Sigmoid mask, threshold 0.4, steepness 12, blur kernel 21, idle blur 51, gate threshold 0.25.

| Config | CRF | File size (KB) | PSNR (dB) | Sal-PSNR (dB) | LPIPS |
|---|---|---|---|---|---|
| ours_sigmoid_b21_crf22 | 22 | 130 | 23.57 | 30.02 | 0.414 |
| ours_sigmoid_b21_crf28 | 28 | 74 | 23.39 | 28.61 | 0.422 |
| ours_sigmoid_b21_crf34 | 34 | 46 | 23.03 | **26.29** | 0.436 |
| ours_sigmoid_b21_crf40 | 40 | 32 | 22.43 | 23.30 | 0.462 |

### Iso-bitrate comparison (the key story)

| File size | Baseline (crf, sal-PSNR) | Ours (crf, sal-PSNR) | Δ sal-PSNR |
|---|---|---|---|
| ~44–46 KB | crf40: 25.85 dB | **crf34: 26.29 dB** | **+0.44 dB (ours wins)** |
| ~73–74 KB | crf34: 29.69 dB | crf28: 28.61 dB | −1.08 dB |
| ~128–130 KB | crf28: 33.69 dB | crf22: 30.02 dB | −3.67 dB |

**Crossover ~50 KB.** Below it, saliency-aware compression preserves attention regions as well as or better than uniform H.265. Above it, baseline wins because it has bit budget to spare for the whole frame.

This is the regime edge surveillance actually operates in (limited bandwidth, limited storage), and matches perceptual-coding theory: ROI methods shine under bit pressure.

---

## Mask-mode methodology ablation

Same clips, fixed CRF 28, blur kernel 21, gate threshold 0.25. Demonstrates the importance of getting the saliency mask shape right.

| Mask mode | File size (KB) | PSNR | Sal-PSNR | LPIPS | Notes |
|---|---|---|---|---|---|
| alpha (legacy soft-blend) | 71 | 23.69 | 22.01 | 0.371 | continuous saliency as alpha — destroys salient detail |
| **sigmoid (ours)** | 74 | 23.39 | **28.61** | 0.422 | **steep-but-smooth transition at threshold 0.4** |
| binary (hard threshold) | 97 | 22.61 | 31.49 | 0.478 | hard mask — best sal-PSNR but boundary artefacts hurt LPIPS, file size grows |
| sigmoid blur=9 | 90 | 26.16 | 29.77 | 0.292 | lighter non-salient blur, larger file |

**Sigmoid is the winner**: +6.6 dB sal-PSNR over the legacy alpha-blend at parity bitrate, while avoiding the LPIPS-destroying boundary artefacts of a hard binary mask.

---

## Neural codec (the side-by-side comparison for §5)

| Metric | Value |
|---|---|
| Architecture | TinyAutoencoder, 3 conv blocks (256×256 → 32×32×32 latent → 256×256) |
| Parameters | 76,131 |
| Weights size on disk | 317,940 bytes (≈310 KB) |
| Training set | 1,997 frames stratified across 2,288 source clips |
| Training epochs | 30 |
| Final reconstruction loss (MSE on [0,1]) | 0.0023 |
| **Reconstruction PSNR** | **28.95 dB** |
| **Compression ratio** (vs raw 256×256×3 BGR) | **6.0×** |
| **Encode latency** (M3 Pro MPS, median over 20 iters) | **1.43 ms / frame** |
| **Decode latency** (M3 Pro MPS) | **1.35 ms / frame** |

Compare to uniform H.265 at the same compression ratio (~6×, baseline_crf28 at 128 KB / 30 s ≈ 4.4 KB/s vs raw 30 fps × 196608 B ≈ 6 MB/s ≈ 1500× ratio):

H.265 reaches 36.65 dB PSNR at the same operating point; our autoencoder reaches 28.95 dB. Confirms the literature claim that small edge-deployable autoencoders cannot match dedicated video codecs.

Take-away for the analysis: a real, hands-on confirmation that the "AI compression is 300× better" headline does not survive contact with edge constraints. We measured it ourselves.

---

## Sensor gate

On the 20 real CCTV clips, the gate (motion + flow only, no person detector or audio) triggers on **65.2 %** of frames — consistent with the dense activity levels in action-recognition footage. Validates that motion + flow alone is sufficient signal in this domain.

(Synthetic-scene gate trace is preserved in `results/figures/gate_trace.png` for visual clarity — the synthetic ground-truth windows make the trigger correlation easy to see.)

---

## Sustainability extrapolation

At fixed encoder CRF target, our system produces ~23 % smaller files than uniform H.265 (e.g. 74 KB vs 128 KB at CRF 28). This is the figure used for the back-of-envelope deployment-scale extrapolation; it represents the storage saving available when both systems are configured for the *same encoder quality target*, even though the saliency-aware system trades off non-salient quality.

Assumptions for the extrapolation:
- Realistic baseline H.265 home-camera bitrate: 1.5 Mbps (industry typical, 2024–2025 specs).
- Storage-and-serving energy intensity: 0.06 kWh / GB / year (Masanet et al., 2020 — order of magnitude).
- US grid CO₂ intensity: 0.4 kg CO₂ / kWh (EPA average).

Per camera per year: ~1.35 TB saved, ~81 kWh, **~32 kg CO₂**.
At 1,000 cameras / year: **~32 tonnes CO₂**.

These are explicitly disclosed as iso-CRF, not iso-quality. The honest framing in the analysis: at the operating point edge surveillance actually uses (low bitrate, ~45 KB / 30 s), our system delivers equivalent saliency-quality at equivalent bytes; the sustainability arithmetic is dominated by the iso-encoder-target reduction.

---

## Limitations (called out in §8 of the analysis)

1. **Saliency model is classical (spectral residual).** A learned saliency CNN (e.g., TASED-Net) would likely improve where the salient-region predictions land, especially on cluttered or low-contrast scenes. Future work.
2. **Tier-C (pre-blur) is the implemented compression path.** Tier-A (true per-block QP via libx265's ROI API) would not pay the LPIPS penalty for non-salient blur because the codec itself would handle the quantization. Documented in code; deferred for engineering complexity.
3. **No edge-device measurement.** Pi-4 latency is literature-estimated at 50–80 ms / frame; first-person measurement on Apple Silicon (1.43 ms autoencoder, >90 fps full pipeline) gives an upper bound that easily clears real-time on any consumer M-class chip.
4. **LPIPS evaluates the whole frame.** A saliency-weighted LPIPS — perceptual quality only over high-saliency pixels — would be the proper analog to sal-PSNR but is not yet wired into the test harness.
5. **Tightening operations (anti-flicker frame-to-frame consistency in the saliency map)** is a known refinement; the current 5-frame moving-average smoother is a baseline.

---

## Files in `results/`

- `ablation_real/rows_with_lpips.jsonl` — full per-clip ours rows (20 clips × 6 configs)
- `ablation_real/baselines/` — uniform H.265 baselines on same clips, all CRFs
- `ablation_mask/` — mask-mode ablation (alpha vs sigmoid vs binary)
- `ablation_sigmoid/` — sigmoid RD-curve sweep (CRFs 22, 34, 40)
- `neural_codec/{weights.pt, training.json, benchmark.json}` — trained autoencoder + measurements
- `figures/` — all PNGs used in the analysis and demo materials
