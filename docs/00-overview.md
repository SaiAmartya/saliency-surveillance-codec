# Project Overview

## The problem
Over 95% of home security camera footage is never viewed, yet every hour of it costs storage, bandwidth, and energy. The project is built around three connected questions:

1. How can stored video be compressed using principles of human perception, so storage drops without hurting perceived quality?
2. How can "useful" footage be identified automatically from the camera's own sensor data?
3. Which algorithms and edge-processing techniques keep that system scalable, privacy-conscious, and precise?

## Our answer
A **perceptually-guided, sensor-gated compression pipeline** that:

1. Uses a multi-signal sensor gate (motion + optical flow + on-device person detection + audio events) to decide when footage is "useful" enough to record at full fidelity.
2. Applies **saliency-aware perceptual compression** to every stored frame — bits go where a human's eyes would actually look, the rest is aggressively compressed.
3. Ships a side-by-side **neural-codec prototype** (a tiny conv autoencoder) so we can directly answer the inspiration video's question — "AI compression is 300× better, but we don't use it" — with real numbers showing why a hybrid is the realistic answer for edge hardware today.

## Where things live in this repo
- [`PLAN.md`](../PLAN.md) — the original design plan: architecture, scientific foundations, evaluation plan, risks.
- [`README.md`](../README.md) — quick-start and code map for engineers landing on the repo.
- [`docs/research/`](research/) — research briefs gathered during planning. Useful when you're writing the analysis sections.
- [`src/`](../src/) — implementation.
- [`scripts/`](../scripts/) — CLI entry points (test-video generator, pipeline run, live demo).
