"""
Qualitative comparison catalog v2 — 4-panel PDF for the bg/fg codec.

Each page:
    original | saliency overlay | baseline H.265 | ours_bgfg

Sweeps every clip × CRF × multiple frames so you can pick the strongest
demo example. Assumes you've already run ablation_bgfg.py first so
the encoded mp4s exist in results/ablation_bgfg/encoded/.

Usage:
    python scripts/qualitative_catalog_v2.py
    python scripts/qualitative_catalog_v2.py --saliency yolo+spectral
    python scripts/qualitative_catalog_v2.py --crfs 28 34          # subset
    python scripts/qualitative_catalog_v2.py --frames 0.25 0.55    # frame fractions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.saliency import SaliencyEstimator

DATA = ROOT / "data/real"
ENCODED = ROOT / "results/ablation_bgfg/encoded"
DEFAULT_OUT = ROOT / "results/figures/qualitative_catalog_v2.pdf"


def load_manifest() -> dict[str, str]:
    items = json.loads((DATA / "manifest.json").read_text())
    return {it["clip_id"]: it["action_class"] for it in items}


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def get_frame(path: Path, idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def saliency_overlay(rgb_frame: np.ndarray, estimator: SaliencyEstimator) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    sal = estimator.predict(bgr)
    sal_u8 = (np.clip(sal, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(bgr, 0.55, heat, 0.45, 0.0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def kb(path: Path) -> float:
    return path.stat().st_size / 1024 if path.exists() else 0.0


def render_page(
    pdf: PdfPages,
    clip_id: str,
    action_class: str,
    frame_idx: int,
    crf: int,
    orig_path: Path,
    base_path: Path,
    bgfg_path: Path | None,
    sal_estimator: SaliencyEstimator,
) -> bool:
    orig = get_frame(orig_path, frame_idx)
    base = get_frame(base_path, frame_idx)
    bgfg = get_frame(bgfg_path, frame_idx) if (bgfg_path and bgfg_path.exists()) else None
    if orig is None or base is None or bgfg is None:
        return False

    sal_img = saliency_overlay(orig, sal_estimator)

    panels = [
        (orig, "original"),
        (sal_img, "saliency overlay"),
        (base, f"baseline H.265 (CRF {crf})\n{kb(base_path):.0f} KB"),
    ]
    bg_pct = (1 - kb(bgfg_path) / kb(base_path)) * 100 if kb(base_path) > 0 else 0
    panels.append((bgfg, f"ours_bgfg (CRF {crf})\n{kb(bgfg_path):.0f} KB  ({bg_pct:+.0f}%)"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.85, 3.7))
    if n == 1:
        axes = [axes]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f'{clip_id} — action class "{action_class}", frame {frame_idx} of {frame_count(orig_path)}',
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crfs", type=int, nargs="*", default=[22, 28, 34])
    parser.add_argument("--frames", type=float, nargs="*", default=[0.20, 0.45, 0.70],
                        help="Frame positions as fractions of clip length")
    parser.add_argument("--clips", nargs="*", default=None,
                        help="Specific clip IDs. Default: all from manifest")
    parser.add_argument("--saliency", default="yolo+spectral",
                        choices=["spectral", "finegrained", "yolo", "yolo+spectral"],
                        help="Saliency backend used for the overlay panel")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = load_manifest()
    if args.clips:
        manifest = {k: v for k, v in manifest.items() if k in args.clips}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sal_estimator = SaliencyEstimator(backend=args.saliency)

    rendered, skipped = 0, 0
    with PdfPages(args.out) as pdf:
        # cover
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.72, "Qualitative comparison catalog v2",
                ha="center", va="center", fontsize=24, fontweight="bold")
        ax.text(0.5, 0.58, "H.265 vs ours_bgfg",
                ha="center", va="center", fontsize=14)
        ax.text(0.5, 0.40,
                f"{len(manifest)} clips × {len(args.crfs)} CRFs × {len(args.frames)} frames\n"
                "Panels: original | saliency overlay | baseline H.265 | ours_bgfg\n"
                f"Saliency overlay uses backend = '{args.saliency}'\n"
                "% under each ours panel = bytes saved vs baseline at the same CRF.",
                ha="center", va="center", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight", facecolor="white"); plt.close(fig)

        for clip_id, action_class in sorted(manifest.items()):
            orig_path = DATA / f"{clip_id}.mp4"
            if not orig_path.exists():
                continue
            n = frame_count(orig_path)
            frames = [max(0, min(n - 1, int(n * f))) for f in args.frames]
            for frame_idx in frames:
                for crf in args.crfs:
                    base_path = ENCODED / f"baseline_crf{crf}_{clip_id}.mp4"
                    bgfg_path = ENCODED / f"ours_bgfg_crf{crf}_{clip_id}.mp4"
                    if not base_path.exists() or not bgfg_path.exists():
                        skipped += 1
                        continue
                    ok = render_page(
                        pdf, clip_id, action_class, frame_idx, crf,
                        orig_path, base_path, bgfg_path,
                        sal_estimator,
                    )
                    if ok:
                        rendered += 1
                        if rendered % 10 == 0:
                            print(f"  rendered {rendered} pages...")
                    else:
                        skipped += 1

    print(f"\ndone: {rendered} pages rendered, {skipped} skipped")
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
