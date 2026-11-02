"""
Saliency-aware surveillance codec — perceptually-guided, sensor-gated compression
pipeline for home-surveillance video.

Submodules:
  gate          - useful-footage detection (motion + flow + YOLO)
  saliency      - per-frame saliency map generation
  compress      - saliency-aware perceptual compression (H.265 or fallback)
  neural_codec  - tiny conv autoencoder for side-by-side comparison
  metrics       - PSNR / SSIM / LPIPS quality metrics
  pipeline      - end-to-end orchestrator
"""
__version__ = "0.1.0"
