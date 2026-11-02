"""
neural_codec.py — Tiny convolutional autoencoder for video compression demo.

A prototype neural codec for this project: shows side-by-side comparison
of classical (H.265 saliency-aware) vs. neural (conv autoencoder) compression.
The point is to demonstrate trade-offs: aggressive compression vs. slow inference.

Public API:
  - TinyAutoencoder: nn.Module, 3x256x256 -> 32x32x32 latent -> 3x256x256
  - NeuralCodec: high-level wrapper with train_on_frames, encode, decode, benchmark
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class TinyAutoencoder(nn.Module):
    """Convolutional autoencoder: 3x256x256 -> latent_channels x32x32 -> 3x256x256.

    Achieves ~8x spatial compression in each dimension via 3 stride-2 conv blocks.
    Decoder mirrors encoder with ConvTranspose2d.
    Total params: ~150K-300K (small, interpretable baseline).
    """

    def __init__(self, latent_channels: int = 32):
        """Initialize encoder and decoder.

        Args:
            latent_channels: Number of channels in the bottleneck (default 32).
        """
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: 3 -> 32 -> 64 -> latent_channels, each stride=2
        self.encoder = nn.Sequential(
            # In: 3x256x256
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> 32x128x128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # In: 32x128x128
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 64x64x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # In: 64x64x64
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=2, padding=1),  # -> latent_channels x32x32
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder: latent_channels -> 64 -> 32 -> 3, each stride=2 upsample
        self.decoder = nn.Sequential(
            # In: latent_channels x32x32
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 64x64x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # In: 64x64x64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 32x128x128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # In: 32x128x128
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 3x256x256
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode and decode.

        Args:
            x: Input tensor, shape (B, 3, 256, 256), values in [0, 1].

        Returns:
            (reconstruction, latent): reconstruction is (B, 3, 256, 256) in [0, 1];
                                      latent is (B, latent_channels, 32, 32).
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


class NeuralCodec:
    """High-level wrapper for neural video codec: train, encode, decode, benchmark.

    Handles BGR<->RGB conversion, resizing to 256x256, quantization, and I/O.
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        """Initialize codec.

        Args:
            weights_path: Optional path to load pre-trained weights.
            device: "cpu" or "cuda".
        """
        self.device = torch.device(device)
        self.model = TinyAutoencoder(latent_channels=32).to(self.device)
        self.latent_channels = 32
        self.target_h, self.target_w = 256, 256

        # Quantization parameters: latent values clipped to [-4, 4], then scaled to int8
        self.quant_min, self.quant_max = -4.0, 4.0
        self.int8_min, self.int8_max = -128, 127

        if weights_path is not None:
            self.load(weights_path)

    def _bgr_to_tensor(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Convert BGR uint8 HxWx3 frame to RGB float32 tensor [0,1], resized to 256x256.

        Args:
            frame_bgr: HxWx3 uint8 BGR.

        Returns:
            Bx3x256x256 float32 RGB tensor in [0, 1].
        """
        import cv2
        # Resize to 256x256
        frame_resized = cv2.resize(frame_bgr, (self.target_w, self.target_h))
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # uint8 -> float32 [0, 1]
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        # HxWx3 -> 1x3xHxW
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
        return frame_tensor.to(self.device)

    def _tensor_to_bgr(self, tensor: torch.Tensor, out_hw: tuple[int, int]) -> np.ndarray:
        """Convert 1x3x256x256 float32 RGB tensor [0,1] back to HxWx3 uint8 BGR.

        Args:
            tensor: 1x3x256x256 float32 in [0, 1].
            out_hw: (H, W) for resizing output.

        Returns:
            HxWx3 uint8 BGR.
        """
        import cv2
        # 1x3xHxW -> HxWx3
        img_rgb = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Clip to [0, 1] and scale to uint8
        img_rgb = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
        # RGB -> BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        # Resize to output dimensions
        img_bgr = cv2.resize(img_bgr, (out_hw[1], out_hw[0]))
        return img_bgr

    def train_on_frames(self, frames: np.ndarray, epochs: int = 10, lr: float = 1e-3,
                       batch_size: int = 8, log_every: int = 50) -> dict:
        """Train autoencoder on a batch of frames.

        Args:
            frames: NxHxWx3 uint8 BGR.
            epochs: Number of training epochs.
            lr: Learning rate for Adam.
            batch_size: Batch size for training.
            log_every: Log loss every N batches.

        Returns:
            dict: {epoch: mean_loss_per_epoch, ...}
        """
        import cv2

        # Preprocess: resize all frames to 256x256, convert BGR->RGB, scale to [0,1]
        n_frames = frames.shape[0]
        frames_tensor = []
        for i in range(n_frames):
            frame = frames[i]  # HxWx3 uint8 BGR
            frame_resized = cv2.resize(frame, (256, 256))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)  # 3xHxW
            frames_tensor.append(frame_tensor)

        frames_tensor = torch.stack(frames_tensor)  # Nx3x256x256
        dataset = TensorDataset(frames_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        loss_history = {}

        for epoch in range(epochs):
            epoch_losses = []
            batch_count = 0

            iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") if HAS_TQDM else dataloader

            for batch_idx, (batch_frames,) in enumerate(iterator):
                batch_frames = batch_frames.to(self.device)

                optimizer.zero_grad()
                recon, _ = self.model(batch_frames)
                loss = loss_fn(recon, batch_frames)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                batch_count += 1

                if not HAS_TQDM and batch_count % log_every == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_count}, Loss: {loss.item():.6f}")

            mean_loss = np.mean(epoch_losses)
            loss_history[epoch] = mean_loss

            if not HAS_TQDM:
                print(f"Epoch {epoch+1}/{epochs}, Mean Loss: {mean_loss:.6f}")

        return loss_history

    def encode(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Encode a single BGR frame to quantized latent bytes.

        Args:
            frame_bgr: HxWx3 uint8 BGR.

        Returns:
            1D array of int8 bytes (latent_channels * 32 * 32 = 32768 bytes).
        """
        self.model.eval()
        with torch.no_grad():
            frame_tensor = self._bgr_to_tensor(frame_bgr)
            _, latent = self.model(frame_tensor)

            # latent: 1 x latent_channels x 32 x 32
            latent_np = latent.squeeze(0).cpu().numpy()  # latent_channels x 32 x 32

            # Clip to quantization range and scale to int8
            latent_np = np.clip(latent_np, self.quant_min, self.quant_max)
            # Linear scale from [quant_min, quant_max] to [int8_min, int8_max]
            latent_int8 = (latent_np - self.quant_min) / (self.quant_max - self.quant_min)
            latent_int8 = (latent_int8 * (self.int8_max - self.int8_min) + self.int8_min).astype(np.int8)

            # Flatten to 1D bytes
            return latent_int8.flatten()

    def decode(self, latent_bytes: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        """Decode quantized latent bytes back to BGR frame.

        Args:
            latent_bytes: 1D int8 array (latent_channels * 32 * 32 bytes).
            out_hw: (H, W) output frame size.

        Returns:
            HxWx3 uint8 BGR.
        """
        self.model.eval()
        with torch.no_grad():
            # Unquantize: reverse the int8 scaling
            latent_np = latent_bytes.astype(np.float32)
            latent_np = (latent_np - self.int8_min) / (self.int8_max - self.int8_min)
            latent_np = latent_np * (self.quant_max - self.quant_min) + self.quant_min

            # Reshape to latent_channels x 32 x 32
            latent_np = latent_np.reshape(self.latent_channels, 32, 32)
            latent_tensor = torch.from_numpy(latent_np).float().unsqueeze(0).to(self.device)

            # Decode
            recon = self.model.decoder(latent_tensor)

            # Convert back to BGR uint8
            return self._tensor_to_bgr(recon, out_hw)

    def benchmark(self, frame_bgr: np.ndarray, n_iters: int = 20) -> dict:
        """Benchmark encode/decode speed and compression ratio.

        Args:
            frame_bgr: HxWx3 uint8 BGR test frame.
            n_iters: Number of iterations to measure (median used).

        Returns:
            dict: {
                'encode_ms': median encode time (ms),
                'decode_ms': median decode time (ms),
                'bytes_per_frame': latent size in bytes,
                'compression_ratio': original_bytes / latent_bytes
            }
        """
        self.model.eval()

        # Warmup
        for _ in range(3):
            _ = self.encode(frame_bgr)

        # Measure encode
        encode_times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            latent_bytes = self.encode(frame_bgr)
            t1 = time.perf_counter()
            encode_times.append((t1 - t0) * 1000)  # ms

        # Measure decode
        decode_times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = self.decode(latent_bytes, frame_bgr.shape[:2])
            t1 = time.perf_counter()
            decode_times.append((t1 - t0) * 1000)  # ms

        # Compute compression ratio
        original_bytes = frame_bgr.size  # H*W*3
        latent_bytes_size = latent_bytes.nbytes
        compression_ratio = original_bytes / latent_bytes_size if latent_bytes_size > 0 else 0.0

        return {
            'encode_ms': float(np.median(encode_times)),
            'decode_ms': float(np.median(decode_times)),
            'bytes_per_frame': int(latent_bytes_size),
            'compression_ratio': float(compression_ratio),
        }

    def save(self, path: str) -> None:
        """Save model weights and config.

        Args:
            path: File path to save checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'latent_channels': self.latent_channels,
            'quant_min': self.quant_min,
            'quant_max': self.quant_max,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model weights and config.

        Args:
            path: File path to load checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.latent_channels = checkpoint.get('latent_channels', 32)
        self.quant_min = checkpoint.get('quant_min', -4.0)
        self.quant_max = checkpoint.get('quant_max', 4.0)


if __name__ == "__main__":
    """End-to-end sanity check: train, benchmark, encode/decode, compute PSNR."""
    import cv2
    try:
        from skimage.metrics import peak_signal_noise_ratio
        HAS_SKIMAGE = True
    except ImportError:
        HAS_SKIMAGE = False
        print("Warning: skimage not available; PSNR will not be computed.")

    print("=" * 60)
    print("Neural Codec Sanity Check")
    print("=" * 60)

    # Create synthetic dataset: 50 random 256x256 BGR frames
    print("\nGenerating 50 synthetic frames...")
    np.random.seed(42)
    synthetic_frames = np.random.randint(0, 256, size=(50, 256, 256, 3), dtype=np.uint8)

    # Initialize codec
    print("Initializing NeuralCodec on CPU...")
    codec = NeuralCodec(device="cpu")

    # Train for 2 epochs with small batch
    print("\nTraining for 2 epochs (batch_size=8)...")
    loss_history = codec.train_on_frames(synthetic_frames, epochs=2, lr=1e-3, batch_size=8)
    for epoch, loss in loss_history.items():
        print(f"  Epoch {epoch}: loss = {loss:.6f}")

    # Benchmark
    print("\nBenchmarking (20 iterations)...")
    test_frame = synthetic_frames[0]
    bench = codec.benchmark(test_frame, n_iters=20)
    print(f"  Encode time (median): {bench['encode_ms']:.2f} ms")
    print(f"  Decode time (median): {bench['decode_ms']:.2f} ms")
    print(f"  Latent bytes per frame: {bench['bytes_per_frame']}")
    print(f"  Compression ratio: {bench['compression_ratio']:.2f}x")

    # Encode/decode one sample
    print("\nEncoding and decoding one frame...")
    latent = codec.encode(test_frame)
    reconstructed = codec.decode(latent, test_frame.shape[:2])

    # Compute PSNR if available
    if HAS_SKIMAGE:
        psnr = peak_signal_noise_ratio(test_frame, reconstructed)
        print(f"  PSNR: {psnr:.2f} dB")
    else:
        mse = np.mean((test_frame.astype(float) - reconstructed.astype(float)) ** 2)
        print(f"  MSE: {mse:.4f}")

    print("\n" + "=" * 60)
    print("Sanity check complete!")
    print("=" * 60)
