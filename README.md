# 🚀 Real-ESRGAN Video & Image Frame Upscaler (720p ➔ 1080p / 4K)

A high-performance, standalone Python suite for upscaling videos and image frame sequences using **Real-ESRGAN** models.

---

## ✨ Features

- 🏎️ **In-Memory Streaming (Zero-Disk Overhead)**: Uses FFmpeg subprocess pipes to stream uncompressed raw frames directly into PyTorch and back out into FFmpeg. No massive intermediate PNG dumps on disk.
- ⚡ **Hardware Accelerated**: Automatically leverages Apple Silicon Metal (`mps`), NVIDIA CUDA (`cuda`), or CPU with FP16 half-precision.
- 🎯 **Proper 1.5x Supersampling (720p ➔ 1080p)**: Runs AI super-resolution and applies an anti-aliased Lanczos downsampler to exactly $1920 \times 1080$.
- 🔊 **Audio & Metadata Preservation**: Automatically preserves audio tracks, bitrates, and sync without re-encoding audio.
- 🧩 **Multiple AI Models Supported**:
  - `compact` *(Default)*: `realesr-general-x4v3` – Best balance of detail, artifact removal, and speed for real-world footage.
  - `anime`: `realesr-animevideov3` – Ultra-fast, sharp line art and animation.
  - `x4plus`: `RealESRGAN_x4plus` – Maximum photo detail reconstruction.
  - `x2plus`: `RealESRGAN_x2plus` – Native 2x model.
- 🗂️ **Batch & Sequence Support**: Upscale entire folders of image frames (`.png`, `.jpg`, etc.) with multithreaded I/O.

---

## 🛠️ Setup & Requirements

1. **Activate the Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **(Optional) If installing elsewhere**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage Examples

### 1. Upscale a Video (720p ➔ 1080p)
```bash
python upscale.py --input my_video_720p.mp4 --output my_video_1080p.mp4
```

### 2. Upscale an Animation / Anime Video (Faster)
```bash
python upscale.py --input anime_clip.mp4 --output anime_1080p.mp4 --model anime --batch-size 2
```

### 3. Upscale a Directory of Image Frames
```bash
python upscale.py --input ./frames_720p/ --output ./frames_1080p/ --target-res 1920x1080
```

### 4. Upscale a Single Image
```bash
python upscale.py --input picture.jpg --output picture_1080p.png
```

### 5. Upscale to 4K / 2160p
```bash
python upscale.py --input input.mp4 --output output_4k.mp4 --target-res 4k
```

---

## ⚙️ CLI Options & Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-i`, `--input` | *Required* | Path to input video file, image file, or folder of frames. |
| `-o`, `--output` | Auto | Path to output video, image, or folder. |
| `-m`, `--model` | `compact` | Choice of model: `compact`, `anime`, `x4plus`, `x2plus`. |
| `-t`, `--target-res` | `1920x1080` | Target resolution: `1920x1080`, `1080p`, `4k`, `1440p`, or `native`. |
| `--tile` | `512` | Tile size to prevent GPU out-of-memory errors (`0` to disable). |
| `--batch-size` | `1` | Frames processed per batch on GPU (increase to `2` or `4` on fast GPUs). |
| `--device` | `auto` | Compute device: `auto`, `mps`, `cuda`, `cpu`. |
| `--crf` | `18` | H.264 video quality factor (17–23 is visually lossless). |
| `--preset` | `medium` | FFmpeg encoding speed preset (`fast`, `medium`, `slow`). |
| `--max-frames` | None | Limit processing to first N frames (useful for quick previews). |
