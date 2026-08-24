import os
import sys
import subprocess
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from upscaler import RealESRGANUpscaler


def probe_video(video_path: str):
    """
    Extract video metadata using OpenCV & ffprobe fallback.
    Returns: (width, height, fps, total_frames, has_audio)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # If fps is invalid or 0, default to 30.0
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    # If frame count is 0, estimate using duration from ffprobe or container
    if total_frames <= 0:
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets',
                '-of', 'csv=p=0',
                video_path
            ]
            res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            total_frames = int(res) if res.isdigit() else 0
        except Exception:
            total_frames = 0

    # Check if audio exists
    has_audio = False
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=index',
            '-of', 'csv=p=0',
            video_path
        ]
        res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        has_audio = len(res) > 0
    except Exception:
        has_audio = False

    return width, height, fps, total_frames, has_audio


def process_video(
    input_path: str,
    output_path: str,
    upscaler: RealESRGANUpscaler,
    target_res: tuple = (1920, 1080),
    batch_size: int = 1,
    crf: int = 18,
    preset: str = 'medium',
    encoder: str = 'libx264',
    max_frames: int = None
):
    """
    Stream video frames from FFmpeg directly through Real-ESRGAN and back into FFmpeg.
    Preserves original audio and container sync.
    """
    in_w, in_h, fps, total_frames, has_audio = probe_video(input_path)
    target_w, target_h = target_res

    if max_frames is not None and max_frames > 0:
        total_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames

    print("\n" + "=" * 60)
    print(f"🎬 VIDEO UPSCALING PIPELINE")
    print(f"   Input:        {input_path} ({in_w}x{in_h} @ {fps:.2f} fps)")
    print(f"   Target:       {target_w}x{target_h} (1080p)")
    print(f"   Scale Factor: {upscaler.scale}x -> downsampled to target")
    print(f"   Model:        {upscaler.model_info['description']}")
    print(f"   Device:       {upscaler.device.type.upper()} (FP16: {upscaler.half})")
    print(f"   Batch Size:   {batch_size} | Tile: {upscaler.tile}")
    print(f"   Audio Track:  {'Preserved' if has_audio else 'None'}")
    print("=" * 60 + "\n")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Start FFmpeg input stream reader
    input_cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-'
    ]
    if max_frames is not None and max_frames > 0:
        input_cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vframes', str(max_frames),
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            '-'
        ]

    ffmpeg_in = subprocess.Popen(
        input_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=in_w * in_h * 3 * max(16, batch_size * 2)
    )

    # 2. Build FFmpeg output stream encoder
    output_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{target_w}x{target_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',                # video from stdin pipe
    ]

    if has_audio:
        output_cmd += [
            '-i', input_path,     # audio source
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:a', 'copy',
        ]
    else:
        output_cmd += [
            '-map', '0:v:0',
        ]

    output_cmd += [
        '-c:v', encoder,
        '-preset', preset,
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    ffmpeg_out = subprocess.Popen(
        output_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=target_w * target_h * 3 * max(16, batch_size * 2)
    )

    frame_bytes = in_w * in_h * 3
    processed_count = 0
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing Frames", unit="frame")
    start_time = time.time()

    try:
        while True:
            # Accumulate a batch of frames
            batch_frames = []
            for _ in range(batch_size):
                raw_bytes = ffmpeg_in.stdout.read(frame_bytes)
                if not raw_bytes or len(raw_bytes) < frame_bytes:
                    break
                frame_np = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((in_h, in_w, 3))
                batch_frames.append(frame_np)

            if not batch_frames:
                break

            current_b = len(batch_frames)

            if current_b == 1:
                # Single frame inference
                enhanced_bgr = upscaler.enhance_frame(batch_frames[0], target_size=(target_w, target_h))
                ffmpeg_out.stdin.write(enhanced_bgr.tobytes())
            else:
                # Batch inference
                # Stack BGR -> RGB tensors
                tensors = []
                for f in batch_frames:
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    tensors.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)

                batch_t = torch.stack(tensors, dim=0)
                out_t = upscaler.enhance_tensor(batch_t)

                # Resize to target on GPU if desired or CPU/Lanczos
                # Using GPU bicubic interpolation for fast batch downscale
                if out_t.shape[3] != target_w or out_t.shape[2] != target_h:
                    out_t = F.interpolate(out_t, size=(target_h, target_w), mode='bicubic', align_corners=False)

                out_np = out_t.permute(0, 2, 3, 1).float().clamp(0.0, 1.0).cpu().numpy()
                for i in range(current_b):
                    out_bgr = cv2.cvtColor((out_np[i] * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
                    ffmpeg_out.stdin.write(out_bgr.tobytes())

            processed_count += current_b
            pbar.update(current_b)

            if max_frames is not None and processed_count >= max_frames:
                break

    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user. Finalizing video...")
    finally:
        pbar.close()
        ffmpeg_in.stdout.close()
        ffmpeg_in.wait()
        ffmpeg_out.stdin.close()
        ffmpeg_out.wait()

    total_time = time.time() - start_time
    avg_fps = processed_count / total_time if total_time > 0 else 0
    print("\n" + "=" * 60)
    print(f"✅ Upscaling Finished!")
    print(f"   Frames Processed: {processed_count}")
    print(f"   Total Time:       {total_time:.2f}s ({avg_fps:.2f} FPS)")
    print(f"   Saved Output:     {output_path}")
    print("=" * 60 + "\n")
