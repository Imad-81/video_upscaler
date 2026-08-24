#!/usr/bin/env python3
import os
import sys
import argparse
from upscaler import RealESRGANUpscaler, MODEL_REGISTRY
from video_pipeline import process_video
from frames_pipeline import process_frames_directory, process_single_image, SUPPORTED_EXTENSIONS

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.webm', '.ts', '.m4v', '.wmv'}


def parse_target_resolution(res_str: str):
    """Parse '1920x1080' or '1080p' or '720p' or 'none'/'native'."""
    if not res_str or res_str.lower() in ('none', 'native', 'false', '0'):
        return None

    res_str = res_str.lower().strip()
    if res_str == '1080p':
        return (1920, 1080)
    elif res_str == '720p':
        return (1280, 720)
    elif res_str == '1440p' or res_str == '2k':
        return (2560, 1440)
    elif res_str == '4k' or res_str == '2160p':
        return (3840, 2160)

    if 'x' in res_str:
        parts = res_str.split('x')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return (int(parts[0]), int(parts[1]))

    raise ValueError(f"Invalid target resolution: '{res_str}'. Expected format like '1920x1080' or '1080p'.")


def main():
    parser = argparse.ArgumentParser(
        description="🚀 High-Performance Real-ESRGAN Video & Frame Upscaler (720p -> 1080p / 4K)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale 720p video to 1080p (Automatic MPS GPU acceleration on Mac):
  python upscale.py --input input_720p.mp4 --output output_1080p.mp4

  # Upscale with Anime model and batch size 4:
  python upscale.py --input anime.mp4 --output anime_1080p.mp4 --model anime --batch-size 4

  # Upscale a folder of image frames to 1080p:
  python upscale.py --input ./frames_720p/ --output ./frames_1080p/ --target-res 1920x1080

  # Upscale a single image:
  python upscale.py --input photo.jpg --output photo_upscaled.png
        """
    )

    parser.add_argument('-i', '--input', type=str, required=True, help="Path to input video file, image file, or directory of frames")
    parser.add_argument('-o', '--output', type=str, default=None, help="Path to output file or directory")
    parser.add_argument('-m', '--model', type=str, default='compact',
                        choices=list(MODEL_REGISTRY.keys()),
                        help="AI Model to use: 'compact' (fast, general video), 'anime' (animation/cartoons), 'x4plus' (high detail photo), 'x2plus' (native 2x)")
    parser.add_argument('-t', '--target-res', type=str, default='1920x1080',
                        help="Target output resolution (e.g. '1920x1080', '1080p', '4k', or 'native') [default: 1920x1080]")
    parser.add_argument('--tile', type=int, default=512,
                        help="Tile size for processing (default: 512). Set to 0 to disable tiling if you have large VRAM.")
    parser.add_argument('--batch-size', type=int, default=1,
                        help="Number of frames processed concurrently on GPU (default: 1, try 2 or 4 for faster GPU throughput)")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'mps', 'cuda', 'cpu'],
                        help="Compute device: 'auto' (detects Apple Silicon MPS or CUDA), 'mps', 'cuda', or 'cpu'")
    parser.add_argument('--crf', type=int, default=18,
                        help="H.264 video encoder CRF quality (17-23 is visually lossless, default: 18)")
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                        help="FFmpeg encoding speed preset (default: medium)")
    parser.add_argument('--max-frames', type=int, default=None,
                        help="Limit processing to the first N frames (useful for testing/previewing)")
    parser.add_argument('--fp32', action='store_true',
                        help="Force full FP32 precision instead of FP16 half precision")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"❌ Error: Input path '{input_path}' does not exist.")
        sys.exit(1)

    target_res = parse_target_resolution(args.target_res)

    # Initialize Real-ESRGAN Upscaler Engine
    print(f"🔄 Initializing Real-ESRGAN [{args.model}] engine...")
    upscaler = RealESRGANUpscaler(
        model_name=args.model,
        weights_dir='weights',
        device=args.device,
        tile=args.tile,
        half=not args.fp32
    )

    # Detect task type: Directory (frames), Single Image, or Video
    if os.path.isdir(input_path):
        # Folder of frames
        output_path = args.output or (input_path.rstrip('/\\') + '_upscaled')
        process_frames_directory(
            input_dir=input_path,
            output_dir=output_path,
            upscaler=upscaler,
            target_res=target_res,
            batch_size=args.batch_size
        )
    else:
        ext = os.path.splitext(input_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            # Video File
            if not args.output:
                base, ext_name = os.path.splitext(input_path)
                output_path = f"{base}_1080p{ext_name}"
            else:
                output_path = args.output

            process_video(
                input_path=input_path,
                output_path=output_path,
                upscaler=upscaler,
                target_res=target_res or (1920, 1080),
                batch_size=args.batch_size,
                crf=args.crf,
                preset=args.preset,
                max_frames=args.max_frames
            )
        elif ext in SUPPORTED_EXTENSIONS:
            # Single Image
            if not args.output:
                base, ext_name = os.path.splitext(input_path)
                output_path = f"{base}_upscaled{ext_name}"
            else:
                output_path = args.output

            process_single_image(
                input_path=input_path,
                output_path=output_path,
                upscaler=upscaler,
                target_res=target_res
            )
        else:
            print(f"❌ Error: Unsupported file format '{ext}'.")
            sys.exit(1)


if __name__ == '__main__':
    main()
