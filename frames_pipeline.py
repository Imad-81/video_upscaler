import os
import glob
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
from upscaler import RealESRGANUpscaler

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}

_worker_upscaler = None


def _init_worker(model_name: str, weights_dir: str, device_str: str, tile: int, half: bool):
    global _worker_upscaler
    import torch
    torch.set_num_threads(3)
    _worker_upscaler = RealESRGANUpscaler(
        model_name=model_name,
        weights_dir=weights_dir,
        device=device_str,
        tile=tile,
        half=half
    )


def _process_frame_worker(task):
    in_path, out_path, target_res = task
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    enhanced = _worker_upscaler.enhance_frame(img, target_size=target_res)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, enhanced)
    return True


def find_image_files(input_dir: str):
    """Find all supported image files in a directory sorted by name."""
    files = []
    for root, _, filenames in os.walk(input_dir):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, f))
    files.sort()
    return files


def process_single_image(
    input_path: str,
    output_path: str,
    upscaler: RealESRGANUpscaler,
    target_res: tuple = None
):
    """Upscale a single image file."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image file: {input_path}")

    h, w = img.shape[:2]
    print(f"\n🖼️ Upscaling Image: {input_path} ({w}x{h})")

    t0 = time.time()
    enhanced = upscaler.enhance_frame(img, target_size=target_res)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, enhanced)

    out_h, out_w = enhanced.shape[:2]
    print(f"✅ Saved to: {output_path} ({out_w}x{out_h}) in {elapsed:.2f}s\n")


def process_frames_directory(
    input_dir: str,
    output_dir: str,
    upscaler: RealESRGANUpscaler,
    target_res: tuple = (1920, 1080),
    batch_size: int = 1,
    num_workers: int = 4,
    max_frames: int = None
):
    """
    Upscale all image frames in a folder with multithreaded I/O or multi-process CPU workers.
    """
    image_paths = find_image_files(input_dir)
    if max_frames is not None and max_frames > 0:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        raise ValueError(f"No valid image files found in {input_dir}")

    total_images = len(image_paths)
    print("\n" + "=" * 60)
    print(f"🖼️ BATCH IMAGE FRAMES UPSCALER")
    print(f"   Input Directory:  {input_dir} ({total_images} images)")
    print(f"   Output Directory: {output_dir}")
    print(f"   Target Res:       {target_res[0]}x{target_res[1]} (1080p)" if target_res else "Target Res: Native Model Scale")
    print(f"   Model:            {upscaler.model_info['description']}")
    print(f"   Device:           {upscaler.device.type.upper()} (FP16: {upscaler.half})")
    print(f"   Batch Size:       {batch_size}")
    print(f"   Workers:          {num_workers}")
    print("=" * 60 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    input_dir_abs = os.path.abspath(input_dir)
    start_time = time.time()

    # If CPU and num_workers > 1, use ProcessPoolExecutor with spawn context for parallel CPU inference
    if upscaler.device.type == 'cpu' and num_workers > 1:
        tasks = []
        skipped_count = 0
        for path in image_paths:
            rel_path = os.path.relpath(path, input_dir_abs)
            out_file_path = os.path.join(output_dir, rel_path)
            if os.path.exists(out_file_path) and os.path.getsize(out_file_path) > 0:
                skipped_count += 1
            else:
                tasks.append((path, out_file_path, target_res))

        if skipped_count > 0:
            print(f"ℹ️ Skipping {skipped_count} already processed frames...")

        pbar = tqdm(total=total_images, initial=skipped_count, desc="Upscaling Frames", unit="frame")

        if tasks:
            import multiprocessing as mp
            model_name = 'compact'
            from upscaler import MODEL_REGISTRY
            for k, v in MODEL_REGISTRY.items():
                if v == upscaler.model_info:
                    model_name = k
                    break

            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(model_name, 'weights', 'cpu', upscaler.tile, upscaler.half)
            ) as pool:
                futures = [pool.submit(_process_frame_worker, t) for t in tasks]
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

        pbar.close()

    else:
        pbar = tqdm(total=total_images, desc="Upscaling Frames", unit="frame")

        with ThreadPoolExecutor(max_workers=num_workers) as io_pool:
            for i in range(0, total_images, batch_size):
                batch_paths = image_paths[i:i + batch_size]
                current_b = len(batch_paths)

                batch_imgs = []
                for path in batch_paths:
                    rel_path = os.path.relpath(path, input_dir_abs)
                    out_file_path = os.path.join(output_dir, rel_path)
                    if os.path.exists(out_file_path) and os.path.getsize(out_file_path) > 0:
                        continue
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    if img is not None:
                        batch_imgs.append((path, out_file_path, img))
                    else:
                        print(f"⚠️ Warning: Could not read {path}, skipping.")

                if not batch_imgs:
                    pbar.update(current_b)
                    continue

                if len(batch_imgs) == 1:
                    path, out_file_path, img = batch_imgs[0]
                    enhanced = upscaler.enhance_frame(img, target_size=target_res)
                    enhanced_list = [(out_file_path, enhanced)]
                else:
                    tensors = []
                    for _, _, f in batch_imgs:
                        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        tensors.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)

                    batch_t = torch.stack(tensors, dim=0)
                    out_t = upscaler.enhance_tensor(batch_t)

                    if target_res is not None:
                        target_w, target_h = target_res
                        if out_t.shape[3] != target_w or out_t.shape[2] != target_h:
                            out_t = F.interpolate(out_t, size=(target_h, target_w), mode='bicubic', align_corners=False)

                    out_np = out_t.permute(0, 2, 3, 1).float().clamp(0.0, 1.0).cpu().numpy()
                    enhanced_list = []
                    for idx, (path, out_file_path, _) in enumerate(batch_imgs):
                        out_bgr = cv2.cvtColor((out_np[idx] * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
                        enhanced_list.append((out_file_path, out_bgr))

                for out_file_path, enhanced_img in enhanced_list:
                    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
                    io_pool.submit(cv2.imwrite, out_file_path, enhanced_img)

                pbar.update(current_b)

        pbar.close()

    total_time = time.time() - start_time
    avg_fps = total_images / total_time if total_time > 0 else 0
    print("\n" + "=" * 60)
    print(f"✅ Frames Upscaling Finished!")
    print(f"   Images Processed: {total_images}")
    print(f"   Total Time:       {total_time:.2f}s ({avg_fps:.2f} FPS)")
    print(f"   Output Directory: {output_dir}")
    print("=" * 60 + "\n")

