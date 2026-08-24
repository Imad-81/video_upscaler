import os
import math
import requests
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from models import SRVGGNetCompact, RRDBNet

# Model registry with URLs and architectures
MODEL_REGISTRY = {
    'compact': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        'scale': 4,
        'arch': 'SRVGGNetCompact',
        'params': dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
        'description': 'Fast compact model for real-world videos and photos'
    },
    'anime': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
        'scale': 4,
        'arch': 'SRVGGNetCompact',
        'params': dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'),
        'description': 'Optimized for animation, anime, and smooth graphics'
    },
    'x4plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'scale': 4,
        'arch': 'RRDBNet',
        'params': dict(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32),
        'description': 'High-detail deep photo reconstruction model'
    },
    'x2plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        'scale': 2,
        'arch': 'RRDBNet',
        'params': dict(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32),
        'description': 'Native 2x RRDBNet model'
    }
}

# Synonyms for convenience
MODEL_ALIASES = {
    'realesr-general-x4v3': 'compact',
    'realesr-general-wdn-x4v3': 'compact',
    'realesr-animevideov3': 'anime',
    'RealESRGAN_x4plus': 'x4plus',
    'RealESRGAN_x2plus': 'x2plus',
    'realesrgan-x4plus': 'x4plus',
    'realesrgan-x2plus': 'x2plus',
}


def download_weight(url: str, save_path: str):
    """Download model weight file with a progress bar."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return save_path

    print(f"Downloading model weights from {url}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    temp_path = save_path + '.download'
    with open(temp_path, 'wb') as f, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    os.rename(temp_path, save_path)
    return save_path


class RealESRGANUpscaler:
    def __init__(
        self,
        model_name: str = 'compact',
        weights_dir: str = 'weights',
        device: str = 'auto',
        tile: int = 512,
        tile_pad: int = 10,
        half: bool = True
    ):
        model_key = MODEL_ALIASES.get(model_name, model_name)
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}")

        self.model_info = MODEL_REGISTRY[model_key]
        self.scale = self.model_info['scale']
        self.tile = tile
        self.tile_pad = tile_pad

        # Setup compute device
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Half precision (FP16) setup
        # Note: On CPU, keep fp32. On MPS/CUDA, half precision works well for speed.
        if self.device.type == 'cpu':
            self.half = False
        else:
            self.half = half

        # Initialize network architecture
        arch_cls = SRVGGNetCompact if self.model_info['arch'] == 'SRVGGNetCompact' else RRDBNet
        self.model = arch_cls(**self.model_info['params'])

        # Download & load weights
        weight_filename = os.path.basename(self.model_info['url'])
        weight_path = os.path.join(weights_dir, weight_filename)
        download_weight(self.model_info['url'], weight_path)

        loadnet = torch.load(weight_path, map_location='cpu', weights_only=True)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        elif 'params' in loadnet:
            keyname = 'params'
        else:
            keyname = None

        if keyname is not None:
            self.model.load_state_dict(loadnet[keyname], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)

        self.model.eval()
        self.model.to(self.device)
        if self.half:
            self.model = self.model.half()

    @torch.no_grad()
    def enhance_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Enhance a PyTorch tensor (B, C, H, W) normalized to [0, 1].
        """
        if self.half:
            img_tensor = img_tensor.half()
        else:
            img_tensor = img_tensor.float()

        img_tensor = img_tensor.to(self.device)

        # Whole image inference if tile == 0 or tile covers full image
        b, c, h, w = img_tensor.shape
        if self.tile <= 0 or (self.tile >= h and self.tile >= w):
            output = self.model(img_tensor)
            return output

        # Tiled inference for large inputs to prevent OOM
        tile = self.tile
        tile_pad = self.tile_pad
        scale = self.scale

        tiles_x = math.ceil(w / tile)
        tiles_y = math.ceil(h / tile)

        output_shape = (b, c, h * scale, w * scale)
        output = torch.zeros(output_shape, dtype=img_tensor.dtype, device=self.device)

        for y in range(tiles_y):
            for x in range(tiles_x):
                # Calculate tile bounds
                ofs_x = x * tile
                ofs_y = y * tile

                # Input tile start and end with padding
                input_start_x = max(ofs_x - tile_pad, 0)
                input_end_x = min(ofs_x + tile + tile_pad, w)
                input_start_y = max(ofs_y - tile_pad, 0)
                input_end_y = min(ofs_y + tile + tile_pad, h)

                input_tile = img_tensor[:, :, input_start_y:input_end_y, input_start_x:input_end_x]

                # Run tile through model
                output_tile = self.model(input_tile)

                # Output tile coordinates
                out_start_x = ofs_x * scale
                out_end_x = min((ofs_x + tile) * scale, w * scale)
                out_start_y = ofs_y * scale
                out_end_y = min((ofs_y + tile) * scale, h * scale)

                # Crop coordinates from the tile output
                pad_left = (ofs_x - input_start_x) * scale
                pad_top = (ofs_y - input_start_y) * scale
                tile_out_w = (out_end_x - out_start_x)
                tile_out_h = (out_end_y - out_start_y)

                output[:, :, out_start_y:out_end_y, out_start_x:out_end_x] = output_tile[
                    :, :, pad_top:pad_top + tile_out_h, pad_left:pad_left + tile_out_w
                ]

        return output

    def enhance_frame(self, img_np: np.ndarray, target_size: tuple = None) -> np.ndarray:
        """
        Enhance a single BGR numpy image frame (H, W, 3) uint8.
        Optional target_size=(target_width, target_height).
        """
        # Convert BGR -> RGB and normalize to [0, 1]
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Enhance
        out_t = self.enhance_tensor(img_t)

        # Convert back to numpy (0..255 uint8 BGR)
        out_np = out_t.squeeze(0).permute(1, 2, 0).float().clamp(0.0, 1.0).cpu().numpy()
        out_bgr = cv2.cvtColor((out_np * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

        # If target size is specified (e.g. 1920x1080), downsample/resize smoothly
        if target_size is not None:
            target_w, target_h = target_size
            if out_bgr.shape[1] != target_w or out_bgr.shape[0] != target_h:
                out_bgr = cv2.resize(out_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        return out_bgr
