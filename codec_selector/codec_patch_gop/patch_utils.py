#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch packing and canvas saving utilities."""

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2

# Pillow for reliable JPEG writing
try:
    from PIL import Image  # type: ignore
    HAS_PIL = True
except Exception:
    Image = None
    HAS_PIL = False


def iter_blocks_in_raster(hb: int, wb: int, block_size: int = 2):
    """Iterate square blocks in raster order of blocks.
    
    hb/wb are patch-grid sizes and must be divisible by block_size.
    Yields (bh, bw) in block-grid.
    """
    b = int(max(1, int(block_size)))
    for bh in range(hb // b):
        for bw in range(wb // b):
            yield bh, bw


def block_to_patches(bh: int, bw: int, block_size: int = 2) -> List[Tuple[int, int]]:
    """Convert block coord to patch coords in the required contiguous order.
    
    For block_size=2 returns [(h0,w0),(h0,w0+1),(h0+1,w0),(h0+1,w0+1)].
    """
    b = int(max(1, int(block_size)))
    h0 = b * int(bh)
    w0 = b * int(bw)
    return [(h0 + dh, w0 + dw) for dh in range(b) for dw in range(b)]


def block_to_4_patches(bh: int, bw: int) -> List[Tuple[int, int]]:
    """Convert 2x2 block coord to 4 patch coords."""
    return block_to_patches(bh, bw, block_size=2)


def extract_patch_rgb(frame_rgb: np.ndarray, ph: int, pw: int, patch: int = 16) -> np.ndarray:
    """Extract a single patch from RGB frame."""
    p = int(patch)
    y0 = int(ph) * p
    x0 = int(pw) * p
    return frame_rgb[y0:y0 + p, x0:x0 + p, :]


def pack_patches_to_canvases(
    patches: np.ndarray,
    hb: int,
    wb: int,
    patch: int,
    placement_order: str = "block_raster",
    block_size: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack patches into one or more full canvases.

    Packing order is raster over square blocks, and within each block patches
    are placed in row-major order. This guarantees that `image.reshape(-1)` has
    consecutive block_size*block_size tokens corresponding to one block.

    Args:
        patches: uint8 array (N, patch, patch, 3) where N is multiple of hb*wb
        hb: Number of patches in height direction
        wb: Number of patches in width direction
        patch: Patch size in pixels
        placement_order: "block_raster" (default) or "wh_raster"

    Returns:
      images_rgb: uint8 (num_images, H, W, 3)
      patch_position: int32 (N, 3) [img_idx, patch_h, patch_w] aligned 1-1 with patches
      img_ptr: int32 (num_images+1,) prefix-sum boundaries (each image has hb*wb patches)
    """
    hb = int(hb)
    wb = int(wb)
    p = int(patch)
    b = int(max(1, int(block_size)))
    S_full = hb * wb
    placement_order = str(placement_order).lower().strip()
    if placement_order != "wh_raster" and (hb % b != 0 or wb % b != 0):
        raise ValueError(f"hb/wb must be divisible by block_size={b}, got hb={hb} wb={wb}")
    
    if patches.size == 0:
        images = np.zeros((0, hb * p, wb * p, 3), dtype=np.uint8)
        patch_pos = np.zeros((0, 3), dtype=np.int32)
        img_ptr = np.zeros((1,), dtype=np.int32)
        return images, patch_pos, img_ptr

    assert patches.ndim == 4 and patches.shape[1] == p and patches.shape[2] == p and patches.shape[3] == 3
    assert patches.shape[0] % S_full == 0, f"patches must be multiple of S_full={S_full}, got {patches.shape[0]}"

    num_images = int(patches.shape[0] // S_full)
    H = hb * p
    W = wb * p
    images = np.zeros((num_images, H, W, 3), dtype=np.uint8)
    patch_pos = np.zeros((patches.shape[0], 3), dtype=np.int32)

    idx = 0
    for img_i in range(num_images):
        if placement_order == "wh_raster":
            # Raster order by patch position (h, w)
            for ph in range(hb):
                for pw in range(wb):
                    y0 = int(ph) * p
                    x0 = int(pw) * p
                    images[img_i, y0:y0 + p, x0:x0 + p, :] = patches[idx]
                    patch_pos[idx, :] = (int(img_i), int(ph), int(pw))
                    idx += 1
        else:
            # Default: block raster order.
            for bh in range(hb // b):
                for bw in range(wb // b):
                    coords = block_to_patches(bh, bw, block_size=b)
                    for (ph, pw) in coords:
                        y0 = int(ph) * p
                        x0 = int(pw) * p
                        images[img_i, y0:y0 + p, x0:x0 + p, :] = patches[idx]
                        patch_pos[idx, :] = (int(img_i), int(ph), int(pw))
                        idx += 1

    img_ptr = (np.arange(0, num_images + 1, dtype=np.int32) * int(S_full)).astype(np.int32)
    return images, patch_pos, img_ptr


def save_canvases_as_jpg(images_rgb: np.ndarray, out_dir: str, quality: int = 95) -> List[str]:
    """Save (num_images, H, W, 3) RGB uint8 canvases into JPEG files.

    Returns list of written filenames (basenames).
    """
    out: List[str] = []
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if images_rgb is None or images_rgb.size == 0:
        return out

    q = int(max(1, min(100, int(quality))))
    for i in range(int(images_rgb.shape[0])):
        fn = f"canvas_{i:03d}.jpg"
        fp = out_p / fn
        arr = images_rgb[i]

        # Prefer Pillow for JPEG reliability; fallback to OpenCV.
        if HAS_PIL and Image is not None:
            Image.fromarray(arr).save(str(fp), format="JPEG", quality=q, subsampling=0, optimize=True)
        else:
            bgr = arr[:, :, ::-1]
            ok = cv2.imwrite(str(fp), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                raise RuntimeError("Failed to write JPEG. Please install pillow: pip install pillow")

        out.append(fn)

    return out
