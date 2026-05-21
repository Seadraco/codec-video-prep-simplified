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


def extract_patch_bgr(frame_bgr: np.ndarray, ph: int, pw: int, patch: int = 16) -> np.ndarray:
    """Extract a single patch from BGR frame and convert to RGB."""
    p = int(patch)
    y0 = int(ph) * p
    x0 = int(pw) * p
    return frame_bgr[y0:y0 + p, x0:x0 + p, ::-1]


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

    if placement_order == "wh_raster":
        images = patches.reshape(num_images, hb, wb, p, p, 3) \
                        .transpose(0, 1, 3, 2, 4, 5) \
                        .reshape(num_images, H, W, 3)
        ph = np.repeat(np.arange(hb, dtype=np.int32), wb)
        pw = np.tile(np.arange(wb, dtype=np.int32), hb)
    else:
        # block_raster: (num_images, S_full, p, p, 3)
        # -> (num_images, hb//b, wb//b, b, b, p, p, 3)
        # -> (num_images, hb//b, b, wb//b, b, p, p, 3)
        # -> (num_images, hb, wb, p, p, 3)
        # -> (num_images, hb, p, wb, p, 3)
        # -> (num_images, H, W, 3)
        images = patches.reshape(num_images, hb // b, wb // b, b, b, p, p, 3) \
                        .transpose(0, 1, 3, 2, 4, 5, 6, 7) \
                        .reshape(num_images, hb, wb, p, p, 3) \
                        .transpose(0, 1, 3, 2, 4, 5) \
                        .reshape(num_images, H, W, 3)

        # Precompute ph, pw for block_raster order
        ph = np.zeros(S_full, dtype=np.int32)
        pw = np.zeros(S_full, dtype=np.int32)
        idx_c = 0
        for bh in range(hb // b):
            for bw in range(wb // b):
                for dh in range(b):
                    for dw in range(b):
                        ph[idx_c] = bh * b + dh
                        pw[idx_c] = bw * b + dw
                        idx_c += 1

    patch_pos = np.zeros((patches.shape[0], 3), dtype=np.int32)
    patch_pos[:, 0] = np.repeat(np.arange(num_images, dtype=np.int32), S_full)
    patch_pos[:, 1] = np.tile(ph, num_images)
    patch_pos[:, 2] = np.tile(pw, num_images)
    img_ptr = np.arange(0, num_images + 1, dtype=np.int32) * int(S_full)

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
