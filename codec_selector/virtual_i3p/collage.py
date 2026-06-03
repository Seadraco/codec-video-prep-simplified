#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collage packing utilities for Virtual I+3P.

Pack selected patches into a canvas image, sorted by (frame_id, patch_y, patch_x).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import cv2

from codec_selector.codec_patch_gop.patch_utils import (
    pack_patches_to_canvases,
    extract_patch_rgb,
    iter_blocks_in_raster,
    block_to_patches,
)


def pack_p_collage(
    frame_dict: Dict[int, np.ndarray],
    selected_patches: List[Dict[str, Any]],
    hb: int,
    wb: int,
    patch_size: int,
    group_size: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack selected patches into a P-collage canvas.

    Args:
        frame_dict: Dict mapping frame_id -> RGB frame.
        selected_patches: List of {frame_id, patch_y, patch_x, score}.
        hb: Number of patches in height direction.
        wb: Number of patches in width direction.
        patch_size: Patch size in pixels.
        group_size: Group size (default 2 for 2x2).

    Returns:
        (images_rgb, patch_position, img_ptr) from pack_patches_to_canvases.
    """
    p = int(patch_size)
    S_full = hb * wb
    b = int(max(1, int(group_size)))
    block_patch_count = b * b
    pz = np.zeros((p, p, 3), dtype=np.uint8)

    # Sort patches by (frame_id, patch_y, patch_x)
    sorted_patches = sorted(
        selected_patches,
        key=lambda x: (int(x["frame_id"]), int(x["patch_y"]), int(x["patch_x"])),
    )

    # Group patches by their original group to maintain block structure
    # First, collect all unique groups
    group_keys: List[Tuple[int, int, int]] = []  # (frame_id, group_h, group_w)
    seen_groups: set = set()
    for patch in sorted_patches:
        fid = int(patch["frame_id"])
        gh = patch.get("group_h", patch["patch_y"] // b)
        gw = patch.get("group_w", patch["patch_x"] // b)
        key = (fid, gh, gw)
        if key not in seen_groups:
            seen_groups.add(key)
            group_keys.append(key)

    # Sort groups by (frame_id, group_h, group_w)
    group_keys.sort(key=lambda x: (int(x[0]), int(x[1]), int(x[2])))

    # Extract patches in block order
    patches_list: List[np.ndarray] = []
    src_pos_list: List[List[int]] = []

    for fid, gh, gw in group_keys:
        fr = frame_dict.get(int(fid))
        if fr is None or fr.size == 0:
            # Pad with zeros
            for _ in range(block_patch_count):
                patches_list.append(pz)
                src_pos_list.append([-1, -1, -1])
            continue

        for ph, pw in block_to_patches(gh, gw, block_size=b):
            if ph < hb and pw < wb:
                patch = extract_patch_rgb(fr, ph, pw, patch=p)
                patches_list.append(patch.astype(np.uint8))
                src_pos_list.append([int(fid), int(ph), int(pw)])
            else:
                patches_list.append(pz)
                src_pos_list.append([-1, -1, -1])

    # Pad to full canvas size if needed
    while len(patches_list) < S_full:
        patches_list.append(pz)
        src_pos_list.append([-1, -1, -1])

    # Truncate if too many (shouldn't happen with proper budget)
    patches_list = patches_list[:S_full]
    src_pos_list = src_pos_list[:S_full]

    if not patches_list:
        images = np.zeros((0, hb * p, wb * p, 3), dtype=np.uint8)
        patch_pos = np.zeros((0, 3), dtype=np.int32)
        img_ptr = np.zeros((1,), dtype=np.int32)
        return images, patch_pos, img_ptr

    patches_arr = np.stack(patches_list, axis=0).astype(np.uint8)
    images_rgb, patch_pos, img_ptr = pack_patches_to_canvases(
        patches_arr, hb=hb, wb=wb, patch=p, block_size=b
    )
    return images_rgb, patch_pos, img_ptr


def pack_anchor_fullframe(
    frame_rgb: np.ndarray,
    hb: int,
    wb: int,
    patch_size: int,
    group_size: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack a full-frame anchor into a canvas with all patches.

    Args:
        frame_rgb: Full RGB frame (already resized and padded).
        hb, wb: Patch grid size.
        patch_size: Patch size.
        group_size: Block size for packing.

    Returns:
        (images_rgb, patch_position, img_ptr)
    """
    p = int(patch_size)
    b = int(max(1, int(group_size)))
    patches_list: List[np.ndarray] = []
    src_pos_list: List[List[int]] = []

    # Iterate all blocks in raster order
    for bh, bw in iter_blocks_in_raster(hb, wb, block_size=b):
        for ph, pw in block_to_patches(bh, bw, block_size=b):
            patch = extract_patch_rgb(frame_rgb, ph, pw, patch=p)
            patches_list.append(patch.astype(np.uint8))
            src_pos_list.append([0, int(ph), int(pw)])

    patches_arr = np.stack(patches_list, axis=0).astype(np.uint8)
    images_rgb, patch_pos, img_ptr = pack_patches_to_canvases(
        patches_arr, hb=hb, wb=wb, patch=p, block_size=b
    )
    return images_rgb, patch_pos, img_ptr
