#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frame resize, padding, and normalization helpers."""

from typing import Any, Dict, List, Tuple

import numpy as np

from codec_selector.codec_patch_gop.frame_utils import _resize_bgr, pad_to_multiple_of_bgr
from codec_selector.codec_patch_gop.scoring import patch_scores_to_block_scores, score_map_to_patch_scores
from codec_selector.codec_patch_gop.utils import smart_resize


def resize_within_max_dim(height: int, width: int, max_dim: int = 616, factor: int = 28) -> Tuple[int, int]:
    h, w = int(height), int(width)
    f = int(max(1, factor))
    max_d = int(max_dim)
    if h <= 0 or w <= 0:
        return f, f
    scale = min(1.0, float(max_d) / float(max(h, w)))
    h_new = int(round(h * scale / f) * f)
    w_new = int(round(w * scale / f) * f)
    h_new = max(f, min(h_new, max_d))
    w_new = max(f, min(w_new, max_d))
    return max(f, int(h_new // f * f)), max(f, int(w_new // f * f))


def resolve_prepared_frame_geometry(
    height: int,
    width: int,
    patch: int,
    max_dim: int,
    block_size: int = 2,
    max_pixels: int = 150000,
    no_resize: bool = False,
) -> Tuple[int, int, int, int, int, int]:
    p = int(patch)
    b = int(max(1, int(block_size)))
    factor = b * p
    h0, w0 = int(height), int(width)
    if bool(no_resize):
        resize_h, resize_w = int(h0), int(w0)
    elif int(max_pixels) > 0:
        resize_h, resize_w = smart_resize(h0, w0, factor=factor, max_pixels=int(max_pixels))
    else:
        resize_h, resize_w = resize_within_max_dim(h0, w0, max_dim=max_dim, factor=factor)
    pad_bottom = (factor - (int(resize_h) % factor)) % factor
    pad_right = (factor - (int(resize_w) % factor)) % factor
    out_h = int(resize_h) + int(pad_bottom)
    out_w = int(resize_w) + int(pad_right)
    return int(resize_h), int(resize_w), int(pad_bottom), int(pad_right), int(out_h), int(out_w)


def prepare_frames(
    frames_bgr: List[np.ndarray],
    patch: int,
    max_dim: int,
    block_size: int = 2,
    max_pixels: int = 150000,
    no_resize: bool = False,
) -> Tuple[List[np.ndarray], int, int, int, int]:
    if not frames_bgr:
        raise RuntimeError("prepare_frames received no frames")

    p = int(patch)
    b = int(max(1, int(block_size)))
    factor = b * p
    h0, w0 = frames_bgr[0].shape[:2]
    resize_h, resize_w, _, _, _, _ = resolve_prepared_frame_geometry(
        height=int(h0),
        width=int(w0),
        patch=int(patch),
        max_dim=int(max_dim),
        block_size=int(block_size),
        max_pixels=int(max_pixels),
        no_resize=bool(no_resize),
    )

    resized = [_resize_bgr(fr, int(resize_h), int(resize_w)) for fr in frames_bgr]
    out: List[np.ndarray] = []
    pad_bottom = pad_right = 0
    for i, fr in enumerate(resized):
        frp, (pb, pr) = pad_to_multiple_of_bgr(fr, factor)
        out.append(frp)
        if i == 0:
            pad_bottom, pad_right = int(pb), int(pr)
    return out, int(resize_h), int(resize_w), int(pad_bottom), int(pad_right)


def normalize_score_maps_frame_mean_floor(
    score_maps: List[np.ndarray],
    patch: int,
    block_size: int = 2,
    floor_ratio: float = 0.5,
    allow_boost: bool = False,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    if not score_maps:
        return [], {
            "frame_block_mean_ref": 0.0,
            "frame_block_mean_floor": 0.0,
            "frame_block_mean_min": 0.0,
            "frame_block_mean_max": 0.0,
            "frame_scale_min": 1.0,
            "frame_scale_max": 1.0,
            "frame_scale_mean": 1.0,
            "frame_count": 0,
        }

    frame_block_means: List[float] = []
    for score in score_maps:
        patch_scores = score_map_to_patch_scores(np.asarray(score, dtype=np.float32), patch=int(patch))
        block_scores = patch_scores_to_block_scores(patch_scores, block_size=int(block_size))
        frame_block_means.append(float(block_scores.mean()) if block_scores.size > 0 else 0.0)

    ref = float(np.median(np.asarray(frame_block_means, dtype=np.float32)))
    floor = float(max(1e-6, float(floor_ratio) * ref))
    out: List[np.ndarray] = []
    scales: List[float] = []
    for score, frame_mean in zip(score_maps, frame_block_means):
        denom = float(max(float(frame_mean), floor))
        scale = float(ref / max(denom, 1e-6))
        if not bool(allow_boost):
            scale = float(min(scale, 1.0))
        scales.append(scale)
        out.append((np.asarray(score, dtype=np.float32) * float(scale)).astype(np.float32))

    return out, {
        "frame_block_mean_ref": float(ref),
        "frame_block_mean_floor": float(floor),
        "frame_block_mean_min": float(min(frame_block_means)),
        "frame_block_mean_max": float(max(frame_block_means)),
        "frame_scale_min": float(min(scales)),
        "frame_scale_max": float(max(scales)),
        "frame_scale_mean": float(np.mean(np.asarray(scales, dtype=np.float32))),
        "frame_count": int(len(frame_block_means)),
    }
