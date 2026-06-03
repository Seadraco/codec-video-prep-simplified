#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anchor frame selection for virtual segments.

Supports two modes:
  - midpoint: select the middle frame of the segment.
  - quality: sample candidate frames and pick the one with best quality score.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2

from codec_selector.codec_patch_gop.frame_utils import frame_is_bad


def compute_anchor_quality_score(frame_bgr: np.ndarray) -> float:
    """Compute a simple quality score for anchor selection.

    score = edge_density - blur_penalty - black_or_white_penalty

    Higher is better.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return -1e9

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Edge density: Sobel gradient magnitude mean
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_density = float(cv2.magnitude(gx, gy).mean())

    # Blur penalty: Laplacian variance (lower = more blurry)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    # Normalize: typical range 50-1000. Penalize if below 100.
    blur_penalty = max(0.0, 100.0 - lap_var) * 0.5

    # Black/white penalty: penalize extreme brightness
    mean_val = float(gray.mean())
    black_penalty = max(0.0, 30.0 - mean_val) * 2.0
    white_penalty = max(0.0, mean_val - 225.0) * 2.0

    score = edge_density - blur_penalty - black_penalty - white_penalty
    return float(score)


def select_anchor_frame(
    video_path: str,
    segment_frame_ids: List[int],
    mode: str = "midpoint",
    num_candidates: int = 16,
    backend: str = "ffmpeg_native",
) -> Tuple[int, Optional[np.ndarray]]:
    """Select the best anchor frame for a virtual segment.

    Args:
        video_path: Path to video file.
        segment_frame_ids: List of frame IDs belonging to this segment.
        mode: "midpoint" or "quality".
        num_candidates: Number of candidate frames to sample for quality mode.
        backend: Frame decode backend.

    Returns:
        (selected_frame_id, frame_bgr) or (selected_frame_id, None) if decode fails.
    """
    if not segment_frame_ids:
        return -1, None

    mode = str(mode).lower().strip()

    if mode == "midpoint":
        mid_idx = len(segment_frame_ids) // 2
        fid = int(segment_frame_ids[mid_idx])
        from codec_selector.codec_patch_gop.frame_utils import decode_frame_bgr_at
        frame = decode_frame_bgr_at(str(video_path), int(fid))
        return int(fid), frame

    # Quality mode: sample candidates and pick best
    n = len(segment_frame_ids)
    if n <= num_candidates:
        candidate_indices = list(range(n))
    else:
        # Uniform sampling across the segment
        step = max(1, n // num_candidates)
        candidate_indices = list(range(0, n, step))[:num_candidates]
        # Ensure last frame is included
        if candidate_indices[-1] != n - 1 and len(candidate_indices) < num_candidates:
            candidate_indices.append(n - 1)

    candidate_fids = [int(segment_frame_ids[i]) for i in candidate_indices]

    from codec_selector.codec_patch_gop.frame_utils import decode_frames_bgr
    frames = decode_frames_bgr(str(video_path), candidate_fids, backend=str(backend))

    best_score = -1e9
    best_fid = candidate_fids[0] if candidate_fids else -1
    best_frame = None

    for fid, frame in zip(candidate_fids, frames):
        if frame is None or frame.size == 0:
            continue
        if frame_is_bad(frame):
            continue
        score = compute_anchor_quality_score(frame)
        if score > best_score:
            best_score = score
            best_fid = int(fid)
            best_frame = frame

    return int(best_fid), best_frame
