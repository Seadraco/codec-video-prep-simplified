#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug visualization for Virtual I+3P pipeline — generates side-by-side video.

Left: original sampled frame
Right: selected patches only (unselected patches blacked out)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import subprocess
import numpy as np
import cv2


def _create_patch_mask_image(
    frame_bgr: np.ndarray,
    selected_patches: List[Dict[str, Any]],
    patch_size: int,
) -> np.ndarray:
    """Create an image where only selected patches are visible, rest is black.

    Args:
        frame_bgr: Original BGR frame.
        selected_patches: List of {patch_y, patch_x, ...} for this frame.
        patch_size: Patch size in pixels.

    Returns:
        BGR image with only selected patches visible.
    """
    p = int(patch_size)
    h, w = frame_bgr.shape[:2]
    masked = np.zeros_like(frame_bgr)

    for patch in selected_patches:
        py = int(patch["patch_y"])
        px = int(patch["patch_x"])
        y0 = py * p
        x0 = px * p
        y1 = min(y0 + p, h)
        x1 = min(x0 + p, w)
        if y0 < h and x0 < w:
            masked[y0:y1, x0:x1] = frame_bgr[y0:y1, x0:x1]

    return masked


def _resize_to_same_height(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resize two images to the same height while preserving aspect ratio."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h = max(h1, h2)

    if h1 != target_h:
        scale = target_h / h1
        new_w = int(w1 * scale)
        img1 = cv2.resize(img1, (new_w, target_h), interpolation=cv2.INTER_AREA)

    if h2 != target_h:
        scale = target_h / h2
        new_w = int(w2 * scale)
        img2 = cv2.resize(img2, (new_w, target_h), interpolation=cv2.INTER_AREA)

    return img1, img2


def generate_segment_debug_video(
    frames_bgr_dict: Dict[int, np.ndarray],
    frame_to_patches: Dict[int, List[Dict[str, Any]]],
    patch_size: int,
    segment_idx: int,
    debug_dir: Path,
    fps: float = 4.0,
) -> str:
    """Generate a side-by-side debug video for one segment.

    Left: original frame    Right: selected patches only (rest black)

    Args:
        frames_bgr_dict: Dict mapping frame_id -> BGR frame.
        frame_to_patches: Dict mapping frame_id -> list of selected patches.
        patch_size: Patch size.
        segment_idx: Segment index for filename.
        debug_dir: Output directory.
        fps: Video FPS.

    Returns:
        Path to generated video file.
    """
    if not frames_bgr_dict:
        return ""

    # Sort frame_ids
    sorted_fids = sorted(frames_bgr_dict.keys())

    # Prepare frames
    side_by_side_frames: List[np.ndarray] = []
    for fid in sorted_fids:
        fr = frames_bgr_dict[fid]
        patches = frame_to_patches.get(fid, [])

        # Right side: masked image
        masked = _create_patch_mask_image(fr, patches, patch_size)

        # Resize to same height and concatenate
        left, right = _resize_to_same_height(fr, masked)
        combined = np.hstack([left, right])

        # Add text labels
        h, w = combined.shape[:2]
        cv2.putText(combined, f"fid={fid}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, f"patches={len(patches)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        side_by_side_frames.append(combined)

    if not side_by_side_frames:
        return ""

    # Determine output size from first frame
    h_out, w_out = side_by_side_frames[0].shape[:2]

    # Write video using ffmpeg subprocess for reliability
    out_path = debug_dir / f"segment_{segment_idx:03d}_debug.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w_out}x{h_out}",
        "-r", f"{float(fps):.2f}",
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        str(out_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for fr in side_by_side_frames:
            proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        if proc.stdout is not None:
            proc.stdout.read()
        proc.wait()
    except Exception:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        proc.kill()
        proc.wait()
        raise

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="ignore") if stderr is not None else ""
        raise RuntimeError(f"ffmpeg encode failed for {out_path}: {err}")

    return str(out_path)


def write_debug_summary(
    summary_records: List[Dict[str, Any]],
    debug_dir: Path,
) -> str:
    """Write debug summary CSV."""
    fp = debug_dir / "summary.csv"
    header = "segment_idx,p_window_idx,num_selected_groups,num_selected_patches,min_frame_id,max_frame_id,mean_score,max_score\n"
    with open(fp, "w", encoding="utf-8") as f:
        f.write(header)
        for rec in summary_records:
            f.write(
                f"{rec['segment_idx']},"
                f"{rec['p_window_idx']},"
                f"{rec['num_selected_groups']},"
                f"{rec['num_selected_patches']},"
                f"{rec['min_frame_id']},"
                f"{rec['max_frame_id']},"
                f"{rec['mean_score']:.6f},"
                f"{rec['max_score']:.6f}\n"
            )
    return str(fp)
