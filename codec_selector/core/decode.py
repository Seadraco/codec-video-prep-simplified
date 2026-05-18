#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ffmpeg-only frame decoding helpers."""

import os
import subprocess
from typing import Dict, List, Optional

import numpy as np

from .config import VideoMetadata


def _decode_unique_frame_batch(
    video_path: str,
    uniq_ids: List[int],
    height: int,
    width: int,
    decode_h: int,
    decode_w: int,
) -> Dict[int, np.ndarray]:
    select_expr = "+".join(f"eq(n\\,{fid})" for fid in uniq_ids)
    filters = [f"select={select_expr}"]
    if int(decode_h) != int(height) or int(decode_w) != int(width):
        filters.append(f"scale={int(decode_w)}:{int(decode_h)}:flags=area")
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        ",".join(filters),
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg decode failed: {err.strip()}")

    frame_size = int(decode_w) * int(decode_h) * 3
    raw = proc.stdout
    num_decoded = len(raw) // frame_size
    if num_decoded <= 0:
        raise RuntimeError(f"ffmpeg returned no frames for {video_path}")
    if len(raw) % frame_size != 0:
        raw = raw[: num_decoded * frame_size]

    decoded: Dict[int, np.ndarray] = {}
    for i in range(min(num_decoded, len(uniq_ids))):
        start = i * frame_size
        end = start + frame_size
        decoded[int(uniq_ids[i])] = np.frombuffer(raw[start:end], dtype=np.uint8).copy().reshape(decode_h, decode_w, 3)
    return decoded


def decode_frames_bgr_ffmpeg(
    video_path: str,
    frame_ids: List[int],
    meta: VideoMetadata,
    out_h: Optional[int] = None,
    out_w: Optional[int] = None,
) -> List[np.ndarray]:
    frame_ids = [int(x) for x in frame_ids]
    if not frame_ids:
        return []

    width = int(meta.width)
    height = int(meta.height)
    total = int(meta.total_frames)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video size for {video_path}: {width}x{height}")

    uniq_ids = sorted(set(max(0, min(int(fid), total - 1)) if total > 0 else max(0, int(fid)) for fid in frame_ids))
    if not uniq_ids:
        return []

    decode_h = int(out_h) if out_h is not None and int(out_h) > 0 else int(height)
    decode_w = int(out_w) if out_w is not None and int(out_w) > 0 else int(width)
    batch_size = int(os.environ.get("CV_PREINFER_FFMPEG_SELECT_BATCH", "64"))
    batch_size = max(1, int(batch_size))
    decoded_map: Dict[int, np.ndarray] = {}
    for start in range(0, len(uniq_ids), batch_size):
        batch_ids = uniq_ids[start:start + batch_size]
        decoded_map.update(
            _decode_unique_frame_batch(
                video_path=str(video_path),
                uniq_ids=batch_ids,
                height=int(height),
                width=int(width),
                decode_h=int(decode_h),
                decode_w=int(decode_w),
            )
        )

    out: List[np.ndarray] = []
    last_good: Optional[np.ndarray] = None
    for fid in frame_ids:
        key = max(0, min(int(fid), total - 1)) if total > 0 else max(0, int(fid))
        fr = decoded_map.get(key)
        if fr is None:
            if last_good is None:
                nearest = decoded_map[min(decoded_map.keys(), key=lambda x: abs(x - key))] if decoded_map else None
                if nearest is None:
                    raise RuntimeError(f"missing decoded frame for fid={fid} in {video_path}")
                fr = nearest
            else:
                fr = last_good
        out.append(np.ascontiguousarray(fr))
        last_good = fr
    return out
