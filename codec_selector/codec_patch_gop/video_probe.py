#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Video probing utilities using ffprobe and OpenCV."""

import json
import math
import subprocess
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2


def get_total_frames_fps(video_path: str) -> Tuple[int, float, int, int]:
    """Get video metadata using OpenCV.
    
    Returns: (total_frames, fps, height, width)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0.0, 0, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not np.isfinite(fps):
        fps = 0.0
    return max(0, total), max(0.0, fps), h, w


def ffprobe_video_codec_name(video_path: str) -> str:
    """Best-effort fetch codec_name of the first video stream via ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return ""
        return str((p.stdout or "").strip()).lower()
    except Exception:
        return ""


def auto_max_total_patches(
    S_full: int,
    total_frames: int,
    fps: float,
    cap_total: int = 30000,
) -> Tuple[int, Dict[str, Any]]:
    """Choose max_total_patches (multiple of S_full) under cap_total.

    Heuristic:
      - duration <= 6s  : short
      - duration >  6s  : long

    Resolution tier via S_full (patches per full canvas):
      ~1080p: ~8160, ~720p: ~3680, <=480p: <=~1500

    Returns: (max_total_patches, debug_dict)
    """
    S_full = int(max(1, S_full))
    cap_total = int(max(S_full, cap_total))
    fps_use = float(fps) if (fps and fps > 0) else 30.0
    duration_sec = float(total_frames) / float(fps_use) if total_frames > 0 else 0.0
    tier_time = "short" if duration_sec <= 6.0 else "long"

    # Soft time cap (stricter, user-tuned)
    if duration_sec <= 7.0:
        max_images_by_time = 8
        time_cap_mode = "<=7s->8"
    elif duration_sec <= 12.0:
        max_images_by_time = 12
        time_cap_mode = "<=12s->12"
    else:
        max_images_by_time = max(1, int(math.ceil(duration_sec * 1.2)))
        time_cap_mode = ">12s->ceil(1.2x)"

    # Resolution tier via S_full
    if S_full >= 7000:  # ~1080p
        target_num_images = 4 if tier_time == "short" else 8
        tier_res = "1080p_like"
    elif S_full >= 2500:  # ~720p
        target_num_images = 8 if tier_time == "short" else 12
        tier_res = "720p_like"
    else:  # small (320p/480p)
        target_num_images = 12 if tier_time == "short" else 28
        tier_res = "small_like"

    num_images_cap = max(1, cap_total // S_full)
    num_images_final = int(min(target_num_images, num_images_cap, max_images_by_time))

    # Ensure output image count is a multiple of 4
    num_images_final = max(4, int(num_images_final // 4) * 4)

    max_total = int(num_images_final * S_full)
    dbg = {
        "cap_total": int(cap_total),
        "S_full": int(S_full),
        "fps_use": float(fps_use),
        "duration_sec": float(duration_sec),
        "time_cap_mode": str(time_cap_mode),
        "tier_time": tier_time,
        "tier_res": tier_res,
        "target_num_images": int(target_num_images),
        "num_images_cap": int(num_images_cap),
        "max_images_by_time": int(max_images_by_time),
        "num_images_final": int(num_images_final),
        "max_total_patches": int(max_total),
    }
    return max_total, dbg


def ffprobe_keyframe_frame_ids(video_path: str, fps: float, total_frames: int) -> List[int]:
    """Return keyframe frame ids (best-effort) using ffprobe.

    Parse keyframe timestamps and convert to frame ids by round(ts * fps).
    Prefer ``-skip_frame nokey`` so ffprobe only emits keyframes; the old
    all-frame probe is kept as a fallback for compatibility with odd streams.
    """
    fps_use = float(fps) if (fps and fps > 0) else 30.0

    def finish(ids: List[int]) -> List[int]:
        out = sorted(set(int(x) for x in ids))
        if not out:
            return [0] if total_frames > 0 else []
        if out[0] != 0:
            out = [0] + out
        return out

    def fid_from_ts(ts: float) -> int:
        fid = int(round(float(ts) * fps_use))
        if total_frames > 0:
            fid = max(0, min(int(total_frames) - 1, fid))
        return int(fid)

    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-skip_frame", "nokey",
            "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time",
            "-of", "csv=p=0",
            str(video_path),
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode == 0:
            out: List[int] = []
            for line in p.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [x.strip() for x in line.split(",") if x.strip()]
                if parts and parts[0].lower() == "frame":
                    parts = parts[1:]
                if not parts:
                    continue
                try:
                    ts = float(parts[0])
                except Exception:
                    continue
                if np.isfinite(ts):
                    out.append(fid_from_ts(ts))
            if out:
                return finish(out)
    except Exception:
        pass

    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=key_frame,best_effort_timestamp_time",
            "-of", "csv=p=0",
            str(video_path),
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return [0] if total_frames > 0 else []

        out: List[int] = []
        for line in p.stdout.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = [x.strip() for x in line.split(",") if x.strip() != ""]
            if not parts:
                continue
            if parts[0].lower() == "frame":
                parts = parts[1:]
            if len(parts) < 2:
                continue

            try:
                kf = int(float(parts[0]))
            except Exception:
                continue
            if kf != 1:
                continue

            try:
                ts = float(parts[1])
            except Exception:
                continue
            if not np.isfinite(ts):
                continue

            out.append(fid_from_ts(ts))

        return finish(out)
    except Exception:
        return [0] if total_frames > 0 else []


def ffprobe_sum_pkt_size(video_path: str, start_sec: float, dur_sec: float) -> int:
    """Sum packet sizes in [start_sec, start_sec + dur_sec] using ffprobe read_intervals."""
    try:
        start_sec = max(0.0, float(start_sec))
        dur_sec = max(0.0, float(dur_sec))
        interval = f"{start_sec:.6f}%+{dur_sec:.6f}"
        cmd = [
            "ffprobe",
            "-v", "error",
            "-read_intervals", interval,
            "-select_streams", "v:0",
            "-show_packets",
            "-show_entries", "packet=size",
            "-of", "csv=p=0",
            str(video_path),
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return 0
        s = 0
        for line in p.stdout.splitlines():
            line = line.strip()
            if not line or line == "N/A":
                continue
            try:
                s += int(line)
            except Exception:
                continue
        return int(s)
    except Exception:
        return 0


def ffprobe_packets_pb_energy_bins(
    video_path: str,
    bin_sec: float = 0.5,
    smooth_bins: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (bin_centers_sec, bin_energy, dbg) using packet sizes of NON-keyframes.

    Uses ffprobe -show_packets and reads packet pts_time/size/flags.
    - Exclude keyframes (flags contains 'K') so I-frames don't dominate.
    - Energy uses log1p(size) and is aggregated per time bin.
    - Optional smoothing over bins.
    """
    dbg: Dict[str, Any] = {
        "bin_sec": float(bin_sec),
        "smooth_bins": int(smooth_bins),
        "packets_total": 0,
        "packets_pb": 0,
        "packets_key_skipped": 0,
        "packets_missing_pts": 0,
        "packets_missing_size": 0,
        "duration_est_sec": 0.0,
    }

    bin_sec = float(bin_sec) if (bin_sec and bin_sec > 1e-6) else 0.5
    smooth_bins = int(max(0, int(smooth_bins)))

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_packets",
        "-show_entries", "packet=pts_time,size,flags",
        "-of", "json",
        str(video_path),
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            dbg["error"] = "ffprobe_failed"
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), dbg
        j = json.loads(p.stdout or "{}")
        packets = j.get("packets", [])
        if not isinstance(packets, list) or not packets:
            dbg["error"] = "no_packets"
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), dbg
    except Exception:
        dbg["error"] = "ffprobe_exception"
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), dbg

    dbg["packets_total"] = int(len(packets))

    ts: List[float] = []
    es: List[float] = []
    t_max = 0.0

    for pkt in packets:
        if not isinstance(pkt, dict):
            continue
        flags = str(pkt.get("flags") or "")
        # Skip keyframes (I)
        if "K" in flags:
            dbg["packets_key_skipped"] += 1
            continue

        pts_s = pkt.get("pts_time")
        size_s = pkt.get("size")
        if pts_s is None:
            dbg["packets_missing_pts"] += 1
            continue
        if size_s is None:
            dbg["packets_missing_size"] += 1
            continue

        try:
            t = float(pts_s)
            sz = float(size_s)
        except Exception:
            continue
        if (not np.isfinite(t)) or t < 0:
            continue
        if (not np.isfinite(sz)) or sz < 0:
            continue

        e = float(np.log1p(sz))
        ts.append(t)
        es.append(e)
        if t > t_max:
            t_max = t

    dbg["packets_pb"] = int(len(ts))
    dbg["duration_est_sec"] = float(t_max)

    if len(ts) == 0:
        dbg["error"] = "no_pb_packets"
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), dbg

    nb = int(max(1, math.ceil((t_max + 1e-9) / bin_sec)))
    energy = np.zeros((nb,), dtype=np.float32)
    counts = np.zeros((nb,), dtype=np.int32)

    for t, e in zip(ts, es):
        bi = int(min(nb - 1, max(0, int(t // bin_sec))))
        energy[bi] += float(e)
        counts[bi] += 1

    if smooth_bins > 0 and nb > 1:
        k = int(smooth_bins)
        sm = np.zeros_like(energy)
        for i in range(nb):
            a = max(0, i - k)
            b = min(nb, i + k + 1)
            sm[i] = float(energy[a:b].mean())
        energy = sm

    centers = (np.arange(nb, dtype=np.float32) + 0.5) * float(bin_sec)

    dbg["bin_count"] = int(nb)
    dbg["bin_nonzero"] = int(np.sum(counts > 0))
    dbg["counts_minmax"] = [int(counts.min()), int(counts.max())] if counts.size else [0, 0]
    return centers.astype(np.float32), energy.astype(np.float32), dbg
