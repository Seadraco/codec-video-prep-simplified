#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frame sampling plugins."""

from typing import Any, Dict, List, Tuple

import numpy as np

from codec_selector.core.config import BitcostReadinessConfig, VideoMetadata
from codec_selector.core.registry import samplers
from codec_selector.codec_patch_gop.energy_sampling import pick_peak_frame_ids_from_pkt_size


def sample_fps(total_frames: int, fps: float, sample_fps_value: float) -> List[int]:
    total_frames = int(max(1, int(total_frames)))
    fps_use = float(fps) if fps and fps > 0 else 30.0
    sample_rate = float(sample_fps_value) if sample_fps_value > 0 else 4.0
    duration_sec = float(total_frames) / float(fps_use)
    num_samples = max(1, int(np.ceil(duration_sec * sample_rate)))
    frame_ids: List[int] = []
    for i in range(num_samples):
        fid = int(round((float(i) / float(sample_rate)) * fps_use))
        frame_ids.append(int(max(0, min(total_frames - 1, fid))))
    return frame_ids


def sample_uniform_count(total_frames: int, target_count: int) -> List[int]:
    total_frames = int(max(1, int(total_frames)))
    target_count = int(max(1, int(target_count)))
    if total_frames == 1:
        return [0] * target_count
    return [int(max(0, min(total_frames - 1, x))) for x in np.linspace(0, total_frames - 1, target_count, dtype=np.int32)]


def sample_all_frames(total_frames: int) -> List[int]:
    return list(range(int(max(1, int(total_frames)))))


def sample_pkt_size_peak(meta: VideoMetadata, cfg: BitcostReadinessConfig) -> Tuple[List[int], Dict[str, Any]]:
    frame_ids, dbg = pick_peak_frame_ids_from_pkt_size(
        video_path=str(meta.video),
        fps=float(meta.fps),
        total_frames=int(meta.total_frames),
        bin_sec=float(cfg.pkt_peak_bin_sec),
        peaks_cap=int(cfg.pkt_peak_cap),
        peaks_per_sec=float(cfg.pkt_peak_per_sec),
        neighbor=int(cfg.pkt_peak_neighbor),
        smooth_bins=int(cfg.pkt_peak_smooth_bins),
    )
    return [int(x) for x in frame_ids], dict(dbg or {})


def sample_fps_plus_pkt_size_peak(meta: VideoMetadata, cfg: BitcostReadinessConfig) -> Tuple[List[int], Dict[str, Any]]:
    fps_ids = sample_fps(meta.total_frames, meta.fps, cfg.sample_fps)
    peak_ids, dbg = sample_pkt_size_peak(meta, cfg)
    merged = sorted(set(int(x) for x in fps_ids) | set(int(x) for x in peak_ids))
    dbg = dict(dbg or {})
    dbg.update({
        "fps_base_count": int(len(fps_ids)),
        "pkt_peak_count": int(len(peak_ids)),
        "merged_count": int(len(merged)),
    })
    return merged, dbg


def sample_frames(meta: VideoMetadata, cfg: BitcostReadinessConfig) -> Tuple[List[int], Dict[str, Any]]:
    mode = str(cfg.frame_sampling_mode).lower().strip()
    debug: Dict[str, Any] = {"frame_sampling_mode": mode}
    if mode == "all_frames":
        frame_ids = sample_all_frames(meta.total_frames)
    elif mode == "uniform_count":
        if int(cfg.num_sampled_frames) <= 0:
            raise ValueError("uniform_count sampling requires num_sampled_frames > 0")
        frame_ids = sample_uniform_count(meta.total_frames, cfg.num_sampled_frames)
    elif mode == "pkt_size_peak":
        frame_ids, pkt_dbg = sample_pkt_size_peak(meta, cfg)
        debug["pkt_size_peak_debug"] = pkt_dbg
        if not frame_ids:
            frame_ids = sample_fps(meta.total_frames, meta.fps, cfg.sample_fps)
            debug["fallback"] = "fps"
    elif mode == "fps_plus_pkt_size_peak":
        frame_ids, pkt_dbg = sample_fps_plus_pkt_size_peak(meta, cfg)
        debug["pkt_size_peak_debug"] = pkt_dbg
        if not frame_ids:
            frame_ids = sample_fps(meta.total_frames, meta.fps, cfg.sample_fps)
            debug["fallback"] = "fps"
    else:
        frame_ids = sample_fps(meta.total_frames, meta.fps, cfg.sample_fps)

    debug["sampled_frames"] = int(len(frame_ids))
    return [int(x) for x in frame_ids], debug


samplers.register("fps", sample_frames)
samplers.register("uniform_count", sample_frames)
samplers.register("all_frames", sample_frames)
samplers.register("pkt_size_peak", sample_frames)
samplers.register("fps_plus_pkt_size_peak", sample_frames)
