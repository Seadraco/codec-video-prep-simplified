#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration objects for codec selector pipelines."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VideoMetadata:
    video: str
    total_frames: int
    fps: float
    height: int
    width: int
    codec_name: str
    stream_bit_rate: float = 0.0
    format_bit_rate: float = 0.0
    duration: float = 0.0

    @property
    def bit_rate(self) -> float:
        return float(self.stream_bit_rate or self.format_bit_rate or 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupSpec:
    group_idx: int
    start: int
    end: int
    frame_count: int
    stop_reason: str
    readiness: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    out_dir: str
    meta_path: str
    canvas_files: List[str]
    summary: Dict[str, Any]


@dataclass
class BitcostReadinessConfig:
    video: str
    out_dir: str

    frame_sampling_mode: str = "fps"
    sample_fps: float = 4.0
    num_sampled_frames: int = 0
    pkt_peak_bin_sec: float = 0.5
    pkt_peak_cap: int = 128
    pkt_peak_per_sec: float = 1.0
    pkt_peak_neighbor: int = 0
    pkt_peak_smooth_bins: int = 1

    grouping_mode: str = "fixed"
    group_size: int = 32
    min_group_frames: int = 8
    max_group_frames: int = 64
    min_group_sec: float = 0.0
    max_group_sec: float = 0.0

    images_per_group: int = 4
    patch: int = 14
    block_size: int = 2
    max_dim: int = 616
    max_pixels: int = 150000
    no_resize: bool = False

    bitcost_grid: str = "sub"
    bitcost_pct: float = 99.0
    bitcost_log_scale: bool = True

    frame_score_norm_mode: str = "none"
    frame_score_norm_floor_ratio: float = 0.5
    frame_score_norm_allow_boost: bool = False

    avoid_keyframes: bool = False
    avoid_keyframe_offset: int = 1

    canvas_format: str = "jpg"
    save_mask_video: bool = False
    video_fps: float = 0.0

    readiness_sum_threshold: float = 0.0
    readiness_sum_threshold_mode: str = "legacy"
    readiness_norm_sum_threshold: float = 1050000.0
    bpppf_clamp_min: float = 0.015
    bpppf_clamp_max: float = 0.09
    readiness_coverage_bins: int = 3
    readiness_delta_ratio: float = 0.05

    # The selector engine intentionally uses ffmpeg subprocess decoding only.
    # The old OpenCV decode backend is slower for this workload and is not exposed here.
    decode_backend: str = "ffmpeg_native"
    ffmpeg_preprocess_frames: bool = True
    parallel_decode_cv_reader: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "BitcostReadinessConfig":
        self.frame_sampling_mode = str(self.frame_sampling_mode).lower().strip()
        self.grouping_mode = str(self.grouping_mode).lower().strip()
        self.bitcost_grid = str(self.bitcost_grid).lower().strip()
        self.frame_score_norm_mode = str(self.frame_score_norm_mode).lower().strip()
        self.readiness_sum_threshold_mode = str(self.readiness_sum_threshold_mode).lower().strip()
        self.canvas_format = str(self.canvas_format).lower().strip()
        self.decode_backend = str(self.decode_backend).lower().strip()
        if self.decode_backend not in {"ffmpeg_native", "cv_reader_pixels"}:
            self.decode_backend = "ffmpeg_native"
        self.ffmpeg_preprocess_frames = bool(self.ffmpeg_preprocess_frames)
        self.parallel_decode_cv_reader = bool(self.parallel_decode_cv_reader)
        self.patch = int(max(1, int(self.patch)))
        self.block_size = int(max(1, int(self.block_size)))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
