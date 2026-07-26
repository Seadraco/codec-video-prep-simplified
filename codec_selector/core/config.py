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
    target_canvas: int = 0
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
    iframe_score_clip_mode: str = "none"
    iframe_score_clip_percentile: float = 95.0

    avoid_keyframes: bool = False
    avoid_keyframe_offset: int = 1

    canvas_format: str = "jpg"
    save_mask_video: bool = False
    video_fps: float = 0.0
    verbose: bool = False

    adaptive_anchor: bool = False
    adaptive_anchor_w_center: float = 0.7
    adaptive_anchor_w_low_bc: float = 0.3
    adaptive_gop: bool = False
    adaptive_gop_gamma: float = 0.0
    adaptive_gop_percentile: float = 75.0
    event_aggregation: bool = False
    event_aggregation_bins: int = 4
    event_aggregation_min_blocks: int = 8

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

    parallel_segments: int = 0
    threads_per_segment: int = 4
    segment_guard_frames: int = 30

    extra: Dict[str, Any] = field(default_factory=dict)
    selector_mode: str = "topk_2x2_bitcost"
    diversity_fraction: float = 0.10
    novelty_weight: float = 0.5
    dedup_enabled: bool = True
    dedup_descriptor: str = "pooled4"
    dedup_threshold: Optional[float] = None
    dedup_threshold_mode: str = "absolute"
    dedup_quantile: float = 0.10
    common_cache_dir: str = ""

    def normalized(self) -> "BitcostReadinessConfig":
        self.frame_sampling_mode = str(self.frame_sampling_mode).lower().strip()
        self.grouping_mode = str(self.grouping_mode).lower().strip()
        self.bitcost_grid = str(self.bitcost_grid).lower().strip()
        self.frame_score_norm_mode = str(self.frame_score_norm_mode).lower().strip()
        self.iframe_score_clip_mode = str(self.iframe_score_clip_mode).lower().strip()
        self.readiness_sum_threshold_mode = str(self.readiness_sum_threshold_mode).lower().strip()
        self.canvas_format = str(self.canvas_format).lower().strip()
        self.decode_backend = str(self.decode_backend).lower().strip()
        if self.decode_backend not in {"ffmpeg_native", "cv_reader_pixels"}:
            self.decode_backend = "ffmpeg_native"
        self.ffmpeg_preprocess_frames = bool(self.ffmpeg_preprocess_frames)
        self.parallel_decode_cv_reader = bool(self.parallel_decode_cv_reader)
        self.parallel_segments = max(0, int(self.parallel_segments))
        self.threads_per_segment = max(1, int(self.threads_per_segment))
        self.segment_guard_frames = max(0, int(self.segment_guard_frames))
        self.target_canvas = max(0, int(self.target_canvas))
        self.verbose = bool(self.verbose)
        self.adaptive_anchor = bool(self.adaptive_anchor)
        self.adaptive_anchor_w_center = float(self.adaptive_anchor_w_center)
        self.adaptive_anchor_w_low_bc = float(self.adaptive_anchor_w_low_bc)
        self.adaptive_gop = bool(self.adaptive_gop)
        self.adaptive_gop_gamma = float(self.adaptive_gop_gamma)
        self.adaptive_gop_percentile = float(self.adaptive_gop_percentile)
        self.event_aggregation = bool(self.event_aggregation)
        self.event_aggregation_bins = max(1, int(self.event_aggregation_bins))
        self.event_aggregation_min_blocks = max(0, int(self.event_aggregation_min_blocks))
        self.selector_mode = str(self.selector_mode).lower().strip()
        self.diversity_fraction = float(self.diversity_fraction)
        self.novelty_weight = float(self.novelty_weight)
        self.dedup_enabled = bool(self.dedup_enabled)
        self.dedup_descriptor = str(self.dedup_descriptor).lower().strip()
        self.dedup_threshold_mode = str(self.dedup_threshold_mode).lower().strip()
        self.dedup_quantile = float(self.dedup_quantile)
        self.common_cache_dir = str(self.common_cache_dir).strip()
        if self.dedup_threshold is not None:
            self.dedup_threshold = float(self.dedup_threshold)
        if self.selector_mode not in {"topk_2x2_bitcost", "diverse_mixed_simple"}:
            raise ValueError(f"unsupported selector_mode: {self.selector_mode}")
        if not 0.0 <= self.diversity_fraction <= 1.0:
            raise ValueError("diversity_fraction must be between 0 and 1")
        if not 0.0 <= self.novelty_weight <= 1.0:
            raise ValueError("novelty_weight must be between 0 and 1")
        if self.dedup_descriptor not in {"pooled4", "full"}:
            raise ValueError(f"unsupported dedup_descriptor: {self.dedup_descriptor}")
        if self.dedup_threshold is not None and self.dedup_threshold < 0.0:
            raise ValueError("dedup_threshold must be >= 0")
        if self.dedup_threshold_mode not in {"absolute", "group_quantile"}:
            raise ValueError(
                f"unsupported dedup_threshold_mode: {self.dedup_threshold_mode}"
            )
        if not 0.0 <= self.dedup_quantile <= 1.0:
            raise ValueError("dedup_quantile must be between 0 and 1")
        if self.selector_mode == "diverse_mixed_simple" and self.event_aggregation:
            raise ValueError("event_aggregation is only supported by selector_mode=topk_2x2_bitcost")
        self.patch = int(max(1, int(self.patch)))
        self.block_size = int(max(1, int(self.block_size)))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
