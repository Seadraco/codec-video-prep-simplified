"""Public configuration for codec-aware video preprocessing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PreinferConfig:
    video: str
    out_dir: str
    num_sampled_frames: int = 1024
    group_size: int = 32
    target_canvas: int = 0
    images_per_group: int = 4
    patch: int = 14
    max_pixels: int = 153664
    block_size: int = 2
    min_group_frames: int = 8
    max_group_frames: int = 64
    thread_type: str = "slice"
    thread_count: int = 1
    disable_target_only: bool = False
    bitcost_grid: str = "adaptive"
    canvas_format: str = "jpg"
    avoid_keyframes: bool = True
    readiness_sum_threshold: float = 0.0
    grouping_mode: str = "readiness"
    frame_sampling_mode: str = "uniform_count"
    sample_fps: float = 4.0
    readiness_sum_threshold_mode: str = "legacy"
    readiness_norm_sum_threshold: float = 2250000.0
    iframe_score_clip_mode: str = "none"
    iframe_score_clip_percentile: float = 95.0
    # ---- misc ----
    save_mask_video: bool = False
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
    parallel_decode_cv_reader: bool = False
    decode_backend: str = "cv_reader_pixels"
    parallel_segments: int = 0
    threads_per_segment: int = 4
    segment_guard_frames: int = 30
    selector_mode: str = "topk_2x2_bitcost"
    diversity_fraction: float = 0.10
    novelty_weight: float = 0.5
    dedup_enabled: bool = True
    dedup_descriptor: str = "pooled4"
    dedup_threshold: float | None = None
