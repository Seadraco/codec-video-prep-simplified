"""Public configuration for codec-aware video preprocessing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PreinferConfig:
    video: str
    out_dir: str
    num_sampled_frames: int = 1024
    group_size: int = 32
    images_per_group: int = 4
    patch: int = 14
    max_pixels: int = 153664
    min_group_frames: int = 8
    max_group_frames: int = 64
    thread_type: str = "auto"
    thread_count: int = 16
    bitcost_grid: str = "adaptive"
    canvas_format: str = "jpg"
    avoid_keyframes: bool = True
    readiness_sum_threshold: float = 0.0
    grouping_mode: str = "readiness"
    frame_sampling_mode: str = "uniform_count"
    sample_fps: float = 4.0
    readiness_sum_threshold_mode: str = "legacy"
    readiness_norm_sum_threshold: float = 2250000.0
    # ---- misc ----
    save_mask_video: bool = False
    parallel_decode_cv_reader: bool = False
