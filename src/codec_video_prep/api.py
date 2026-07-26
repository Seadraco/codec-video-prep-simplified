"""Python API for codec-aware video preprocessing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codec_selector.core.config import BitcostReadinessConfig
from codec_selector.core.pipeline import run_bitcost_readiness

from .config import PreinferConfig
from typing import Dict, List, Tuple


@dataclass
class PreinferResult:
    out_dir: str
    meta_path: str
    canvas_files: List[str]
    summary: Dict[str, Any]

    @property
    def timings(self) -> Dict[str, Any]:
        return dict(self.summary.get("timings") or self.summary.get("timing_sec") or {})


def _configure_native_threads(config: PreinferConfig) -> None:
    thread_type = str(config.thread_type).lower().strip()
    if thread_type and thread_type != "auto":
        os.environ["CV_READER_FAST_THREAD_TYPE"] = thread_type
    # Preserve an existing CV_READER_FAST_THREAD_TYPE when thread_type=auto.
    os.environ["CV_READER_FAST_THREAD_COUNT"] = str(int(config.thread_count))
    if bool(config.disable_target_only):
        os.environ["CVR_DISABLE_TARGET_ONLY"] = "1"


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _environment_optional_float(name: str, default: float | None) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


def run_preinfer(
    video: str,
    out_dir: str,
    num_sampled_frames: int = 1024,
    group_size: int = 32,
    target_canvas: int = 0,
    images_per_group: int = 4,
    patch: int = 14,
    max_pixels: int = 153664,
    block_size: int = 2,
    min_group_frames: int = 8,
    max_group_frames: int = 64,
    thread_type: str = "slice",
    thread_count: int = 1,
    disable_target_only: bool = False,
    bitcost_grid: str = "adaptive",
    canvas_format: str = "jpg",
    grouping_mode: str = "readiness",
    frame_sampling_mode: str = "uniform_count",
    sample_fps: float = 4.0,
    readiness_sum_threshold: float = 0.0,
    readiness_sum_threshold_mode: str = "legacy",
    readiness_norm_sum_threshold: float = 2250000.0,
    iframe_score_clip_mode: str = "none",
    iframe_score_clip_percentile: float = 95.0,
    avoid_keyframes: bool = True,
    save_mask_video: bool = False,
    verbose: bool = False,
    adaptive_anchor: bool = False,
    adaptive_anchor_w_center: float = 0.7,
    adaptive_anchor_w_low_bc: float = 0.3,
    adaptive_gop: bool = False,
    adaptive_gop_gamma: float = 0.0,
    adaptive_gop_percentile: float = 75.0,
    event_aggregation: bool = False,
    event_aggregation_bins: int = 4,
    event_aggregation_min_blocks: int = 8,
    parallel_decode_cv_reader: bool = False,
    decode_backend: str = "cv_reader_pixels",
    parallel_segments: int = 0,
    threads_per_segment: int = 4,
    segment_guard_frames: int = 30,
    selector_mode: str = "topk_2x2_bitcost",
    diversity_fraction: float = 0.25,
    novelty_weight: float = 0.5,
    dedup_enabled: bool = True,
    dedup_descriptor: str = "pooled4",
    dedup_threshold: float | None = None,
) -> PreinferResult:
    """Run the optimized H.264/HEVC bitcost readiness preprocessing path."""
    config = PreinferConfig(
        video=video,
        out_dir=out_dir,
        num_sampled_frames=num_sampled_frames,
        group_size=group_size,
        target_canvas=target_canvas,
        images_per_group=images_per_group,
        patch=patch,
        max_pixels=max_pixels,
        block_size=block_size,
        min_group_frames=min_group_frames,
        max_group_frames=max_group_frames,
        thread_type=thread_type,
        thread_count=thread_count,
        disable_target_only=disable_target_only,
        bitcost_grid=bitcost_grid,
        canvas_format=canvas_format,
        grouping_mode=grouping_mode,
        frame_sampling_mode=frame_sampling_mode,
        sample_fps=sample_fps,
        readiness_sum_threshold=readiness_sum_threshold,
        readiness_sum_threshold_mode=readiness_sum_threshold_mode,
        readiness_norm_sum_threshold=readiness_norm_sum_threshold,
        iframe_score_clip_mode=iframe_score_clip_mode,
        iframe_score_clip_percentile=iframe_score_clip_percentile,
        avoid_keyframes=avoid_keyframes,
        save_mask_video=save_mask_video,
        verbose=verbose,
        adaptive_anchor=adaptive_anchor,
        adaptive_anchor_w_center=adaptive_anchor_w_center,
        adaptive_anchor_w_low_bc=adaptive_anchor_w_low_bc,
        adaptive_gop=adaptive_gop,
        adaptive_gop_gamma=adaptive_gop_gamma,
        adaptive_gop_percentile=adaptive_gop_percentile,
        event_aggregation=event_aggregation,
        event_aggregation_bins=event_aggregation_bins,
        event_aggregation_min_blocks=event_aggregation_min_blocks,
        selector_mode=selector_mode,
        diversity_fraction=diversity_fraction,
        novelty_weight=novelty_weight,
        dedup_enabled=dedup_enabled,
        dedup_descriptor=dedup_descriptor,
        dedup_threshold=dedup_threshold,
        parallel_decode_cv_reader=parallel_decode_cv_reader,
        decode_backend=decode_backend,
        parallel_segments=parallel_segments,
        threads_per_segment=threads_per_segment,
        segment_guard_frames=segment_guard_frames,
    )
    return run_preinfer_config(config)


def run_preinfer_config(config: PreinferConfig) -> PreinferResult:
    _configure_native_threads(config)
    selector_mode = os.environ.get("CODEC_SELECTOR_MODE", str(config.selector_mode))
    diversity_fraction = float(
        os.environ.get("CODEC_DIVERSITY_FRACTION", str(config.diversity_fraction))
    )
    novelty_weight = float(
        os.environ.get("CODEC_NOVELTY_WEIGHT", str(config.novelty_weight))
    )
    dedup_enabled = _environment_bool("CODEC_DEDUP_ENABLED", config.dedup_enabled)
    dedup_descriptor = os.environ.get("CODEC_DEDUP_DESCRIPTOR", str(config.dedup_descriptor))
    dedup_threshold = _environment_optional_float(
        "CODEC_DEDUP_THRESHOLD", config.dedup_threshold
    )
    cfg = BitcostReadinessConfig(
        video=str(config.video),
        out_dir=str(config.out_dir),
        frame_sampling_mode=str(config.frame_sampling_mode),
        sample_fps=float(config.sample_fps),
        num_sampled_frames=int(config.num_sampled_frames),
        grouping_mode=str(config.grouping_mode),
        group_size=int(config.group_size),
        target_canvas=int(config.target_canvas),
        images_per_group=int(config.images_per_group),
        patch=int(config.patch),
        max_pixels=int(config.max_pixels),
        block_size=int(config.block_size),
        min_group_frames=int(config.min_group_frames),
        max_group_frames=int(config.max_group_frames),
        bitcost_grid=str(config.bitcost_grid),
        canvas_format=str(config.canvas_format),
        avoid_keyframes=bool(config.avoid_keyframes),
        readiness_sum_threshold=float(config.readiness_sum_threshold),
        readiness_sum_threshold_mode=str(config.readiness_sum_threshold_mode),
        readiness_norm_sum_threshold=float(config.readiness_norm_sum_threshold),
        iframe_score_clip_mode=str(config.iframe_score_clip_mode),
        iframe_score_clip_percentile=float(config.iframe_score_clip_percentile),
        bpppf_clamp_min=0.015,
        bpppf_clamp_max=0.09,
        readiness_coverage_bins=3,
        readiness_delta_ratio=0.05,
        frame_score_norm_mode="none",
        frame_score_norm_floor_ratio=0.5,
        frame_score_norm_allow_boost=False,
        save_mask_video=bool(config.save_mask_video),
        verbose=bool(config.verbose),
        adaptive_anchor=bool(config.adaptive_anchor),
        adaptive_anchor_w_center=float(config.adaptive_anchor_w_center),
        adaptive_anchor_w_low_bc=float(config.adaptive_anchor_w_low_bc),
        adaptive_gop=bool(config.adaptive_gop),
        adaptive_gop_gamma=float(config.adaptive_gop_gamma),
        adaptive_gop_percentile=float(config.adaptive_gop_percentile),
        event_aggregation=bool(config.event_aggregation),
        event_aggregation_bins=int(config.event_aggregation_bins),
        event_aggregation_min_blocks=int(config.event_aggregation_min_blocks),
        selector_mode=str(selector_mode),
        diversity_fraction=float(diversity_fraction),
        novelty_weight=float(novelty_weight),
        dedup_enabled=bool(dedup_enabled),
        dedup_descriptor=str(dedup_descriptor),
        dedup_threshold=dedup_threshold,
        parallel_decode_cv_reader=bool(config.parallel_decode_cv_reader),
        decode_backend=str(config.decode_backend),
        parallel_segments=int(config.parallel_segments),
        threads_per_segment=int(config.threads_per_segment),
        segment_guard_frames=int(config.segment_guard_frames),
    )
    result = run_bitcost_readiness(cfg)
    meta_path = Path(result.meta_path)
    summary = dict(result.summary)
    timings = dict(summary.get("timing_sec") or {})
    summary["timings"] = timings
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["timings"] = timings
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        summary = meta
    return PreinferResult(
        out_dir=str(result.out_dir),
        meta_path=str(result.meta_path),
        canvas_files=list(result.canvas_files),
        summary=summary,
    )
