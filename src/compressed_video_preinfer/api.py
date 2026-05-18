"""Python API for the optimized pre-infer pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codec_selector.core.config import BitcostReadinessConfig
from codec_selector.core.pipeline import run_bitcost_readiness

from .config import PreinferConfig


@dataclass
class PreinferResult:
    out_dir: str
    meta_path: str
    canvas_files: list[str]
    summary: dict[str, Any]

    @property
    def timings(self) -> dict[str, Any]:
        return dict(self.summary.get("timings") or self.summary.get("timing_sec") or {})


def _configure_native_threads(config: PreinferConfig) -> None:
    thread_type = str(config.thread_type).lower().strip()
    if thread_type and thread_type != "auto":
        os.environ["CV_READER_FAST_THREAD_TYPE"] = thread_type
    else:
        os.environ.pop("CV_READER_FAST_THREAD_TYPE", None)
    os.environ["CV_READER_FAST_THREAD_COUNT"] = str(int(config.thread_count))


def run_preinfer(
    video: str,
    out_dir: str,
    num_sampled_frames: int = 1024,
    group_size: int = 32,
    images_per_group: int = 4,
    patch: int = 14,
    max_pixels: int = 153664,
    min_group_frames: int = 8,
    max_group_frames: int = 64,
    bitcost_grid: str = "adaptive",
) -> PreinferResult:
    """Run the optimized H.264/HEVC bitcost readiness pre-infer path."""
    config = PreinferConfig(
        video=video,
        out_dir=out_dir,
        num_sampled_frames=num_sampled_frames,
        group_size=group_size,
        images_per_group=images_per_group,
        patch=patch,
        max_pixels=max_pixels,
        min_group_frames=min_group_frames,
        max_group_frames=max_group_frames,
        bitcost_grid=bitcost_grid,
    )
    return run_preinfer_config(config)


def run_preinfer_config(config: PreinferConfig) -> PreinferResult:
    _configure_native_threads(config)
    cfg = BitcostReadinessConfig(
        video=str(config.video),
        out_dir=str(config.out_dir),
        frame_sampling_mode="uniform_count",
        num_sampled_frames=int(config.num_sampled_frames),
        grouping_mode="readiness",
        group_size=int(config.group_size),
        images_per_group=int(config.images_per_group),
        patch=int(config.patch),
        max_pixels=int(config.max_pixels),
        min_group_frames=int(config.min_group_frames),
        max_group_frames=int(config.max_group_frames),
        bitcost_grid=str(config.bitcost_grid),
        canvas_format=str(config.canvas_format),
        avoid_keyframes=bool(config.avoid_keyframes),
        readiness_sum_threshold=float(config.readiness_sum_threshold),
        parallel_decode_cv_reader=False,
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
