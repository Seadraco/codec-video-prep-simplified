#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bitcost scorer plugin."""

from typing import Any, Dict, List

import numpy as np

from codec_selector.core.config import BitcostReadinessConfig, VideoMetadata
from codec_selector.core.registry import scorers
from codec_selector.codec_patch_gop.video_processor import bitcost_item_to_score_map, cv_reader_fetch_bitcost


def bitcost_items_to_score_maps(
    bitcost_items: List[Dict[str, Any]],
    out_h: int,
    out_w: int,
    bitcost_grid: str,
    bitcost_pct: float,
    bitcost_log_scale: bool,
    codec_name: str,
) -> List[np.ndarray]:
    score_maps: List[np.ndarray] = []
    for item in bitcost_items:
        score = bitcost_item_to_score_map(
            item,
            out_h=int(out_h),
            out_w=int(out_w),
            grid=str(bitcost_grid),
            pct=float(bitcost_pct),
            log_scale=bool(bitcost_log_scale),
            codec_name=str(codec_name),
        )
        score_maps.append(np.asarray(score, dtype=np.float32))
    return score_maps


def score_bitcost(
    video_path: str,
    frame_ids: List[int],
    out_h: int,
    out_w: int,
    meta: VideoMetadata,
    cfg: BitcostReadinessConfig,
) -> List[np.ndarray]:
    items = cv_reader_fetch_bitcost(str(video_path), [int(x) for x in frame_ids], bitcost_grid=str(cfg.bitcost_grid))
    if len(items) != len(frame_ids):
        raise RuntimeError(f"bitcost fetch mismatch: {len(items)} vs {len(frame_ids)}")
    return bitcost_items_to_score_maps(
        bitcost_items=items,
        out_h=int(out_h),
        out_w=int(out_w),
        bitcost_grid=str(cfg.bitcost_grid),
        bitcost_pct=float(cfg.bitcost_pct),
        bitcost_log_scale=bool(cfg.bitcost_log_scale),
        codec_name=str(meta.codec_name),
    )


scorers.register("bitcost", score_bitcost)
