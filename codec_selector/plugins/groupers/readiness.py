#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Readiness-based grouping plugin."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from codec_selector.core.config import BitcostReadinessConfig, GroupSpec
from codec_selector.core.registry import groupers
from codec_selector.codec_patch_gop.scoring import patch_scores_to_block_scores, score_map_to_patch_scores


def precompute_frame_block_scores(score_maps: List[np.ndarray], patch: int, block_size: int = 2) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for score in score_maps:
        ps = score_map_to_patch_scores(np.asarray(score, dtype=np.float32), patch=int(patch))
        bs = patch_scores_to_block_scores(ps, block_size=int(block_size))
        out.append(bs.reshape(-1).astype(np.float32))
    return out


def compute_group_block_budget(
    h1: int,
    w1: int,
    patch: int,
    images_per_group: int,
    block_size: int = 2,
) -> Tuple[int, int, int, int]:
    p = int(patch)
    b = int(max(1, int(block_size)))
    hb = int(h1 // p)
    wb = int(w1 // p)
    if hb % b != 0 or wb % b != 0:
        raise RuntimeError(f"grid not divisible by block_size={b}: hb={hb} wb={wb}")
    s_full = int(hb * wb)
    total_patches = int(images_per_group) * int(s_full)
    block_patch_count = int(b * b)
    block_budget = max(0, int((total_patches - s_full) // block_patch_count))
    return hb, wb, s_full, block_budget


def compute_group_readiness_stats(
    group_scores: List[np.ndarray],
    patch: int,
    images_per_group: int,
    coverage_bins_target: int,
    block_size: int = 2,
) -> Dict[str, Any]:
    group_block_scores = precompute_frame_block_scores(
        group_scores,
        patch=int(patch),
        block_size=int(block_size),
    )
    h1, w1 = group_scores[0].shape[:2] if group_scores else (0, 0)
    return compute_group_readiness_stats_from_blocks(
        group_block_scores=group_block_scores,
        h1=int(h1),
        w1=int(w1),
        patch=int(patch),
        images_per_group=int(images_per_group),
        coverage_bins_target=int(coverage_bins_target),
        block_size=int(block_size),
    )


def compute_group_readiness_stats_from_blocks(
    group_block_scores: List[np.ndarray],
    h1: int,
    w1: int,
    patch: int,
    images_per_group: int,
    coverage_bins_target: int,
    block_size: int = 2,
) -> Dict[str, Any]:
    if not group_block_scores:
        return {
            "candidate_block_count": 0,
            "topk_candidate_block_score_sum": 0.0,
            "topk_candidate_block_score_mean": 0.0,
            "topk_candidate_block_score_density": 0.0,
            "selected_time_bins": 0,
            "selected_time_bin_ratio": 0.0,
            "selected_unique_frames": 0,
            "selected_unique_frame_ratio": 0.0,
            "block_budget": 0,
        }

    _, _, _, block_budget = compute_group_block_budget(
        int(h1),
        int(w1),
        patch=int(patch),
        images_per_group=int(images_per_group),
        block_size=int(block_size),
    )

    scores_all: List[np.ndarray] = []
    t_all: List[np.ndarray] = []
    for t in range(1, len(group_block_scores)):
        flat = np.asarray(group_block_scores[t], dtype=np.float32).reshape(-1)
        if flat.size <= 0:
            continue
        scores_all.append(flat)
        t_all.append(np.full((flat.size,), int(t), dtype=np.int32))

    if not scores_all or int(block_budget) <= 0:
        return {
            "candidate_block_count": 0,
            "topk_candidate_block_score_sum": 0.0,
            "topk_candidate_block_score_mean": 0.0,
            "topk_candidate_block_score_density": 0.0,
            "selected_time_bins": 0,
            "selected_time_bin_ratio": 0.0,
            "selected_unique_frames": 0,
            "selected_unique_frame_ratio": 0.0,
            "block_budget": int(block_budget),
        }

    scores_cat = np.concatenate(scores_all)
    t_cat = np.concatenate(t_all)
    k = int(min(int(block_budget), int(len(scores_cat))))
    if k < len(scores_cat):
        topk_idx = np.argpartition(-scores_cat, kth=k - 1)[:k]
        topk_idx = topk_idx[np.argsort(-scores_cat[topk_idx])]
    else:
        topk_idx = np.arange(len(scores_cat), dtype=np.int32)

    selected_t = t_cat[topk_idx] if topk_idx.size > 0 else np.zeros((0,), dtype=np.int32)
    selected_scores = scores_cat[topk_idx] if topk_idx.size > 0 else np.zeros((0,), dtype=np.float32)
    unique_frames = sorted(set(int(x) for x in selected_t.tolist()))

    coverage_bins = int(max(1, coverage_bins_target))
    selected_time_bins = 0
    total_candidate_frames = int(max(0, len(group_block_scores) - 1))
    if unique_frames:
        t_edges = np.linspace(1, max(2, len(group_block_scores)), coverage_bins + 1, dtype=np.int32).tolist()
        covered = set()
        for tt in unique_frames:
            for bi in range(coverage_bins):
                lo = int(t_edges[bi])
                hi = int(t_edges[bi + 1]) if (bi + 1) < len(t_edges) else int(len(group_scores))
                if hi <= lo:
                    hi = lo + 1
                if lo <= int(tt) < hi:
                    covered.add(int(bi))
                    break
        selected_time_bins = int(len(covered))

    return {
        "candidate_block_count": int(len(scores_cat)),
        "topk_candidate_block_score_sum": float(selected_scores.sum()) if selected_scores.size > 0 else 0.0,
        "topk_candidate_block_score_mean": float(selected_scores.mean()) if selected_scores.size > 0 else 0.0,
        "topk_candidate_block_score_density": float(selected_scores.sum()) / float(max(1, len(group_block_scores))),
        "selected_time_bins": int(selected_time_bins),
        "selected_time_bin_ratio": float(selected_time_bins) / float(max(1, coverage_bins)),
        "selected_unique_frames": int(len(unique_frames)),
        "selected_unique_frame_ratio": float(len(unique_frames)) / float(max(1, total_candidate_frames)),
        "block_budget": int(block_budget),
        "selected_t_rel": [int(x) for x in unique_frames],
    }


def estimate_readiness_threshold(
    score_maps: List[np.ndarray],
    group_size: int,
    patch: int,
    images_per_group: int,
    coverage_bins_target: int,
    block_size: int = 2,
    block_scores_all: Optional[List[np.ndarray]] = None,
) -> float:
    vals: List[float] = []
    step = int(max(1, group_size))
    if block_scores_all is None:
        block_scores_all = precompute_frame_block_scores(score_maps, patch=int(patch), block_size=int(block_size))
    h1, w1 = score_maps[0].shape[:2] if score_maps else (0, 0)
    for start in range(0, len(score_maps), step):
        end = min(len(score_maps), start + step)
        if (end - start) < max(4, coverage_bins_target + 1):
            continue
        stats = compute_group_readiness_stats_from_blocks(
            group_block_scores=block_scores_all[start:end],
            h1=int(h1),
            w1=int(w1),
            patch=int(patch),
            images_per_group=int(images_per_group),
            coverage_bins_target=int(coverage_bins_target),
            block_size=int(block_size),
        )
        vals.append(float(stats.get("topk_candidate_block_score_sum", 0.0)))
    if not vals:
        return 0.0
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def resolve_group_frame_limits(cfg: BitcostReadinessConfig) -> Tuple[int, int, Dict[str, Any]]:
    fps_use = float(cfg.sample_fps) if float(cfg.sample_fps) > 0 else 1.0
    if float(cfg.min_group_sec) > 0.0:
        min_frames = int(max(4, round(float(cfg.min_group_sec) * fps_use)))
        min_source = "seconds"
    else:
        min_frames = int(max(4, int(cfg.min_group_frames)))
        min_source = "frames"

    if float(cfg.max_group_sec) > 0.0:
        max_frames = int(max(int(min_frames), round(float(cfg.max_group_sec) * fps_use)))
        max_source = "seconds"
    else:
        max_frames = int(max(int(min_frames), int(cfg.max_group_frames)))
        max_source = "frames"

    return int(min_frames), int(max_frames), {
        "sample_fps": float(cfg.sample_fps),
        "min_group_frames_resolved": int(min_frames),
        "max_group_frames_resolved": int(max_frames),
        "min_group_source": str(min_source),
        "max_group_source": str(max_source),
        "min_group_sec_input": float(cfg.min_group_sec),
        "max_group_sec_input": float(cfg.max_group_sec),
        "min_group_frames_input": int(cfg.min_group_frames),
        "max_group_frames_input": int(cfg.max_group_frames),
    }


def build_readiness_groups(
    frame_ids: List[int],
    score_maps: List[np.ndarray],
    cfg: BitcostReadinessConfig,
    readiness_sum_threshold: float,
    min_group_frames: int,
    max_group_frames: int,
    block_scores_all: Optional[List[np.ndarray]] = None,
) -> List[GroupSpec]:
    groups: List[GroupSpec] = []
    n = int(len(frame_ids))
    if block_scores_all is None:
        block_scores_all = precompute_frame_block_scores(
            score_maps,
            patch=int(cfg.patch),
            block_size=int(cfg.block_size),
        )
    h1, w1 = score_maps[0].shape[:2] if score_maps else (0, 0)
    start = 0
    while start < n:
        min_end = int(min(n, start + max(1, int(min_group_frames))))
        max_end = int(min(n, start + max(1, int(max_group_frames))))
        if (n - start) < max(4, int(min_group_frames)):
            break

        prev_sum: Optional[float] = None
        chosen_end: Optional[int] = None
        chosen_stats: Optional[Dict[str, Any]] = None
        stop_reason = "max_group_frames"

        end = int(min_end)
        while end <= max_end:
            stats = compute_group_readiness_stats_from_blocks(
                group_block_scores=block_scores_all[start:end],
                h1=int(h1),
                w1=int(w1),
                patch=int(cfg.patch),
                images_per_group=int(cfg.images_per_group),
                coverage_bins_target=int(cfg.readiness_coverage_bins),
                block_size=int(cfg.block_size),
            )
            cur_sum = float(stats.get("topk_candidate_block_score_sum", 0.0))
            coverage_ok = int(stats.get("selected_time_bins", 0)) >= int(min(cfg.readiness_coverage_bins, cfg.images_per_group - 1))
            threshold_ok = cur_sum >= float(readiness_sum_threshold)
            delta_ratio = 1.0
            if prev_sum is not None:
                delta_ratio = float((cur_sum - prev_sum) / max(abs(prev_sum), 1e-6))

            if threshold_ok and coverage_ok:
                chosen_end = int(end)
                chosen_stats = stats
                stop_reason = "readiness_threshold"
                if prev_sum is not None and delta_ratio < float(cfg.readiness_delta_ratio):
                    stop_reason = "readiness_threshold_low_gain"
                    break

            prev_sum = float(cur_sum)
            end += 1

        if chosen_end is None:
            chosen_end = int(max_end)
            chosen_stats = compute_group_readiness_stats_from_blocks(
                group_block_scores=block_scores_all[start:chosen_end],
                h1=int(h1),
                w1=int(w1),
                patch=int(cfg.patch),
                images_per_group=int(cfg.images_per_group),
                coverage_bins_target=int(cfg.readiness_coverage_bins),
                block_size=int(cfg.block_size),
            )
            if int(chosen_end) >= int(n):
                stop_reason = "tail_flush"

        groups.append(
            GroupSpec(
                group_idx=int(len(groups)),
                start=int(start),
                end=int(chosen_end),
                frame_count=int(chosen_end - start),
                stop_reason=str(stop_reason),
                readiness=chosen_stats if chosen_stats is not None else {},
            )
        )
        start = int(chosen_end)
    return groups


groupers.register("readiness", build_readiness_groups)
