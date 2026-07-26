#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simplified mixed block selector with adjacent-position deduplication."""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from codec_selector.codec_patch_gop.frame_utils import frame_is_bad
from codec_selector.codec_patch_gop.patch_utils import (
    block_to_patches,
    extract_patch_bgr,
    iter_blocks_in_raster,
    pack_patches_to_canvases,
)
from codec_selector.codec_patch_gop.scoring import patch_scores_to_block_scores, score_map_to_patch_scores
from codec_selector.core.registry import selectors


DEFAULT_DIVERSITY_FRACTION = 0.25
DEFAULT_NOVELTY_WEIGHT = 0.5
DEFAULT_DEDUP_THRESHOLDS = {
    "pooled4": 0.025,
    "full": 0.035,
}
ADJACENT_MAD_CDF_THRESHOLDS = (
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.035,
    0.040,
    0.050,
    0.075,
    0.100,
)


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    n = int(values.size)
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts, dtype=np.int64)
    starts = ends - counts
    unique_ranks = (starts + ends - 1).astype(np.float64) / (2.0 * float(n - 1))
    return unique_ranks[inverse].astype(np.float32)


def _score_order(scores: np.ndarray, frame_indices: np.ndarray, block_indices: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    frame_indices = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    block_indices = np.asarray(block_indices, dtype=np.int32).reshape(-1)
    return np.lexsort((block_indices, frame_indices, -scores)).astype(np.int64)


def _frame_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _pooled4_descriptors(
    frames_bgr: List[np.ndarray],
    block_rows: int,
    block_cols: int,
) -> np.ndarray:
    out = np.empty((len(frames_bgr), block_rows, block_cols, 4, 4), dtype=np.float32)
    target_h = int(block_rows * 4)
    target_w = int(block_cols * 4)
    for frame_idx, frame in enumerate(frames_bgr):
        gray = _frame_gray(frame)
        pooled = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        out[frame_idx] = pooled.reshape(block_rows, 4, block_cols, 4).transpose(0, 2, 1, 3)
    return out


def _edge_scores(descriptors: np.ndarray) -> np.ndarray:
    horizontal = np.abs(descriptors[..., :, 1:] - descriptors[..., :, :-1]).mean(axis=(-2, -1))
    vertical = np.abs(descriptors[..., 1:, :] - descriptors[..., :-1, :]).mean(axis=(-2, -1))
    return (0.5 * horizontal + 0.5 * vertical).astype(np.float32)


def _adjacent_diff_pooled4(descriptors: np.ndarray) -> np.ndarray:
    frame_count, block_rows, block_cols = descriptors.shape[:3]
    out = np.full((frame_count, block_rows, block_cols), np.inf, dtype=np.float32)
    if frame_count > 1:
        out[1:] = np.abs(descriptors[1:] - descriptors[:-1]).mean(axis=(-2, -1))
    return out


def _adjacent_diff_full(
    frames_bgr: List[np.ndarray],
    block_rows: int,
    block_cols: int,
) -> np.ndarray:
    frame_count = len(frames_bgr)
    out = np.full((frame_count, block_rows, block_cols), np.inf, dtype=np.float32)
    if frame_count <= 1:
        return out
    height, width = frames_bgr[0].shape[:2]
    block_h = int(height // block_rows)
    block_w = int(width // block_cols)
    previous = _frame_gray(frames_bgr[0])
    for frame_idx in range(1, frame_count):
        current = _frame_gray(frames_bgr[frame_idx])
        diff = np.abs(current - previous)
        out[frame_idx] = diff.reshape(block_rows, block_h, block_cols, block_w).mean(axis=(1, 3))
        previous = current
    return out


def _candidate_is_duplicate(
    frame_idx: int,
    block_idx: int,
    selected_grid: np.ndarray,
    adjacent_diff_flat: np.ndarray,
    threshold: float,
) -> Tuple[bool, int]:
    comparisons = 0
    frame_count = int(selected_grid.shape[0])
    for neighbor in (int(frame_idx) - 1, int(frame_idx) + 1):
        if neighbor < 0 or neighbor >= frame_count:
            continue
        if not bool(selected_grid[neighbor, block_idx]):
            continue
        pair_idx = max(int(frame_idx), int(neighbor))
        comparisons += 1
        if float(adjacent_diff_flat[pair_idx, block_idx]) <= float(threshold):
            return True, comparisons
    return False, comparisons


def _select_from_order(
    order: np.ndarray,
    limit: int,
    selected_mask: np.ndarray,
    selected_grid: np.ndarray,
    frame_indices: np.ndarray,
    block_indices: np.ndarray,
    adjacent_diff_flat: np.ndarray,
    threshold: float,
    dedup_enabled: bool,
    rejected_mask: np.ndarray,
) -> Tuple[List[int], int, int]:
    selected: List[int] = []
    rejected = 0
    comparisons = 0
    for candidate_idx in order.tolist():
        if len(selected) >= int(limit):
            break
        if bool(selected_mask[candidate_idx]):
            continue
        frame_idx = int(frame_indices[candidate_idx])
        block_idx = int(block_indices[candidate_idx])
        if dedup_enabled:
            duplicate, candidate_comparisons = _candidate_is_duplicate(
                frame_idx=frame_idx,
                block_idx=block_idx,
                selected_grid=selected_grid,
                adjacent_diff_flat=adjacent_diff_flat,
                threshold=float(threshold),
            )
            comparisons += int(candidate_comparisons)
            if duplicate:
                rejected += 1
                rejected_mask[candidate_idx] = True
                continue
        selected_mask[candidate_idx] = True
        selected_grid[frame_idx, block_idx] = True
        selected.append(int(candidate_idx))
    return selected, int(rejected), int(comparisons)


def process_group_diverse_mixed_simple(
    group_idx: int,
    group_frame_ids: List[int],
    group_frames_bgr: List[np.ndarray],
    group_scores: List[np.ndarray],
    images_per_group: int,
    patch: int,
    block_size: int = 2,
    group_block_scores: Optional[List[np.ndarray]] = None,
    good_mask: Optional[List[bool]] = None,
    anchor_idx: int = 0,
    anchor_strategy: str = "fixed_anchor",
    diversity_fraction: float = DEFAULT_DIVERSITY_FRACTION,
    novelty_weight: float = DEFAULT_NOVELTY_WEIGHT,
    dedup_enabled: bool = True,
    dedup_descriptor: str = "pooled4",
    dedup_threshold: Optional[float] = None,
    dedup_threshold_mode: str = "absolute",
    dedup_quantile: float = 0.10,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    started = time.perf_counter()
    p = int(patch)
    b = int(max(1, int(block_size)))
    block_patch_count = int(b * b)
    diversity_fraction = float(diversity_fraction)
    novelty_weight = float(novelty_weight)
    dedup_enabled = bool(dedup_enabled)
    dedup_descriptor = str(dedup_descriptor).lower().strip()
    dedup_threshold_mode = str(dedup_threshold_mode).lower().strip()
    dedup_quantile = float(dedup_quantile)
    if not 0.0 <= diversity_fraction <= 1.0:
        raise ValueError("diversity_fraction must be between 0 and 1")
    if not 0.0 <= novelty_weight <= 1.0:
        raise ValueError("novelty_weight must be between 0 and 1")
    if dedup_descriptor not in DEFAULT_DEDUP_THRESHOLDS:
        raise ValueError(f"unsupported dedup descriptor: {dedup_descriptor}")
    if dedup_threshold_mode not in {"absolute", "group_quantile"}:
        raise ValueError(
            f"unsupported dedup threshold mode: {dedup_threshold_mode}"
        )
    if not 0.0 <= dedup_quantile <= 1.0:
        raise ValueError("dedup_quantile must be between 0 and 1")
    threshold = (
        float(DEFAULT_DEDUP_THRESHOLDS[dedup_descriptor])
        if dedup_threshold is None
        else float(dedup_threshold)
    )
    if threshold < 0.0:
        raise ValueError("dedup_threshold must be >= 0")
    if diversity_fraction == 0.0 and not dedup_enabled:
        from codec_selector.plugins.selectors.topk_2x2_bitcost import (
            process_group_topk_2x2,
        )

        result = process_group_topk_2x2(
            group_idx=group_idx,
            group_frame_ids=group_frame_ids,
            group_frames_bgr=group_frames_bgr,
            group_scores=group_scores,
            images_per_group=images_per_group,
            patch=patch,
            block_size=block_size,
            group_block_scores=group_block_scores,
            good_mask=good_mask,
            anchor_idx=anchor_idx,
            anchor_strategy=anchor_strategy,
        )
        result[0]["selector"] = {
            "mode": "diverse_mixed_simple",
            "control_path": "topk_2x2_bitcost",
            "bitcost_fraction": 1.0,
            "diversity_fraction": 0.0,
            "novelty_weight": float(novelty_weight),
            "edge_weight": float(1.0 - novelty_weight),
            "dedup_enabled": False,
            "dedup_descriptor": str(dedup_descriptor),
            "dedup_threshold": float(threshold),
            "dedup_threshold_mode": str(dedup_threshold_mode),
            "dedup_quantile": float(dedup_quantile),
        }
        return result
    h1, w1 = group_frames_bgr[0].shape[:2]
    hb, wb = h1 // p, w1 // p
    if hb % b != 0 or wb % b != 0:
        raise RuntimeError(f"group {group_idx}: grid not divisible by block_size={b}: hb={hb} wb={wb}")

    block_rows = int(hb // b)
    block_cols = int(wb // b)
    blocks_per_frame = int(block_rows * block_cols)
    s_full = int(hb * wb)
    pz = np.zeros((p, p, 3), dtype=np.uint8)
    keep_patch_mask = np.zeros((len(group_frames_bgr), hb, wb), dtype=np.uint8)

    if good_mask is None:
        good_mask = [not frame_is_bad(frame) for frame in group_frames_bgr]
    else:
        good_mask = [bool(value) for value in good_mask]
    anchor_idx = int(max(0, min(len(group_frames_bgr) - 1, int(anchor_idx))))
    if str(anchor_strategy) == "fixed_anchor" and anchor_idx == 0 and not good_mask[0]:
        for swap_idx in range(1, len(good_mask)):
            if good_mask[swap_idx]:
                group_frames_bgr[0], group_frames_bgr[swap_idx] = group_frames_bgr[swap_idx], group_frames_bgr[0]
                group_frame_ids[0], group_frame_ids[swap_idx] = group_frame_ids[swap_idx], group_frame_ids[0]
                group_scores[0], group_scores[swap_idx] = group_scores[swap_idx], group_scores[0]
                if group_block_scores is not None:
                    group_block_scores[0], group_block_scores[swap_idx] = group_block_scores[swap_idx], group_block_scores[0]
                good_mask[0], good_mask[swap_idx] = good_mask[swap_idx], good_mask[0]
                break
    elif not good_mask[anchor_idx]:
        for swap_idx in range(len(good_mask)):
            if swap_idx != anchor_idx and good_mask[swap_idx]:
                anchor_idx = int(swap_idx)
                break

    total_patches = int(images_per_group) * int(s_full)
    block_budget = max(0, int((total_patches - s_full) // block_patch_count))
    patches_list: List[np.ndarray] = []
    src_pos_list: List[List[int]] = []
    src_fid_list: List[int] = []

    anchor_frame = group_frames_bgr[anchor_idx]
    anchor_fid = int(group_frame_ids[anchor_idx])
    for block_h, block_w in iter_blocks_in_raster(hb, wb, block_size=b):
        for patch_h, patch_w in block_to_patches(block_h, block_w, block_size=b):
            patches_list.append(extract_patch_bgr(anchor_frame, patch_h, patch_w, patch=p))
            src_pos_list.append([anchor_fid, int(patch_h), int(patch_w)])
            src_fid_list.append(anchor_fid)
            keep_patch_mask[anchor_idx, int(patch_h), int(patch_w)] = 1

    descriptor_started = time.perf_counter()
    descriptors = _pooled4_descriptors(group_frames_bgr, block_rows=block_rows, block_cols=block_cols)
    edge_maps = _edge_scores(descriptors)
    descriptor_sec = float(time.perf_counter() - descriptor_started)

    scores_parts: List[np.ndarray] = []
    novelty_parts: List[np.ndarray] = []
    edge_parts: List[np.ndarray] = []
    frame_parts: List[np.ndarray] = []
    block_parts: List[np.ndarray] = []
    anchor_descriptors = descriptors[anchor_idx]
    for frame_idx in range(len(group_frames_bgr)):
        if frame_idx == anchor_idx or not good_mask[frame_idx]:
            continue
        if group_block_scores is not None:
            bitcost = np.asarray(group_block_scores[frame_idx], dtype=np.float32).reshape(-1)
        else:
            patch_scores = score_map_to_patch_scores(group_scores[frame_idx], patch=p)
            bitcost = patch_scores_to_block_scores(patch_scores, block_size=b).reshape(-1).astype(np.float32)
        novelty = np.abs(descriptors[frame_idx] - anchor_descriptors).mean(axis=(-2, -1)).reshape(-1)
        edge = edge_maps[frame_idx].reshape(-1)
        scores_parts.append(bitcost)
        novelty_parts.append(novelty.astype(np.float32))
        edge_parts.append(edge.astype(np.float32))
        frame_parts.append(np.full((blocks_per_frame,), int(frame_idx), dtype=np.int32))
        block_parts.append(np.arange(blocks_per_frame, dtype=np.int32))

    selected_blocks_by_frame: Dict[int, List[Tuple[int, int]]] = {}
    selected_scores_by_frame: Dict[int, List[float]] = {}
    selector_debug: Dict[str, Any] = {
        "mode": "diverse_mixed_simple",
        "bitcost_fraction": float(1.0 - diversity_fraction),
        "diversity_fraction": float(diversity_fraction),
        "novelty_weight": float(novelty_weight),
        "edge_weight": float(1.0 - novelty_weight),
        "dedup_enabled": bool(dedup_enabled),
        "dedup_descriptor": str(dedup_descriptor),
        "dedup_threshold": float(threshold),
        "dedup_threshold_configured": float(threshold),
        "dedup_threshold_mode": str(dedup_threshold_mode),
        "dedup_quantile": float(dedup_quantile),
        "dedup_threshold_fallback": False,
        "candidate_blocks": 0,
        "target_blocks": 0,
        "bitcost_selected": 0,
        "diversity_selected": 0,
        "backfill_selected": 0,
        "dedup_rejected": 0,
        "dedup_rejected_unique": 0,
        "dedup_comparisons": 0,
        "unique_source_frames": 0,
        "unique_spatial_positions": 0,
        "temporal_distribution_entropy": 0.0,
        "temporal_distribution_entropy_normalized": 0.0,
        "max_blocks_per_frame_fraction": 0.0,
        "adjacent_same_position_pairs": 0,
        "adjacent_same_position_duplicates": 0,
        "adjacent_same_position_duplicate_rate": 0.0,
        "adjacent_mad_count": 0,
        "adjacent_mad_quantiles": {},
        "adjacent_mad_cdf": {},
        "adjacent_mad_fraction_le_threshold": 0.0,
        "selected_bitcost_mean": 0.0,
        "selected_novelty_mean": 0.0,
        "selected_edge_mean": 0.0,
    }
    selected_score_sum = 0.0
    selected_score_mean = 0.0

    if scores_parts and block_budget > 0:
        score_started = time.perf_counter()
        bitcost_scores = np.concatenate(scores_parts)
        novelty_scores = np.concatenate(novelty_parts)
        edge_scores = np.concatenate(edge_parts)
        frame_indices = np.concatenate(frame_parts)
        block_indices = np.concatenate(block_parts)
        diversity_scores = (
            novelty_weight * _rank01(novelty_scores)
            + (1.0 - novelty_weight) * _rank01(edge_scores)
        ).astype(np.float32)
        bitcost_order = _score_order(bitcost_scores, frame_indices, block_indices)
        diversity_order = _score_order(diversity_scores, frame_indices, block_indices)
        score_sec = float(time.perf_counter() - score_started)

        dedup_started = time.perf_counter()
        descriptor_mode = str(dedup_descriptor)
        if not dedup_enabled or descriptor_mode == "pooled4":
            adjacent_diff = _adjacent_diff_pooled4(descriptors)
        elif descriptor_mode == "full":
            adjacent_diff = _adjacent_diff_full(
                group_frames_bgr,
                block_rows=block_rows,
                block_cols=block_cols,
            )
        else:
            raise ValueError(f"unsupported dedup descriptor: {descriptor_mode}")
        dedup_sec = float(time.perf_counter() - dedup_started)
        adjacent_diff_flat = adjacent_diff.reshape(len(group_frames_bgr), blocks_per_frame)
        valid_adjacent_parts = [
            adjacent_diff_flat[frame_idx]
            for frame_idx in range(1, len(group_frames_bgr))
            if bool(good_mask[frame_idx - 1]) and bool(good_mask[frame_idx])
        ]
        if valid_adjacent_parts:
            valid_adjacent_mad = np.concatenate(valid_adjacent_parts).astype(
                np.float32,
                copy=False,
            )
            if dedup_threshold_mode == "group_quantile":
                threshold = float(
                    np.quantile(valid_adjacent_mad, float(dedup_quantile))
                )
            quantile_values = np.quantile(
                valid_adjacent_mad,
                (0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95),
            )
            selector_debug.update({
                "adjacent_mad_count": int(valid_adjacent_mad.size),
                "adjacent_mad_quantiles": {
                    name: float(value)
                    for name, value in zip(
                        ("p05", "p10", "p20", "p50", "p80", "p90", "p95"),
                        quantile_values.tolist(),
                    )
                },
                "adjacent_mad_cdf": {
                    f"{cdf_threshold:.3f}": float(
                        np.mean(valid_adjacent_mad <= float(cdf_threshold))
                    )
                    for cdf_threshold in ADJACENT_MAD_CDF_THRESHOLDS
                },
                "adjacent_mad_fraction_le_threshold": float(
                    np.mean(valid_adjacent_mad <= float(threshold))
                ),
                "dedup_threshold": float(threshold),
            })
        elif dedup_threshold_mode == "group_quantile":
            selector_debug["dedup_threshold_fallback"] = True

        selection_started = time.perf_counter()
        target_blocks = min(int(block_budget), int(bitcost_scores.size))
        diversity_quota = int(np.floor(float(target_blocks) * diversity_fraction + 0.5))
        bitcost_quota = int(target_blocks - diversity_quota)
        selected_mask = np.zeros((bitcost_scores.size,), dtype=bool)
        rejected_mask = np.zeros((bitcost_scores.size,), dtype=bool)
        selected_grid = np.zeros((len(group_frames_bgr), blocks_per_frame), dtype=bool)
        selected_grid[anchor_idx, :] = True

        bitcost_selected, rejected_a, comparisons_a = _select_from_order(
            order=bitcost_order,
            limit=bitcost_quota,
            selected_mask=selected_mask,
            selected_grid=selected_grid,
            frame_indices=frame_indices,
            block_indices=block_indices,
            adjacent_diff_flat=adjacent_diff_flat,
            threshold=threshold,
            dedup_enabled=dedup_enabled,
            rejected_mask=rejected_mask,
        )
        diversity_selected: List[int] = []
        rejected_b = 0
        comparisons_b = 0
        if diversity_quota > 0:
            diversity_selected, rejected_b, comparisons_b = _select_from_order(
                order=diversity_order,
                limit=max(0, target_blocks - len(bitcost_selected)),
                selected_mask=selected_mask,
                selected_grid=selected_grid,
                frame_indices=frame_indices,
                block_indices=block_indices,
                adjacent_diff_flat=adjacent_diff_flat,
                threshold=threshold,
                dedup_enabled=dedup_enabled,
                rejected_mask=rejected_mask,
            )
        selected_indices = list(bitcost_selected) + list(diversity_selected)
        backfill_selected: List[int] = []
        if len(selected_indices) < target_blocks:
            for candidate_idx in bitcost_order.tolist():
                if len(selected_indices) >= target_blocks:
                    break
                if bool(selected_mask[candidate_idx]):
                    continue
                selected_mask[candidate_idx] = True
                frame_idx = int(frame_indices[candidate_idx])
                block_idx = int(block_indices[candidate_idx])
                selected_grid[frame_idx, block_idx] = True
                selected_indices.append(int(candidate_idx))
                backfill_selected.append(int(candidate_idx))
        selection_sec = float(time.perf_counter() - selection_started)

        selected_array = np.asarray(selected_indices, dtype=np.int64)
        if selected_array.size:
            selected_score_sum = float(bitcost_scores[selected_array].sum())
            selected_score_mean = float(bitcost_scores[selected_array].mean())
            selected_frame_indices = frame_indices[selected_array]
            selected_block_indices = block_indices[selected_array]
            frame_counts = np.bincount(
                selected_frame_indices,
                minlength=len(group_frames_bgr),
            ).astype(np.int64)
            nonzero_frame_counts = frame_counts[frame_counts > 0]
            probabilities = nonzero_frame_counts.astype(np.float64) / float(selected_array.size)
            temporal_entropy = float(-(probabilities * np.log(probabilities)).sum())
            eligible_frame_count = int(
                sum(
                    1
                    for frame_idx, is_good in enumerate(good_mask)
                    if frame_idx != anchor_idx and bool(is_good)
                )
            )
            entropy_denominator = float(np.log(max(1, eligible_frame_count)))
            normalized_entropy = (
                float(temporal_entropy / entropy_denominator)
                if entropy_denominator > 0.0
                else 0.0
            )
            selected_grid_for_metrics = np.zeros(
                (len(group_frames_bgr), blocks_per_frame),
                dtype=bool,
            )
            selected_grid_for_metrics[anchor_idx, :] = True
            selected_grid_for_metrics[selected_frame_indices, selected_block_indices] = True
            adjacent_pair_count = 0
            adjacent_duplicate_count = 0
            for frame_idx in range(1, len(group_frames_bgr)):
                both_selected = (
                    selected_grid_for_metrics[frame_idx - 1]
                    & selected_grid_for_metrics[frame_idx]
                )
                pair_count = int(both_selected.sum())
                if pair_count == 0:
                    continue
                adjacent_pair_count += pair_count
                adjacent_duplicate_count += int(
                    (
                        adjacent_diff_flat[frame_idx, both_selected]
                        <= float(threshold)
                    ).sum()
                )
            source_frame_ids = sorted(
                {
                    int(group_frame_ids[int(frame_idx)])
                    for frame_idx in selected_frame_indices.tolist()
                }
            )
            selector_debug.update({
                "unique_source_frames": int(len(source_frame_ids)),
                "selected_source_frame_ids": source_frame_ids,
                "unique_spatial_positions": int(np.unique(selected_block_indices).size),
                "temporal_distribution_entropy": float(temporal_entropy),
                "temporal_distribution_entropy_normalized": float(normalized_entropy),
                "max_blocks_per_frame_fraction": float(nonzero_frame_counts.max())
                / float(selected_array.size),
                "adjacent_same_position_pairs": int(adjacent_pair_count),
                "adjacent_same_position_duplicates": int(adjacent_duplicate_count),
                "adjacent_same_position_duplicate_rate": (
                    float(adjacent_duplicate_count) / float(adjacent_pair_count)
                    if adjacent_pair_count
                    else 0.0
                ),
                "selected_bitcost_mean": float(bitcost_scores[selected_array].mean()),
                "selected_novelty_mean": float(novelty_scores[selected_array].mean()),
                "selected_edge_mean": float(edge_scores[selected_array].mean()),
            })
        for candidate_idx in selected_indices:
            frame_idx = int(frame_indices[candidate_idx])
            block_idx = int(block_indices[candidate_idx])
            block_h = int(block_idx // block_cols)
            block_w = int(block_idx % block_cols)
            selected_blocks_by_frame.setdefault(frame_idx, []).append((block_h, block_w))
            selected_scores_by_frame.setdefault(frame_idx, []).append(float(bitcost_scores[candidate_idx]))
        for frame_idx in selected_blocks_by_frame:
            selected_blocks_by_frame[frame_idx].sort(key=lambda item: (item[0], item[1]))

        selector_debug.update({
            "candidate_blocks": int(bitcost_scores.size),
            "target_blocks": int(target_blocks),
            "bitcost_quota": int(bitcost_quota),
            "diversity_quota": int(diversity_quota),
            "bitcost_selected": int(len(bitcost_selected)),
            "diversity_selected": int(len(diversity_selected)),
            "backfill_selected": int(len(backfill_selected)),
            "dedup_rejected": int(rejected_a + rejected_b),
            "dedup_rejected_unique": int(rejected_mask.sum()),
            "dedup_comparisons": int(comparisons_a + comparisons_b),
            "timing_sec": {
                "descriptor": float(descriptor_sec),
                "score": float(score_sec),
                "dedup_map": float(dedup_sec),
                "selection": float(selection_sec),
            },
        })
    else:
        selector_debug["timing_sec"] = {
            "descriptor": float(descriptor_sec),
            "score": 0.0,
            "dedup_map": 0.0,
            "selection": 0.0,
        }

    for frame_idx in range(len(group_frames_bgr)):
        if frame_idx == anchor_idx or not good_mask[frame_idx]:
            continue
        frame = group_frames_bgr[frame_idx]
        frame_id = int(group_frame_ids[frame_idx])
        for block_h, block_w in selected_blocks_by_frame.get(frame_idx, []):
            for patch_h, patch_w in block_to_patches(block_h, block_w, block_size=b):
                patches_list.append(extract_patch_bgr(frame, patch_h, patch_w, patch=p))
                src_pos_list.append([frame_id, int(patch_h), int(patch_w)])
                src_fid_list.append(frame_id)
                keep_patch_mask[frame_idx, int(patch_h), int(patch_w)] = 1

    n_raw = len(patches_list)
    if n_raw % block_patch_count != 0:
        raise RuntimeError(f"group {group_idx}: patches count {n_raw} not multiple of {block_patch_count}")

    valid_blocks = []
    for block_idx in range(n_raw // block_patch_count):
        start_idx = int(block_idx * block_patch_count)
        end_idx = int(start_idx + block_patch_count)
        block_patches = patches_list[start_idx:end_idx]
        block_positions = src_pos_list[start_idx:end_idx]
        block_frame_ids = src_fid_list[start_idx:end_idx]
        frame_id = block_positions[0][0]
        if frame_id >= 0:
            is_anchor_block = start_idx < s_full
            sort_key = (
                0 if is_anchor_block else 1,
                int(frame_id),
                int(block_positions[0][1] // b),
                int(block_positions[0][2] // b),
            )
            valid_blocks.append((sort_key, block_patches, block_positions, block_frame_ids))
    valid_blocks.sort(key=lambda item: item[0])

    patches_list = []
    src_pos_list = []
    src_fid_list = []
    for _, block_patches, block_positions, block_frame_ids in valid_blocks:
        pos_in_canvas = len(patches_list) % s_full
        needed = (block_patch_count - (pos_in_canvas % block_patch_count)) % block_patch_count
        for _ in range(needed):
            patches_list.append(pz)
            src_pos_list.append([-1, -1, -1])
            src_fid_list.append(-1)
        patches_list.extend(block_patches)
        src_pos_list.extend(block_positions)
        src_fid_list.extend(block_frame_ids)

    while len(patches_list) < total_patches:
        patches_list.append(pz)
        src_pos_list.append([-1, -1, -1])
        src_fid_list.append(-1)
    patches_list = patches_list[:total_patches]
    src_pos_list = src_pos_list[:total_patches]
    src_fid_list = src_fid_list[:total_patches]

    patches = np.stack(patches_list, axis=0).astype(np.uint8)
    images_rgb, patch_pos, img_ptr = pack_patches_to_canvases(
        patches,
        hb=hb,
        wb=wb,
        patch=p,
        block_size=b,
    )
    selector_debug["timing_sec"]["total"] = float(time.perf_counter() - started)
    meta = {
        "group_idx": int(group_idx),
        "frame_ids": [int(value) for value in group_frame_ids],
        "images_per_group": int(images_per_group),
        "patch": int(p),
        "block_size": int(b),
        "block_patch_count": int(block_patch_count),
        "grid_hw": [int(hb), int(wb)],
        "total_patches": int(len(src_pos_list)),
        "selected_blocks": int(max(0, (len(src_fid_list) - s_full) // block_patch_count)),
        "selected_block_score_sum": float(selected_score_sum),
        "selected_block_score_mean": float(selected_score_mean),
        "selected_block_score_density": float(selected_score_sum) / float(max(1, len(group_frame_ids))),
        "selected_blocks_per_frame": {
            str(int(frame_idx)): int(len(blocks))
            for frame_idx, blocks in selected_blocks_by_frame.items()
        },
        "selected_block_score_sum_per_frame": {
            str(int(frame_idx)): float(sum(selected_scores_by_frame.get(frame_idx, [])))
            for frame_idx in selected_blocks_by_frame
        },
        "selector": selector_debug,
    }
    if str(anchor_strategy) == "adaptive_anchor":
        meta.update({
            "group_id": int(group_idx),
            "anchor_frame_id": int(anchor_fid),
            "anchor_idx": int(anchor_idx),
            "anchor_strategy": str(anchor_strategy),
        })
    return (
        meta,
        keep_patch_mask.astype(np.uint8),
        images_rgb.astype(np.uint8),
        patch_pos.astype(np.int32),
        np.asarray(src_pos_list, dtype=np.int32),
        img_ptr.astype(np.int32),
    )


selectors.register("diverse_mixed_simple", process_group_diverse_mixed_simple)
