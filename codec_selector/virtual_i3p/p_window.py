#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P-window bitcost processing with group-complete selection and temporal accumulation."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2

from codec_selector.codec_patch_gop.video_processor import (
    cv_reader_fetch_bitcost,
    bitcost_item_to_score_map,
)
from codec_selector.codec_patch_gop.scoring import (
    score_map_to_patch_scores,
    patch_scores_to_block_scores,
)
from codec_selector.codec_patch_gop.patch_utils import (
    iter_blocks_in_raster,
    block_to_patches,
    extract_patch_bgr,
)
from codec_selector.codec_patch_gop.frame_utils import (
    frame_is_bad,
    _resize_bgr,
    pad_to_multiple_of_bgr,
)


def _normalize_bitcost_scores(
    score_map: np.ndarray,
    p1: float = 1.0,
    p99: float = 99.0,
) -> np.ndarray:
    """Robust percentile-based normalization of bitcost scores to [0, 1]."""
    arr = np.asarray(score_map, dtype=np.float32)
    low = float(np.percentile(arr, p1))
    high = float(np.percentile(arr, p99))
    eps = 1e-6
    norm = np.clip((arr - low) / (high - low + eps), 0.0, 1.0)
    return norm.astype(np.float32)


def compute_group_score_map(
    bitcost_item: Dict[str, Any],
    out_h: int,
    out_w: int,
    patch_size: int,
    group_size: int,
    bitcost_grid: str,
    bitcost_pct: float,
    bitcost_log_scale: bool,
    codec_name: str,
    aggregation: str = "max",
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert bitcost item to patch scores and aggregate into group scores.

    Args:
        bitcost_item: Bitcost data from cv_reader.
        out_h, out_w: Output dimensions (padded).
        patch_size: Patch size.
        group_size: Group size (e.g. 2 for 2x2).
        ... (other params)
        aggregation: "max" or "mean".

    Returns:
        (patch_score_map (hb, wb), group_score_map (gh, gw))
    """
    score_map = bitcost_item_to_score_map(
        bitcost_item,
        out_h=int(out_h),
        out_w=int(out_w),
        grid=str(bitcost_grid),
        pct=float(bitcost_pct),
        log_scale=bool(bitcost_log_scale),
        codec_name=str(codec_name),
    )
    # Robust normalization
    score_map = _normalize_bitcost_scores(score_map)

    # Convert to patch scores
    patch_scores = score_map_to_patch_scores(score_map, patch=int(patch_size))
    hb, wb = patch_scores.shape

    # Aggregate to group scores
    b = int(max(1, int(group_size)))
    if hb % b != 0 or wb % b != 0:
        # Pad patch score map if needed
        pad_h = (b - (hb % b)) % b
        pad_w = (b - (wb % b)) % b
        if pad_h > 0 or pad_w > 0:
            patch_scores = np.pad(patch_scores, ((0, pad_h), (0, pad_w)), mode="constant")
            hb, wb = patch_scores.shape

    # Reshape to blocks
    gh, gw = hb // b, wb // b
    blocks = patch_scores.reshape(gh, b, gw, b)

    agg = str(aggregation).lower().strip()
    if agg == "mean":
        group_scores = blocks.mean(axis=(1, 3))
    else:
        group_scores = blocks.max(axis=(1, 3))

    return patch_scores.astype(np.float32), group_scores.astype(np.float32)


def select_topk_groups(
    group_scores: np.ndarray,
    k: int,
    exclude_mask: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, float]]:
    """Select top-k groups from group score map.

    Returns:
        List of (group_h, group_w, score) sorted by score desc.
    """
    gh, gw = group_scores.shape
    flat = group_scores.reshape(-1).astype(np.float32)

    if exclude_mask is not None:
        flat = flat.copy()
        flat[exclude_mask.reshape(-1)] = -1.0

    if flat.size == 0:
        return []

    k = min(int(k), flat.size)
    if k <= 0:
        return []

    # Top-k via argpartition for efficiency
    if k < flat.size:
        topk_idx = np.argpartition(-flat, kth=k - 1)[:k]
    else:
        topk_idx = np.arange(flat.size)

    # Sort by score descending
    topk_idx = topk_idx[np.argsort(-flat[topk_idx])]

    results: List[Tuple[int, int, float]] = []
    for idx in topk_idx.tolist():
        if flat[idx] < 0:
            continue
        gh_idx = int(idx // gw)
        gw_idx = int(idx % gw)
        results.append((gh_idx, gw_idx, float(flat[idx])))

    return results


def _get_neighbor_coords(gh: int, gw: int, gh_max: int, gw_max: int) -> List[Tuple[int, int]]:
    """Get valid 4-connected neighbor coordinates."""
    neighbors = []
    for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nh, nw = gh + dh, gw + dw
        if 0 <= nh < gh_max and 0 <= nw < gw_max:
            neighbors.append((nh, nw))
    return neighbors


def _fill_with_connectivity(
    selected_set: set,
    selected: List[Tuple[int, int, int, float]],
    acc_scores: np.ndarray,
    group_best_frame: Dict[Tuple[int, int], int],
    min_groups: int,
    default_fid: int,
) -> Tuple[set, List[Tuple[int, int, int, float]]]:
    """Fill selected groups up to min_groups, preferring neighbors of already-selected groups.

    This reduces fragmentation by growing selected regions outward from existing selections.
    """
    gh, gw = acc_scores.shape
    if len(selected_set) >= min_groups:
        return selected_set, selected

    need = min_groups - len(selected_set)

    # Iteratively add neighbor groups with highest scores
    for _ in range(need):
        # Find all unselected neighbors of selected groups
        candidate_neighbors: Dict[Tuple[int, int], float] = {}
        for (sg_h, sg_w) in selected_set:
            for nh, nw in _get_neighbor_coords(sg_h, sg_w, gh, gw):
                if (nh, nw) not in selected_set:
                    candidate_neighbors[(nh, nw)] = acc_scores[nh, nw]

        if candidate_neighbors:
            # Pick highest-scoring neighbor
            best_key = max(candidate_neighbors.keys(), key=lambda k: candidate_neighbors[k])
            best_score = float(candidate_neighbors[best_key])
        else:
            # No neighbors left, pick highest globally
            flat = acc_scores.reshape(-1).copy()
            for (sh, sw) in selected_set:
                flat[sh * gw + sw] = -1.0
            best_idx = int(np.argmax(flat))
            if flat[best_idx] < 0:
                break  # All groups selected
            best_key = (best_idx // gw, best_idx % gw)
            best_score = float(flat[best_idx])

        selected_set.add(best_key)
        fid = group_best_frame.get(best_key, default_fid)
        selected.append((int(fid), best_key[0], best_key[1], best_score))

    return selected_set, selected


def temporal_accumulation_group_scores(
    group_scores_list: List[np.ndarray],
    decay: float = 0.9,
) -> np.ndarray:
    """Accumulate group scores over time with exponential decay.

    acc_t = decay * acc_{t-1} + score_t

    Returns the final accumulated score map.
    """
    if not group_scores_list:
        return np.zeros((0, 0), dtype=np.float32)

    acc = np.zeros_like(group_scores_list[0], dtype=np.float32)
    for gs in group_scores_list:
        acc = float(decay) * acc + gs
    return acc.astype(np.float32)


def temporal_accumulation_with_tracking(
    group_scores_list: List[np.ndarray],
    frame_ids: List[int],
    decay: float = 0.9,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """Accumulate group scores and track which frame contributed max for each group.

    Returns:
        (accumulated_score_map, group_best_frame: dict mapping (gh, gw) -> frame_id)
    """
    if not group_scores_list:
        return np.zeros((0, 0), dtype=np.float32), {}

    acc = np.zeros_like(group_scores_list[0], dtype=np.float32)
    # Track best frame for each group
    best_frame: Dict[Tuple[int, int], int] = {}

    for i, gs in enumerate(group_scores_list):
        fid = int(frame_ids[i])
        # Update accumulator
        acc = float(decay) * acc + gs
        # Track which frame had the highest contribution so far
        for gh in range(gs.shape[0]):
            for gw in range(gs.shape[1]):
                key = (int(gh), int(gw))
                if key not in best_frame:
                    best_frame[key] = fid
                else:
                    # Simple heuristic: if current score is higher than what we've seen, update
                    # We approximate by comparing the "increment" gs[gh, gw]
                    if gs[gh, gw] > acc[gh, gw] * (1.0 - decay) + 1e-6:
                        best_frame[key] = fid

    return acc.astype(np.float32), best_frame


def select_groups_temporal_balance(
    group_scores_list: List[np.ndarray],
    frame_ids: List[int],
    total_budget: int,
    num_buckets: int = 4,
    balance_ratio: float = 0.5,
    decay: float = 0.9,
    use_accumulation: bool = True,
) -> Tuple[List[Tuple[int, int, int, float]], Dict[Tuple[int, int], int]]:
    """Select groups with temporal balance.

    Strategy:
      1. If use_accumulation: accumulate scores across the window.
      2. Divide window into num_buckets sub-windows.
      3. Allocate balance_ratio * total_budget to per-bucket selection.
      4. Allocate remaining to global top-k.

    Returns:
        selected: List of (frame_id, group_h, group_w, score)
        group_best_frame: Dict mapping (gh, gw) -> frame_id
    """
    if not group_scores_list:
        return [], {}

    n_frames = len(group_scores_list)
    gh, gw = group_scores_list[0].shape

    # Step 1: accumulation
    if use_accumulation:
        acc_scores, group_best_frame = temporal_accumulation_with_tracking(
            group_scores_list, frame_ids, decay=float(decay)
        )
    else:
        # Max over time
        acc_scores = np.zeros((gh, gw), dtype=np.float32)
        group_best_frame: Dict[Tuple[int, int], int] = {}
        for i, gs in enumerate(group_scores_list):
            fid = int(frame_ids[i])
            for gh_idx in range(gh):
                for gw_idx in range(gw):
                    key = (gh_idx, gw_idx)
                    if gs[gh_idx, gw_idx] > acc_scores[gh_idx, gw_idx]:
                        acc_scores[gh_idx, gw_idx] = gs[gh_idx, gw_idx]
                        group_best_frame[key] = fid

    # Step 2: bucket division
    bucket_size = max(1, n_frames // num_buckets)
    per_bucket_budget = max(1, int(total_budget * balance_ratio) // num_buckets)
    global_budget = max(0, total_budget - per_bucket_budget * num_buckets)

    selected_set: set = set()
    selected: List[Tuple[int, int, int, float]] = []

    # Per-bucket selection
    for b in range(num_buckets):
        start = b * bucket_size
        end = min(n_frames, (b + 1) * bucket_size) if b < num_buckets - 1 else n_frames
        bucket_scores = [group_scores_list[i] for i in range(start, end)]
        if not bucket_scores:
            continue

        # Simple max over bucket
        bucket_max = np.zeros((gh, gw), dtype=np.float32)
        bucket_best_frame: Dict[Tuple[int, int], int] = {}
        for i, gs in enumerate(bucket_scores):
            fid = int(frame_ids[start + i])
            for gh_idx in range(gh):
                for gw_idx in range(gw):
                    if gs[gh_idx, gw_idx] > bucket_max[gh_idx, gw_idx]:
                        bucket_max[gh_idx, gw_idx] = gs[gh_idx, gw_idx]
                        bucket_best_frame[(gh_idx, gw_idx)] = fid

        flat = bucket_max.reshape(-1)
        k = min(per_bucket_budget, flat.size)
        if k > 0 and flat.size > 0:
            if k < flat.size:
                topk = np.argpartition(-flat, kth=k - 1)[:k]
            else:
                topk = np.arange(flat.size)
            topk = topk[np.argsort(-flat[topk])]
            for idx in topk.tolist():
                gh_idx = int(idx // gw)
                gw_idx = int(idx % gw)
                key = (gh_idx, gw_idx)
                if key not in selected_set:
                    selected_set.add(key)
                    fid = bucket_best_frame.get(key, int(frame_ids[start]))
                    selected.append((int(fid), gh_idx, gw_idx, float(flat[idx])))

    # Global selection with remaining budget
    global_exclude = np.zeros((gh, gw), dtype=bool)
    for gh_idx, gw_idx in selected_set:
        global_exclude[gh_idx, gw_idx] = True

    if global_budget > 0:
        global_topk = select_topk_groups(acc_scores, k=global_budget, exclude_mask=global_exclude)
        for gh_idx, gw_idx, score in global_topk:
            key = (gh_idx, gw_idx)
            if key not in selected_set:
                selected_set.add(key)
                fid = group_best_frame.get(key, int(frame_ids[0]))
                selected.append((int(fid), gh_idx, gw_idx, float(score)))

    # Sort by frame_id, then group position for consistent ordering
    selected.sort(key=lambda x: (int(x[0]), int(x[1]), int(x[2])))
    return selected, group_best_frame


def process_p_window_bitcost(
    video_path: str,
    window_frame_ids: List[int],
    out_h: int,
    out_w: int,
    patch_size: int,
    group_size: int,
    max_patches_per_p_image: int,
    bitcost_grid: str,
    bitcost_pct: float,
    bitcost_log_scale: bool,
    codec_name: str,
    use_temporal_accumulation: bool = True,
    decay: float = 0.9,
    use_temporal_balance: bool = True,
    temporal_balance_ratio: float = 0.5,
    num_buckets_per_p_window: int = 4,
    bitcost_items: Optional[List[Dict[str, Any]]] = None,
    resize_h: int = 0,
    resize_w: int = 0,
    pad_base: int = 0,
    decode_backend: str = "ffmpeg_native",
    min_patches_per_p_image: Optional[int] = None,
) -> Dict[str, Any]:
    """Process one P-window: fetch bitcost, select top groups, return selection metadata.

    Returns:
        Dict with keys:
          - "selected_patches": List[Dict] of {frame_id, patch_y, patch_x, score}
          - "group_best_frame": Dict mapping (gh, gw) -> frame_id
          - "window_score_stats": Dict with mean, max, min score
          - "num_frames": int
          - "num_selected_groups": int
    """
    if not window_frame_ids:
        return {
            "selected_patches": [],
            "group_best_frame": {},
            "window_score_stats": {"mean": 0.0, "max": 0.0, "min": 0.0},
            "num_frames": 0,
            "num_selected_groups": 0,
        }

    # Fetch bitcost if not provided
    if bitcost_items is None:
        bitcost_items = cv_reader_fetch_bitcost(
            str(video_path),
            [int(x) for x in window_frame_ids],
            bitcost_grid=str(bitcost_grid),
        )

    if len(bitcost_items) != len(window_frame_ids):
        raise RuntimeError(
            f"bitcost fetch mismatch: {len(bitcost_items)} vs {len(window_frame_ids)}"
        )

    # Compute group scores for each frame
    group_scores_list: List[np.ndarray] = []
    patch_scores_list: List[np.ndarray] = []

    for item in bitcost_items:
        patch_scores, group_scores = compute_group_score_map(
            item,
            out_h=int(out_h),
            out_w=int(out_w),
            patch_size=int(patch_size),
            group_size=int(group_size),
            bitcost_grid=str(bitcost_grid),
            bitcost_pct=float(bitcost_pct),
            bitcost_log_scale=bool(bitcost_log_scale),
            codec_name=str(codec_name),
            aggregation="max",
        )
        group_scores_list.append(group_scores)
        patch_scores_list.append(patch_scores)

    if not group_scores_list:
        return {
            "selected_patches": [],
            "group_best_frame": {},
            "window_score_stats": {"mean": 0.0, "max": 0.0, "min": 0.0},
            "num_frames": 0,
            "num_selected_groups": 0,
        }

    # Total budget in groups
    b = int(max(1, int(group_size)))
    patches_per_group = b * b
    total_group_budget = max_patches_per_p_image // patches_per_group
    if total_group_budget <= 0:
        total_group_budget = max(1, (out_h // patch_size) * (out_w // patch_size) // patches_per_group)

    # Minimum patch constraint
    if min_patches_per_p_image is not None and min_patches_per_p_image > 0:
        min_group_budget = int(min_patches_per_p_image) // patches_per_group
    else:
        # Default: at least 20% of canvas
        min_group_budget = max(1, total_group_budget // 5)

    # Select groups
    if use_temporal_balance:
        selected, group_best_frame = select_groups_temporal_balance(
            group_scores_list=group_scores_list,
            frame_ids=[int(x) for x in window_frame_ids],
            total_budget=int(total_group_budget),
            num_buckets=int(num_buckets_per_p_window),
            balance_ratio=float(temporal_balance_ratio),
            decay=float(decay),
            use_accumulation=bool(use_temporal_accumulation),
        )
        # Get accumulated scores for connectivity fill
        if use_temporal_accumulation:
            acc_scores = temporal_accumulation_group_scores(group_scores_list, decay=float(decay))
        else:
            acc_scores = np.max(np.stack(group_scores_list, axis=0), axis=0)
    else:
        # Simple global top-k on accumulated scores
        if use_temporal_accumulation:
            acc_scores, group_best_frame = temporal_accumulation_with_tracking(
                group_scores_list,
                [int(x) for x in window_frame_ids],
                decay=float(decay),
            )
        else:
            acc_scores = np.max(np.stack(group_scores_list, axis=0), axis=0)
            group_best_frame = {}
            for i, gs in enumerate(group_scores_list):
                fid = int(window_frame_ids[i])
                for gh_idx in range(gs.shape[0]):
                    for gw_idx in range(gs.shape[1]):
                        key = (gh_idx, gw_idx)
                        if key not in group_best_frame:
                            group_best_frame[key] = fid

        topk = select_topk_groups(acc_scores, k=total_group_budget)
        selected = []
        for gh_idx, gw_idx, score in topk:
            fid = group_best_frame.get((gh_idx, gw_idx), int(window_frame_ids[0]))
            selected.append((int(fid), gh_idx, gw_idx, float(score)))
        selected.sort(key=lambda x: (int(x[0]), int(x[1]), int(x[2])))

    # Apply minimum patch constraint with connectivity-aware fill
    selected_set = set((int(s[1]), int(s[2])) for s in selected)
    if len(selected_set) < min_group_budget:
        default_fid = int(window_frame_ids[0]) if window_frame_ids else 0
        selected_set, selected = _fill_with_connectivity(
            selected_set=selected_set,
            selected=selected,
            acc_scores=acc_scores,
            group_best_frame=group_best_frame,
            min_groups=int(min_group_budget),
            default_fid=default_fid,
        )
        selected.sort(key=lambda x: (int(x[0]), int(x[1]), int(x[2])))

    # Expand groups to individual patches
    b = int(max(1, int(group_size)))
    selected_patches: List[Dict[str, Any]] = []

    for fid, gh, gw, score in selected:
        # Each group expands to b*b patches
        for dh in range(b):
            for dw in range(b):
                ph = gh * b + dh
                pw = gw * b + dw
                selected_patches.append({
                    "frame_id": int(fid),
                    "patch_y": int(ph),
                    "patch_x": int(pw),
                    "group_h": int(gh),
                    "group_w": int(gw),
                    "score": float(score),
                })

    # Score stats from accumulated scores
    if use_temporal_accumulation:
        acc_scores = temporal_accumulation_group_scores(group_scores_list, decay=float(decay))
    else:
        acc_scores = np.max(np.stack(group_scores_list, axis=0), axis=0)

    return {
        "selected_patches": selected_patches,
        "group_best_frame": {f"{k[0]}_{k[1]}": int(v) for k, v in group_best_frame.items()},
        "window_score_stats": {
            "mean": float(acc_scores.mean()),
            "max": float(acc_scores.max()),
            "min": float(acc_scores.min()),
        },
        "num_frames": len(window_frame_ids),
        "num_selected_groups": len(selected),
        "group_scores_list": group_scores_list,  # For debug visualization
    }
