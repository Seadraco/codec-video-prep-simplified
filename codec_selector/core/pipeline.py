#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Composable bitcost readiness pipeline."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from codec_selector.core.config import BitcostReadinessConfig, GroupSpec, PipelineResult
from codec_selector.core.decode import decode_frames_bgr_ffmpeg
from codec_selector.core.frame_ops import normalize_score_maps_frame_mean_floor, prepare_frames, resolve_prepared_frame_geometry
from codec_selector.core.probe import probe_video
from codec_selector.plugins.groupers.fixed import build_fixed_groups
from codec_selector.plugins.groupers.readiness import (
    build_readiness_groups,
    estimate_readiness_threshold,
    precompute_frame_block_scores,
    resolve_group_frame_limits,
)
from codec_selector.plugins.packers.collage import save_canvases, save_mask_video
from codec_selector.plugins.samplers.basic import sample_frames
from codec_selector.plugins.scorers.bitcost import bitcost_items_to_score_maps
from codec_selector.plugins.selectors.diverse_mixed_simple import process_group_diverse_mixed_simple
from codec_selector.plugins.selectors.topk_2x2_bitcost import process_group_topk_2x2
from codec_selector.codec_patch_gop.utils import ensure_dir, sha1_8
from codec_selector.codec_patch_gop.video_probe import ffprobe_keyframe_frame_ids, mp4_keyframe_frame_ids
from codec_selector.codec_patch_gop.frame_utils import frame_is_bad
from codec_selector.codec_patch_gop.video_processor import cv_reader_fetch_bitcost


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _select_adaptive_anchor(
    group_scores: List[np.ndarray],
    good_mask: List[bool],
    w_center: float,
    w_low_bc: float,
) -> Tuple[int, Dict[str, Any]]:
    n = int(len(group_scores))
    if n <= 0:
        return 0, {"center_rank": 1, "bitcost_rank": 1}

    center = float(n - 1) / 2.0
    max_dist = max(center, float(n - 1) - center, 1.0)
    center_scores = np.asarray(
        [1.0 - (abs(float(i) - center) / max_dist) for i in range(n)],
        dtype=np.float32,
    )
    bitcost_totals = np.asarray([float(np.asarray(score).sum()) for score in group_scores], dtype=np.float64)
    bc_min = float(bitcost_totals.min()) if bitcost_totals.size else 0.0
    bc_max = float(bitcost_totals.max()) if bitcost_totals.size else 0.0
    if bc_max > bc_min:
        low_bitcost_scores = 1.0 - ((bitcost_totals - bc_min) / (bc_max - bc_min))
    else:
        low_bitcost_scores = np.ones((n,), dtype=np.float64)

    scores = (float(w_center) * center_scores.astype(np.float64)) + (
        float(w_low_bc) * low_bitcost_scores.astype(np.float64)
    )
    for i, is_good in enumerate(good_mask):
        if not bool(is_good):
            scores[int(i)] = -np.inf
    if not np.isfinite(scores).any():
        scores = (float(w_center) * center_scores.astype(np.float64)) + (
            float(w_low_bc) * low_bitcost_scores.astype(np.float64)
        )

    anchor_idx = int(np.argmax(scores))
    center_order = np.argsort(-center_scores)
    bitcost_order = np.argsort(bitcost_totals)
    center_rank = int(np.where(center_order == anchor_idx)[0][0]) + 1
    bitcost_rank = int(np.where(bitcost_order == anchor_idx)[0][0]) + 1
    debug = {
        "center_rank": int(center_rank),
        "bitcost_rank": int(bitcost_rank),
        "score": float(scores[anchor_idx]),
        "center_score": float(center_scores[anchor_idx]),
        "low_bitcost_score": float(low_bitcost_scores[anchor_idx]),
        "bitcost_total": float(bitcost_totals[anchor_idx]),
        "weights": {
            "center": float(w_center),
            "low_bitcost": float(w_low_bc),
        },
    }
    return anchor_idx, debug


def _build_adaptive_gop_groups(
    frame_ids: List[int],
    score_maps: List[np.ndarray],
    min_group_frames: int,
    max_group_frames: int,
    gamma: float,
    percentile: float,
) -> Tuple[List[GroupSpec], Dict[str, Any]]:
    costs = np.asarray([float(np.asarray(score).sum()) for score in score_maps], dtype=np.float64)
    if costs.size == 0:
        return [], {"threshold": 0.0, "costs": []}
    threshold = float(gamma) if float(gamma) > 0.0 else float(np.percentile(costs, float(percentile)))
    min_len = max(1, int(min_group_frames))
    max_len = max(min_len, int(max_group_frames))
    groups: List[GroupSpec] = []
    start = 0
    split_points: List[int] = []
    for i in range(1, len(frame_ids)):
        cur_len = int(i - start)
        hit_cost = bool(float(costs[i]) > threshold and cur_len >= min_len)
        hit_max = bool(cur_len >= max_len)
        if hit_cost or hit_max:
            reason = "adaptive_gop_cost" if hit_cost else "adaptive_gop_max_group_frames"
            groups.append(
                GroupSpec(
                    group_idx=int(len(groups)),
                    start=int(start),
                    end=int(i),
                    frame_count=int(i - start),
                    stop_reason=reason,
                    readiness={
                        "adaptive_gop_threshold": float(threshold),
                        "split_cost": float(costs[i]),
                        "split_frame_id": int(frame_ids[i]),
                    },
                )
            )
            split_points.append(int(i))
            start = int(i)
    if start < len(frame_ids):
        groups.append(
            GroupSpec(
                group_idx=int(len(groups)),
                start=int(start),
                end=int(len(frame_ids)),
                frame_count=int(len(frame_ids) - start),
                stop_reason="adaptive_gop_tail",
                readiness={"adaptive_gop_threshold": float(threshold)},
            )
        )
    debug = {
        "threshold": float(threshold),
        "gamma": float(gamma),
        "percentile": float(percentile),
        "cost_min": float(costs.min()),
        "cost_mean": float(costs.mean()),
        "cost_max": float(costs.max()),
        "split_points": split_points,
    }
    return groups, debug


def _merge_adaptive_gop_groups_to_target(
    groups: List[GroupSpec],
    target_groups: int,
) -> Tuple[List[GroupSpec], Dict[str, Any]]:
    target = max(1, int(target_groups))
    work = [
        GroupSpec(
            group_idx=int(i),
            start=int(g.start),
            end=int(g.end),
            frame_count=int(g.end - g.start),
            stop_reason=str(g.stop_reason),
            readiness=dict(g.readiness),
        )
        for i, g in enumerate(groups)
    ]
    merge_records: List[Dict[str, Any]] = []
    while len(work) > target:
        best_i = 0
        best_cost = float("inf")
        for i in range(len(work) - 1):
            cost = float(work[i].readiness.get("split_cost", 0.0))
            if cost < best_cost:
                best_i = int(i)
                best_cost = float(cost)
        left = work[best_i]
        right = work[best_i + 1]
        merged_readiness = dict(left.readiness)
        merged_readiness.update({
            "merged_adaptive_gop": True,
            "merged_stop_reasons": [
                str(left.stop_reason),
                str(right.stop_reason),
            ],
            "merged_boundary_cost": float(best_cost),
        })
        merged = GroupSpec(
            group_idx=int(best_i),
            start=int(left.start),
            end=int(right.end),
            frame_count=int(right.end - left.start),
            stop_reason="adaptive_gop_target_canvas_merge",
            readiness=merged_readiness,
        )
        merge_records.append({
            "left_start": int(left.start),
            "left_end": int(left.end),
            "right_start": int(right.start),
            "right_end": int(right.end),
            "boundary_cost": float(best_cost),
        })
        work[best_i:best_i + 2] = [merged]
    out: List[GroupSpec] = []
    for i, g in enumerate(work):
        out.append(
            GroupSpec(
                group_idx=int(i),
                start=int(g.start),
                end=int(g.end),
                frame_count=int(g.end - g.start),
                stop_reason=str(g.stop_reason),
                readiness=dict(g.readiness),
            )
        )
    return out, {
        "target_groups": int(target),
        "merge_count": int(len(merge_records)),
        "merge_records": merge_records,
    }


def _shift_frame_ids_off_keyframes(
    frame_ids: List[int],
    keyframe_ids: List[int],
    total_frames: int,
    offset: int = 1,
) -> List[int]:
    key_set = set(int(x) for x in keyframe_ids)
    total = int(max(1, int(total_frames)))
    step = int(max(1, int(offset)))
    out: List[int] = []
    for fid in frame_ids:
        fid0 = int(max(0, min(total - 1, int(fid))))
        if fid0 not in key_set:
            out.append(fid0)
            continue
        repl = fid0
        found = False
        for d in range(step, step * 4 + 1, step):
            for cand in (fid0 + d, fid0 - d):
                if 0 <= int(cand) < total and int(cand) not in key_set:
                    repl = int(cand)
                    found = True
                    break
            if found:
                break
        out.append(int(repl))
    return out


def _dedup_and_refill_frame_ids(
    frame_ids: List[int],
    total_frames: int,
    keyframe_ids: Optional[List[int]] = None,
) -> List[int]:
    target_len = int(len(frame_ids))
    total = int(max(1, int(total_frames)))
    key_set = set(int(x) for x in (keyframe_ids or []))

    used = set()
    deduped: List[int] = []
    for fid in frame_ids:
        fid0 = int(max(0, min(total - 1, int(fid))))
        if fid0 in used:
            continue
        deduped.append(fid0)
        used.add(fid0)

    if len(deduped) >= target_len:
        return deduped[:target_len]

    seed = sorted(deduped) if deduped else [0]
    candidates = []
    for fid in range(total):
        if fid in used or fid in key_set:
            continue
        nearest = min(abs(int(fid) - int(s)) for s in seed) if seed else int(fid)
        candidates.append((nearest, abs(int(fid) - int(total // 2)), int(fid)))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    for _, _, fid in candidates:
        if len(deduped) >= target_len:
            break
        deduped.append(int(fid))
        used.add(int(fid))

    if len(deduped) < target_len:
        for fid in range(total):
            if fid in used:
                continue
            deduped.append(int(fid))
            used.add(int(fid))
            if len(deduped) >= target_len:
                break

    deduped.sort()
    return deduped[:target_len]


def _resolve_readiness_threshold(
    cfg: BitcostReadinessConfig,
    score_maps: List[np.ndarray],
    bits_per_pixel_per_frame: float,
    block_scores_all: Optional[List[np.ndarray]] = None,
) -> tuple[float, float, float]:
    mode = str(cfg.readiness_sum_threshold_mode).lower().strip()
    threshold = float(cfg.readiness_sum_threshold)
    clamp_min = float(max(0.0, float(cfg.bpppf_clamp_min)))
    clamp_max = float(max(0.0, float(cfg.bpppf_clamp_max)))
    clamped_bpppf = float(bits_per_pixel_per_frame)
    if clamp_min > 0.0:
        clamped_bpppf = max(clamped_bpppf, clamp_min)
    if clamp_max > 0.0:
        clamped_bpppf = min(clamped_bpppf, clamp_max)
    norm_factor = float(np.sqrt(clamped_bpppf)) if clamped_bpppf > 1e-12 else 1.0

    if mode == "legacy":
        if threshold <= 0.0:
            threshold = estimate_readiness_threshold(
                score_maps=score_maps,
                group_size=int(cfg.group_size),
                patch=int(cfg.patch),
                images_per_group=int(cfg.images_per_group),
                coverage_bins_target=int(cfg.readiness_coverage_bins),
                block_size=int(cfg.block_size),
                block_scores_all=block_scores_all,
            )
    elif mode == "auto":
        threshold = estimate_readiness_threshold(
            score_maps=score_maps,
            group_size=int(cfg.group_size),
            patch=int(cfg.patch),
            images_per_group=int(cfg.images_per_group),
            coverage_bins_target=int(cfg.readiness_coverage_bins),
            block_size=int(cfg.block_size),
            block_scores_all=block_scores_all,
        )
    elif mode == "clamped_sqrt_bpppf":
        threshold = float(cfg.readiness_norm_sum_threshold) * float(norm_factor)
    elif mode == "fixed":
        threshold = float(threshold)
    else:
        raise ValueError(f"unsupported readiness_sum_threshold_mode: {mode}")
    return float(threshold), float(clamped_bpppf), float(norm_factor)


def _clip_iframe_score_maps(
    score_maps: List[np.ndarray],
    bitcost_items: List[Dict[str, Any]],
    mode: str,
    percentile: float,
) -> tuple[List[np.ndarray], Optional[Dict[str, Any]]]:
    mode = str(mode).lower().strip()
    if mode in {"", "none"}:
        return score_maps, None
    if mode != "non_i_percentile":
        raise ValueError(f"unsupported iframe_score_clip_mode: {mode}")

    iframe_indices = [
        idx for idx, item in enumerate(bitcost_items)
        if str(item.get("pict_type", "")).upper() == "I"
    ]
    iframe_index_set = set(iframe_indices)
    non_iframe_scores = [
        np.asarray(score_maps[idx], dtype=np.float32).reshape(-1)
        for idx in range(len(score_maps))
        if idx not in iframe_index_set
    ]
    if not iframe_indices or not non_iframe_scores:
        return score_maps, {
            "mode": mode,
            "percentile": float(percentile),
            "iframe_count": int(len(iframe_indices)),
            "applied": False,
        }

    pct = float(np.clip(float(percentile), 0.0, 100.0))
    cap = float(np.percentile(np.concatenate(non_iframe_scores), pct))
    clipped: List[np.ndarray] = []
    changed_pixels = 0
    for idx, score in enumerate(score_maps):
        arr = np.asarray(score, dtype=np.float32)
        if idx in iframe_index_set:
            changed_pixels += int(np.count_nonzero(arr > cap))
            arr = np.minimum(arr, cap).astype(np.float32)
        clipped.append(arr)
    return clipped, {
        "mode": mode,
        "percentile": pct,
        "cap": cap,
        "iframe_count": int(len(iframe_indices)),
        "changed_pixels": int(changed_pixels),
        "applied": True,
    }


def run_bitcost_readiness(config: BitcostReadinessConfig) -> PipelineResult:
    t_pipeline_start = time.perf_counter()
    timing: Dict[str, float] = {}

    def elapsed_since(start: float) -> float:
        return float(time.perf_counter() - start)

    def timed_decode_frames() -> tuple[List[np.ndarray], float]:
        t_inner = time.perf_counter()
        frames = decode_frames_bgr_ffmpeg(
            str(cfg.video),
            frame_ids,
            meta=meta,
            out_h=decode_out_h,
            out_w=decode_out_w,
        )
        return frames, elapsed_since(t_inner)

    def timed_fetch_bitcost() -> tuple[List[Dict[str, Any]], float]:
        t_inner = time.perf_counter()
        items = cv_reader_fetch_bitcost(
            str(cfg.video), [int(x) for x in frame_ids],
            bitcost_grid=str(cfg.bitcost_grid),
            parallel_segments=int(cfg.parallel_segments),
            threads_per_segment=int(cfg.threads_per_segment),
            segment_guard_frames=int(cfg.segment_guard_frames),
        )
        return items, elapsed_since(t_inner)

    def timed_fetch_bitcost_and_pixels() -> tuple[List[Dict[str, Any]], List[np.ndarray], float]:
        t_inner = time.perf_counter()
        items = cv_reader_fetch_bitcost(
            str(cfg.video), [int(x) for x in frame_ids],
            bitcost_grid=str(cfg.bitcost_grid),
            export_pixels=1,
            out_w=int(prepared_w),
            out_h=int(prepared_h),
            parallel_segments=int(cfg.parallel_segments),
            threads_per_segment=int(cfg.threads_per_segment),
            segment_guard_frames=int(cfg.segment_guard_frames),
        )
        frames = [it["pixels"] for it in items]
        return items, frames, elapsed_since(t_inner)

    cfg = config.normalized()
    ensure_dir(cfg.out_dir)
    out_dir = Path(cfg.out_dir)

    t0 = time.perf_counter()
    meta = probe_video(cfg.video)
    timing["probe_video"] = elapsed_since(t0)
    if int(meta.total_frames) <= 0 or int(meta.height) <= 0 or int(meta.width) <= 0:
        raise RuntimeError("cannot read video metadata with ffprobe")

    t0 = time.perf_counter()
    frame_ids, sampling_debug = sample_frames(meta, cfg)
    timing["sample_frames"] = elapsed_since(t0)
    keyframe_ids: List[int] = []
    if cfg.frame_sampling_mode == "all_frames" and bool(cfg.avoid_keyframes):
        sampling_debug["avoid_keyframes_note"] = "ignored for all_frames sampling"
    elif bool(cfg.avoid_keyframes):
        t0 = time.perf_counter()
        keyframe_ids = mp4_keyframe_frame_ids(str(cfg.video))
        if keyframe_ids is None:
            keyframe_ids = ffprobe_keyframe_frame_ids(str(cfg.video), fps=float(meta.fps), total_frames=int(meta.total_frames))
        frame_ids = _shift_frame_ids_off_keyframes(
            frame_ids=frame_ids,
            keyframe_ids=keyframe_ids,
            total_frames=int(meta.total_frames),
            offset=int(cfg.avoid_keyframe_offset),
        )
        frame_ids = _dedup_and_refill_frame_ids(
            frame_ids=frame_ids,
            total_frames=int(meta.total_frames),
            keyframe_ids=keyframe_ids,
        )
        sampling_debug["keyframes_found"] = int(len(keyframe_ids))
        sampling_debug["sampled_frames_after_keyframe_shift"] = int(len(frame_ids))
        timing["avoid_keyframes"] = elapsed_since(t0)

    t0 = time.perf_counter()
    resize_h_target, resize_w_target, pad_bottom_target, pad_right_target, prepared_h, prepared_w = resolve_prepared_frame_geometry(
        height=int(meta.height),
        width=int(meta.width),
        patch=int(cfg.patch),
        max_dim=int(cfg.max_dim),
        block_size=int(cfg.block_size),
        max_pixels=int(cfg.max_pixels),
        no_resize=bool(cfg.no_resize),
    )
    decode_out_h = int(prepared_h) if bool(cfg.ffmpeg_preprocess_frames) and not bool(cfg.no_resize) else None
    decode_out_w = int(prepared_w) if bool(cfg.ffmpeg_preprocess_frames) and not bool(cfg.no_resize) else None
    timing["resolve_frame_geometry"] = elapsed_since(t0)

    use_cv_pixels = str(cfg.decode_backend) == "cv_reader_pixels"
    t0 = time.perf_counter()
    if use_cv_pixels:
        bitcost_items, frames_bgr, timing["cv_reader_fetch_bitcost"] = timed_fetch_bitcost_and_pixels()
        timing["decode_frames"] = 0.0
        timing["decode_and_cv_reader_serial_wall"] = elapsed_since(t0)
    elif bool(cfg.parallel_decode_cv_reader):
        with ThreadPoolExecutor(max_workers=2) as executor:
            decode_future = executor.submit(timed_decode_frames)
            bitcost_future = executor.submit(timed_fetch_bitcost)
            frames_bgr, timing["decode_frames"] = decode_future.result()
            bitcost_items, timing["cv_reader_fetch_bitcost"] = bitcost_future.result()
        timing["decode_and_cv_reader_parallel_wall"] = elapsed_since(t0)
    else:
        frames_bgr, timing["decode_frames"] = timed_decode_frames()
        bitcost_items, timing["cv_reader_fetch_bitcost"] = timed_fetch_bitcost()
        timing["decode_and_cv_reader_serial_wall"] = elapsed_since(t0)
    if len(frames_bgr) != len(frame_ids):
        raise RuntimeError(f"decoded {len(frames_bgr)} frames, expected {len(frame_ids)}")

    t0 = time.perf_counter()
    frames_bgr, resize_h, resize_w, pad_bottom, pad_right = prepare_frames(
        frames_bgr,
        patch=int(cfg.patch),
        max_dim=int(cfg.max_dim),
        block_size=int(cfg.block_size),
        max_pixels=int(cfg.max_pixels),
        no_resize=bool(cfg.no_resize),
    )
    timing["prepare_frames"] = elapsed_since(t0)
    h1, w1 = frames_bgr[0].shape[:2]

    # Precompute good_mask for all frames to avoid repeated frame_is_bad checks in group loop
    t0 = time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        all_good_masks = list(ex.map(lambda f: not frame_is_bad(f), frames_bgr))
    timing["precompute_good_masks"] = elapsed_since(t0)

    if len(bitcost_items) != len(frame_ids):
        raise RuntimeError(f"bitcost fetch mismatch: {len(bitcost_items)} vs {len(frame_ids)}")

    t0 = time.perf_counter()
    score_maps = bitcost_items_to_score_maps(
        bitcost_items=bitcost_items,
        out_h=int(h1),
        out_w=int(w1),
        bitcost_grid=str(cfg.bitcost_grid),
        bitcost_pct=float(cfg.bitcost_pct),
        bitcost_log_scale=bool(cfg.bitcost_log_scale),
        codec_name=str(meta.codec_name),
    )
    timing["bitcost_to_score_maps"] = elapsed_since(t0)

    iframe_score_clip_debug: Optional[Dict[str, Any]] = None
    if str(cfg.iframe_score_clip_mode).lower().strip() != "none":
        t0 = time.perf_counter()
        score_maps, iframe_score_clip_debug = _clip_iframe_score_maps(
            score_maps=score_maps,
            bitcost_items=bitcost_items,
            mode=str(cfg.iframe_score_clip_mode),
            percentile=float(cfg.iframe_score_clip_percentile),
        )
        timing["clip_iframe_score_maps"] = elapsed_since(t0)

    frame_score_norm_debug: Optional[Dict[str, Any]] = None
    if str(cfg.frame_score_norm_mode) == "frame_mean_floor":
        t0 = time.perf_counter()
        score_maps, frame_score_norm_debug = normalize_score_maps_frame_mean_floor(
            score_maps=score_maps,
            patch=int(cfg.patch),
            block_size=int(cfg.block_size),
            floor_ratio=float(cfg.frame_score_norm_floor_ratio),
            allow_boost=bool(cfg.frame_score_norm_allow_boost),
        )
        timing["normalize_score_maps"] = elapsed_since(t0)

    fps_safe = float(meta.fps) if float(meta.fps) > 0 else 1.0
    pixel_count = float(max(1, int(meta.width) * int(meta.height)))
    bits_per_frame = float(meta.bit_rate) / float(max(fps_safe, 1e-6)) if float(meta.bit_rate) > 0 else 0.0
    bits_per_pixel_per_frame = bits_per_frame / pixel_count if bits_per_frame > 0 else 0.0
    min_group_frames, max_group_frames, group_limit_debug = resolve_group_frame_limits(cfg)

    t0 = time.perf_counter()
    block_scores_all = precompute_frame_block_scores(
        score_maps,
        patch=int(cfg.patch),
        block_size=int(cfg.block_size),
    )
    timing["precompute_block_scores"] = elapsed_since(t0)

    readiness_threshold = float(cfg.readiness_sum_threshold)
    clamped_bpppf = float(bits_per_pixel_per_frame)
    readiness_norm_factor = 1.0
    adaptive_gop_debug: Optional[Dict[str, Any]] = None
    t0 = time.perf_counter()
    if bool(cfg.adaptive_gop):
        group_specs, adaptive_gop_debug = _build_adaptive_gop_groups(
            frame_ids=frame_ids,
            score_maps=score_maps,
            min_group_frames=int(min_group_frames),
            max_group_frames=int(max_group_frames),
            gamma=float(cfg.adaptive_gop_gamma),
            percentile=float(cfg.adaptive_gop_percentile),
        )
        if int(cfg.target_canvas) > 0:
            target_groups = max(1, int(cfg.target_canvas) // max(1, int(cfg.images_per_group)))
            before_groups = int(len(group_specs))
            group_specs, merge_debug = _merge_adaptive_gop_groups_to_target(
                group_specs,
                target_groups=int(target_groups),
            )
            adaptive_gop_debug["target_canvas"] = int(cfg.target_canvas)
            adaptive_gop_debug["target_groups"] = int(target_groups)
            adaptive_gop_debug["groups_before_target_merge"] = int(before_groups)
            adaptive_gop_debug["groups_after_target_merge"] = int(len(group_specs))
            adaptive_gop_debug["target_merge"] = merge_debug
        if bool(cfg.verbose):
            print(
                f"adaptive_gop: groups={len(group_specs)} "
                f"threshold={float(adaptive_gop_debug.get('threshold', 0.0)):.4f} "
                f"splits={adaptive_gop_debug.get('split_points', [])}"
            )
    elif cfg.grouping_mode == "readiness":
        readiness_threshold, clamped_bpppf, readiness_norm_factor = _resolve_readiness_threshold(
            cfg=cfg,
            score_maps=score_maps,
            bits_per_pixel_per_frame=float(bits_per_pixel_per_frame),
            block_scores_all=block_scores_all,
        )
        group_specs = build_readiness_groups(
            frame_ids=frame_ids,
            score_maps=score_maps,
            cfg=cfg,
            readiness_sum_threshold=float(readiness_threshold),
            min_group_frames=int(min_group_frames),
            max_group_frames=int(max_group_frames),
            block_scores_all=block_scores_all,
        )
    else:
        group_specs = build_fixed_groups(frame_ids=frame_ids, group_size=int(cfg.group_size))
    timing["build_groups"] = elapsed_since(t0)

    all_group_meta: List[Dict[str, Any]] = []
    all_keep_patch_masks: List[np.ndarray] = []
    all_canvases: List[np.ndarray] = []
    all_patch_pos: List[np.ndarray] = []
    all_src_pos: List[np.ndarray] = []
    global_img_offset = 0

    t0 = time.perf_counter()
    group_processing_timings: List[Dict[str, Any]] = []
    for group_idx, spec in enumerate(group_specs):
        t_group_start = time.perf_counter()
        start = int(spec.start)
        end = int(spec.end)
        group_frame_ids = [int(x) for x in frame_ids[start:end]]
        if len(group_frame_ids) < 4:
            continue

        group_frames = list(frames_bgr[start:end])
        group_scores = list(score_maps[start:end])
        group_block_scores = list(block_scores_all[start:end])
        group_good_mask = all_good_masks[start:end]
        anchor_idx = 0
        anchor_strategy = "fixed_anchor"
        anchor_debug: Optional[Dict[str, Any]] = None
        if bool(cfg.adaptive_anchor):
            anchor_idx, anchor_debug = _select_adaptive_anchor(
                group_scores=group_scores,
                good_mask=[bool(x) for x in group_good_mask],
                w_center=float(cfg.adaptive_anchor_w_center),
                w_low_bc=float(cfg.adaptive_anchor_w_low_bc),
            )
            anchor_strategy = "adaptive_anchor"
            if bool(cfg.verbose):
                print(
                    f"Group {group_idx}: frames={group_frame_ids} "
                    f"selected_anchor={int(group_frame_ids[int(anchor_idx)])} "
                    f"reason: center_rank={int(anchor_debug.get('center_rank', 0))} "
                    f"bitcost_rank={int(anchor_debug.get('bitcost_rank', 0))}"
                )
        selector_kwargs = {
            "group_idx": int(group_idx),
            "group_frame_ids": group_frame_ids,
            "group_frames_bgr": group_frames,
            "group_scores": group_scores,
            "images_per_group": int(cfg.images_per_group),
            "patch": int(cfg.patch),
            "block_size": int(cfg.block_size),
            "group_block_scores": group_block_scores,
            "good_mask": group_good_mask,
            "anchor_idx": int(anchor_idx),
            "anchor_strategy": str(anchor_strategy),
        }
        if str(cfg.selector_mode) == "diverse_mixed_simple":
            group_meta, keep_patch_mask, images_rgb, patch_pos, src_pos, _img_ptr = (
                process_group_diverse_mixed_simple(
                    **selector_kwargs,
                    diversity_fraction=float(cfg.diversity_fraction),
                    novelty_weight=float(cfg.novelty_weight),
                    dedup_enabled=bool(cfg.dedup_enabled),
                    dedup_descriptor=str(cfg.dedup_descriptor),
                    dedup_threshold=cfg.dedup_threshold,
                )
            )
        else:
            group_meta, keep_patch_mask, images_rgb, patch_pos, src_pos, _img_ptr = process_group_topk_2x2(
                **selector_kwargs,
                event_aggregation=bool(cfg.event_aggregation),
                event_aggregation_bins=int(cfg.event_aggregation_bins),
                event_aggregation_min_blocks=int(cfg.event_aggregation_min_blocks),
            )
        if anchor_debug is not None:
            group_meta["anchor_debug"] = anchor_debug

        patch_pos_global = patch_pos.copy()
        patch_pos_global[:, 0] += int(global_img_offset)
        all_patch_pos.append(patch_pos_global)
        all_src_pos.append(src_pos)
        all_canvases.extend([img for img in images_rgb])
        all_keep_patch_masks.append(keep_patch_mask)

        group_meta["global_img_offset"] = int(global_img_offset)
        group_meta["num_canvases"] = int(images_rgb.shape[0])
        group_meta["group_start_idx"] = int(spec.start)
        group_meta["group_end_idx"] = int(spec.end)
        group_meta["group_frame_count"] = int(spec.frame_count)
        group_meta["group_stop_reason"] = str(spec.stop_reason)
        group_meta["group_readiness"] = spec.readiness
        all_group_meta.append(group_meta)
        global_img_offset += int(images_rgb.shape[0])
        group_processing_timings.append({"group_idx": int(group_idx), "sec": elapsed_since(t_group_start)})
    timing["process_groups_make_canvases"] = elapsed_since(t0)

    selector_groups = [
        group["selector"]
        for group in all_group_meta
        if isinstance(group.get("selector"), dict)
    ]
    selector_summary: Optional[Dict[str, Any]] = None
    if selector_groups:
        total_selected = int(sum(int(group.get("target_blocks", 0)) for group in selector_groups))
        total_adjacent_pairs = int(
            sum(int(group.get("adjacent_same_position_pairs", 0)) for group in selector_groups)
        )
        total_adjacent_duplicates = int(
            sum(
                int(group.get("adjacent_same_position_duplicates", 0))
                for group in selector_groups
            )
        )

        def weighted_mean(field: str) -> float:
            if total_selected <= 0:
                return 0.0
            return float(
                sum(
                    float(group.get(field, 0.0)) * int(group.get("target_blocks", 0))
                    for group in selector_groups
                )
                / float(total_selected)
            )

        timing_fields = ("descriptor", "score", "dedup_map", "selection", "total")
        selector_summary = {
            "mode": "diverse_mixed_simple",
            "group_count": int(len(selector_groups)),
            "target_blocks": int(total_selected),
            "bitcost_selected": int(
                sum(int(group.get("bitcost_selected", 0)) for group in selector_groups)
            ),
            "diversity_selected": int(
                sum(int(group.get("diversity_selected", 0)) for group in selector_groups)
            ),
            "backfill_selected": int(
                sum(int(group.get("backfill_selected", 0)) for group in selector_groups)
            ),
            "dedup_rejected": int(
                sum(int(group.get("dedup_rejected", 0)) for group in selector_groups)
            ),
            "dedup_rejected_unique": int(
                sum(int(group.get("dedup_rejected_unique", 0)) for group in selector_groups)
            ),
            "dedup_comparisons": int(
                sum(int(group.get("dedup_comparisons", 0)) for group in selector_groups)
            ),
            "mean_unique_source_frames_per_group": float(
                np.mean([float(group.get("unique_source_frames", 0)) for group in selector_groups])
            ),
            "mean_unique_spatial_positions_per_group": float(
                np.mean(
                    [float(group.get("unique_spatial_positions", 0)) for group in selector_groups]
                )
            ),
            "mean_temporal_distribution_entropy_normalized": weighted_mean(
                "temporal_distribution_entropy_normalized"
            ),
            "mean_max_blocks_per_frame_fraction": weighted_mean(
                "max_blocks_per_frame_fraction"
            ),
            "adjacent_same_position_pairs": int(total_adjacent_pairs),
            "adjacent_same_position_duplicates": int(total_adjacent_duplicates),
            "adjacent_same_position_duplicate_rate": (
                float(total_adjacent_duplicates) / float(total_adjacent_pairs)
                if total_adjacent_pairs
                else 0.0
            ),
            "adjacent_mad_count": int(
                sum(int(group.get("adjacent_mad_count", 0)) for group in selector_groups)
            ),
            "mean_adjacent_mad_fraction_le_threshold": float(
                np.average(
                    [
                        float(group.get("adjacent_mad_fraction_le_threshold", 0.0))
                        for group in selector_groups
                    ],
                    weights=[
                        int(group.get("adjacent_mad_count", 0))
                        for group in selector_groups
                    ],
                )
                if sum(
                    int(group.get("adjacent_mad_count", 0))
                    for group in selector_groups
                )
                else 0.0
            ),
            "adjacent_mad_cdf": {
                threshold: float(
                    sum(
                        float((group.get("adjacent_mad_cdf") or {}).get(threshold, 0.0))
                        * int(group.get("adjacent_mad_count", 0))
                        for group in selector_groups
                    )
                    / float(
                        sum(
                            int(group.get("adjacent_mad_count", 0))
                            for group in selector_groups
                        )
                    )
                )
                for threshold in sorted(
                    {
                        threshold
                        for group in selector_groups
                        for threshold in (group.get("adjacent_mad_cdf") or {})
                    }
                )
            }
            if sum(
                int(group.get("adjacent_mad_count", 0))
                for group in selector_groups
            )
            else {},
            "selected_bitcost_mean": weighted_mean("selected_bitcost_mean"),
            "selected_novelty_mean": weighted_mean("selected_novelty_mean"),
            "selected_edge_mean": weighted_mean("selected_edge_mean"),
            "timing_sec": {
                field: float(
                    sum(
                        float((group.get("timing_sec") or {}).get(field, 0.0))
                        for group in selector_groups
                    )
                )
                for field in timing_fields
            },
        }

    if not all_canvases:
        raise RuntimeError("no canvases produced")

    t0 = time.perf_counter()
    images_rgb_all = np.stack(all_canvases, axis=0).astype(np.uint8)
    timing["stack_canvases"] = elapsed_since(t0)

    t0 = time.perf_counter()
    canvas_files = save_canvases(images_rgb_all, str(out_dir), fmt=str(cfg.canvas_format), quality=95)
    timing["save_canvases"] = elapsed_since(t0)

    t0 = time.perf_counter()
    patch_position = np.concatenate(all_patch_pos, axis=0).astype(np.int32)
    src_patch_position_raw = np.concatenate(all_src_pos, axis=0).astype(np.int32)
    hb = int(h1 // int(cfg.patch))
    wb = int(w1 // int(cfg.patch))
    s_full = int(hb * wb)
    raster_idx = (
        patch_position[:, 0].astype(np.int64) * int(s_full)
        + patch_position[:, 1].astype(np.int64) * int(wb)
        + patch_position[:, 2].astype(np.int64)
    )
    src_patch_position = np.zeros_like(src_patch_position_raw)
    src_patch_position[raster_idx] = src_patch_position_raw
    np.save(str(out_dir / "src_patch_position.npy"), src_patch_position, allow_pickle=False)
    np.save(str(out_dir / "frame_ids.npy"), np.asarray(frame_ids, dtype=np.int32), allow_pickle=False)
    timing["save_numpy_outputs"] = elapsed_since(t0)

    sbs_video_path = None
    masked_video_path = None
    if bool(cfg.save_mask_video) and all_keep_patch_masks:
        t0 = time.perf_counter()
        keep_patch_mask_full = np.concatenate(all_keep_patch_masks, axis=0).astype(np.uint8)
        out_fps = float(cfg.video_fps) if float(cfg.video_fps) > 0 else float(cfg.sample_fps)
        sbs_frames: List[np.ndarray] = []
        masked_frames: List[np.ndarray] = []
        p = int(cfg.patch)
        for fr, keep_grid in zip(frames_bgr, keep_patch_mask_full):
            masked = np.zeros_like(fr)
            ys, xs = np.where(keep_grid > 0)
            for ph, pw in zip(ys.tolist(), xs.tolist()):
                y0 = int(ph) * p
                x0 = int(pw) * p
                masked[y0:y0 + p, x0:x0 + p, :] = fr[y0:y0 + p, x0:x0 + p, :]
            sbs_frames.append(np.concatenate([fr, masked], axis=1))
            masked_frames.append(masked)
        sbs_video_path = save_mask_video(out_dir / "sampled_masked_sbs.mp4", out_fps, sbs_frames)
        masked_video_path = save_mask_video(out_dir / "sampled_masked_only.mp4", out_fps, masked_frames)
        timing["save_mask_video"] = elapsed_since(t0)

    timing["total_before_meta_write"] = elapsed_since(t_pipeline_start)

    summary = {
        "video": str(cfg.video),
        "video_hash": sha1_8(str(cfg.video)),
        "codec_selector_version": 1,
        "config": cfg.to_dict(),
        "metadata": meta.to_dict(),
        "total_frames": int(meta.total_frames),
        "orig_hw": [int(meta.height), int(meta.width)],
        "processed_hw": [int(h1), int(w1)],
        "resize_hw": [int(resize_h), int(resize_w)],
        "padded_hw": [int(h1), int(w1)],
        "pad_bottom_right": [int(pad_bottom), int(pad_right)],
        "fps": float(meta.fps),
        "stream_hw": [int(meta.height), int(meta.width)],
        "video_bit_rate": float(meta.bit_rate),
        "bits_per_frame": float(bits_per_frame),
        "bits_per_pixel_per_frame": float(bits_per_pixel_per_frame),
        "bits_per_pixel_per_frame_clamped": float(clamped_bpppf),
        "sampled_frames": int(len(frame_ids)),
        "sampling_debug": sampling_debug,
        "avoid_keyframes": bool(cfg.avoid_keyframes),
        "avoid_keyframe_offset": int(cfg.avoid_keyframe_offset),
        "keyframes_found": int(len(keyframe_ids)),
        "grouping_mode": str(cfg.grouping_mode),
        "group_size": int(cfg.group_size),
        "target_canvas": int(cfg.target_canvas),
        "images_per_group": int(cfg.images_per_group),
        "readiness_sum_threshold_mode": str(cfg.readiness_sum_threshold_mode),
        "readiness_sum_threshold": float(readiness_threshold),
        "readiness_norm_sum_threshold": float(cfg.readiness_norm_sum_threshold),
        "readiness_threshold_norm_factor": float(readiness_norm_factor),
        "bpppf_clamp_min": float(cfg.bpppf_clamp_min),
        "bpppf_clamp_max": float(cfg.bpppf_clamp_max),
        "min_group_frames": int(min_group_frames),
        "max_group_frames": int(max_group_frames),
        "group_limit_debug": group_limit_debug,
        "readiness_coverage_bins": int(cfg.readiness_coverage_bins),
        "readiness_delta_ratio": float(cfg.readiness_delta_ratio),
        "save_mask_video": bool(cfg.save_mask_video),
        "verbose": bool(cfg.verbose),
        "adaptive_anchor": bool(cfg.adaptive_anchor),
        "adaptive_anchor_w_center": float(cfg.adaptive_anchor_w_center),
        "adaptive_anchor_w_low_bc": float(cfg.adaptive_anchor_w_low_bc),
        "adaptive_gop": bool(cfg.adaptive_gop),
        "adaptive_gop_gamma": float(cfg.adaptive_gop_gamma),
        "adaptive_gop_percentile": float(cfg.adaptive_gop_percentile),
        "adaptive_gop_debug": adaptive_gop_debug,
        "event_aggregation": bool(cfg.event_aggregation),
        "event_aggregation_bins": int(cfg.event_aggregation_bins),
        "event_aggregation_min_blocks": int(cfg.event_aggregation_min_blocks),
        "selector_mode": str(cfg.selector_mode),
        "diversity_fraction": float(cfg.diversity_fraction),
        "novelty_weight": float(cfg.novelty_weight),
        "dedup_enabled": bool(cfg.dedup_enabled),
        "dedup_descriptor": str(cfg.dedup_descriptor),
        "dedup_threshold": cfg.dedup_threshold,
        "selector_summary": selector_summary,
        "mask_video_sbs": str(Path(sbs_video_path).name) if sbs_video_path is not None else None,
        "mask_video_only": str(Path(masked_video_path).name) if masked_video_path is not None else None,
        "patch": int(cfg.patch),
        "block_size": int(cfg.block_size),
        "bitcost_grid": str(cfg.bitcost_grid),
        "codec_name": str(meta.codec_name),
        "bitcost_pct": float(cfg.bitcost_pct),
        "bitcost_log_scale": bool(cfg.bitcost_log_scale),
        "frame_score_norm_mode": str(cfg.frame_score_norm_mode),
        "frame_score_norm_debug": frame_score_norm_debug,
        "iframe_score_clip_mode": str(cfg.iframe_score_clip_mode),
        "iframe_score_clip_percentile": float(cfg.iframe_score_clip_percentile),
        "iframe_score_clip_debug": iframe_score_clip_debug,
        "score_source": "bitcost",
        "decode_backend": "ffmpeg_native",
        "hb_wb": [int(hb), int(wb)],
        "S_full": int(s_full),
        "seq_len": int(len(frame_ids)),
        "num_groups": int(len(all_group_meta)),
        "tail_frames": int(len(frame_ids) % int(cfg.group_size)) if cfg.grouping_mode == "fixed" else 0,
        "group_specs": [spec.to_dict() if isinstance(spec, GroupSpec) else dict(spec) for spec in group_specs],
        "total_images": int(images_rgb_all.shape[0]),
        "total_patches": int(patch_position.shape[0]),
        "canvas_files": canvas_files,
        "jpg_files": canvas_files,
        "out_dir": str(out_dir),
        "src_patch_position_sorted_by_t": True,
        "canvas_pixels_sorted_by_t": True,
        "groups": all_group_meta,
        "timing_sec": timing,
        "group_processing_timing_sec": group_processing_timings,
    }
    summary = _jsonable(summary)
    meta_path = out_dir / "meta.json"
    t0 = time.perf_counter()
    timing["total"] = elapsed_since(t_pipeline_start)
    timing["write_meta_json"] = 0.0
    summary["timing_sec"] = _jsonable(timing)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    timing["write_meta_json"] = elapsed_since(t0)
    summary["timing_sec"] = _jsonable(timing)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        "[timing] "
        f"total={timing.get('total', 0.0):.3f}s "
        f"cv_reader={timing.get('cv_reader_fetch_bitcost', 0.0):.3f}s "
        f"decode={timing.get('decode_frames', 0.0):.3f}s "
        f"canvas_build={timing.get('process_groups_make_canvases', 0.0):.3f}s "
        f"canvas_save={timing.get('save_canvases', 0.0):.3f}s"
    )

    return PipelineResult(
        out_dir=str(out_dir),
        meta_path=str(meta_path),
        canvas_files=canvas_files,
        summary=summary,
    )
