#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-k block selector used by the bitcost readiness pipeline."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from codec_selector.core.registry import selectors
from codec_selector.codec_patch_gop.frame_utils import frame_is_bad
from codec_selector.codec_patch_gop.patch_utils import (
    block_to_patches,
    extract_patch_rgb,
    iter_blocks_in_raster,
    pack_patches_to_canvases,
)
from codec_selector.codec_patch_gop.scoring import patch_scores_to_block_scores, score_map_to_patch_scores


def process_group_topk_2x2(
    group_idx: int,
    group_frame_ids: List[int],
    group_frames_bgr: List[np.ndarray],
    group_scores: List[np.ndarray],
    images_per_group: int,
    patch: int,
    block_size: int = 2,
    group_block_scores: Optional[List[np.ndarray]] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = int(patch)
    b = int(max(1, int(block_size)))
    block_patch_count = int(b * b)
    h1, w1 = group_frames_bgr[0].shape[:2]
    hb, wb = h1 // p, w1 // p
    if hb % b != 0 or wb % b != 0:
        raise RuntimeError(f"group {group_idx}: grid not divisible by block_size={b}: hb={hb} wb={wb}")

    s_full = int(hb * wb)
    bwb = int(wb // b)
    pz = np.zeros((p, p, 3), dtype=np.uint8)
    keep_patch_mask = np.zeros((len(group_frames_bgr), hb, wb), dtype=np.uint8)

    good_mask = [not frame_is_bad(fr) for fr in group_frames_bgr]
    if not good_mask[0]:
        for swap_i in range(1, len(good_mask)):
            if good_mask[swap_i]:
                group_frames_bgr[0], group_frames_bgr[swap_i] = group_frames_bgr[swap_i], group_frames_bgr[0]
                group_frame_ids[0], group_frame_ids[swap_i] = group_frame_ids[swap_i], group_frame_ids[0]
                group_scores[0], group_scores[swap_i] = group_scores[swap_i], group_scores[0]
                if group_block_scores is not None:
                    group_block_scores[0], group_block_scores[swap_i] = group_block_scores[swap_i], group_block_scores[0]
                good_mask[0], good_mask[swap_i] = good_mask[swap_i], good_mask[0]
                break

    total_patches = int(images_per_group) * int(s_full)
    block_budget = max(0, int((total_patches - s_full) // block_patch_count))

    patches_list: List[np.ndarray] = []
    src_pos_list: List[List[int]] = []
    src_fid_list: List[int] = []

    iframe_rgb = group_frames_bgr[0][:, :, ::-1]
    iframe_fid = int(group_frame_ids[0])
    for bh, bw_idx in iter_blocks_in_raster(hb, wb, block_size=b):
        for ph, pw in block_to_patches(bh, bw_idx, block_size=b):
            patches_list.append(extract_patch_rgb(iframe_rgb, ph, pw, patch=p))
            src_pos_list.append([iframe_fid, int(ph), int(pw)])
            src_fid_list.append(iframe_fid)
            keep_patch_mask[0, int(ph), int(pw)] = 1

    scores_all: List[np.ndarray] = []
    t_all: List[np.ndarray] = []
    idx_all: List[np.ndarray] = []
    for t in range(1, len(group_frames_bgr)):
        if not good_mask[t]:
            continue
        if group_block_scores is not None:
            flat = np.asarray(group_block_scores[t], dtype=np.float32).reshape(-1)
        else:
            ps = score_map_to_patch_scores(group_scores[t], patch=p)
            bs = patch_scores_to_block_scores(ps, block_size=b)
            flat = bs.reshape(-1).astype(np.float32)
        scores_all.append(flat)
        t_all.append(np.full((flat.size,), int(t), dtype=np.int32))
        idx_all.append(np.arange(flat.size, dtype=np.int32))

    selected_blocks_by_frame: Dict[int, List[Tuple[int, int]]] = {}
    selected_scores_by_frame: Dict[int, List[float]] = {}
    selected_score_sum = 0.0
    selected_score_mean = 0.0
    if scores_all and block_budget > 0:
        scores_cat = np.concatenate(scores_all)
        t_cat = np.concatenate(t_all)
        idx_cat = np.concatenate(idx_all)
        k = min(block_budget, len(scores_cat))
        if k < len(scores_cat):
            topk_idx = np.argpartition(-scores_cat, kth=k - 1)[:k]
        else:
            topk_idx = np.arange(len(scores_cat))
        topk_idx = topk_idx[np.argsort(-scores_cat[topk_idx])]
        if topk_idx.size > 0:
            selected_score_sum = float(scores_cat[topk_idx].sum())
            selected_score_mean = float(scores_cat[topk_idx].mean())
        for j in topk_idx.tolist():
            tt = int(t_cat[j])
            fi = int(idx_cat[j])
            bh_sel = int(fi // bwb)
            bw_sel = int(fi % bwb)
            selected_blocks_by_frame.setdefault(tt, []).append((bh_sel, bw_sel))
            selected_scores_by_frame.setdefault(tt, []).append(float(scores_cat[j]))
        for tt in selected_blocks_by_frame:
            selected_blocks_by_frame[tt].sort(key=lambda x: (x[0], x[1]))

    for t in range(1, len(group_frames_bgr)):
        if not good_mask[t]:
            continue
        fr_rgb = group_frames_bgr[t][:, :, ::-1]
        pfid = int(group_frame_ids[t])
        for bh_sel, bw_sel in selected_blocks_by_frame.get(t, []):
            for ph, pw in block_to_patches(bh_sel, bw_sel, block_size=b):
                patches_list.append(extract_patch_rgb(fr_rgb, ph, pw, patch=p))
                src_pos_list.append([pfid, int(ph), int(pw)])
                src_fid_list.append(pfid)
                keep_patch_mask[t, int(ph), int(pw)] = 1

    n_raw = len(patches_list)
    if n_raw % block_patch_count != 0:
        raise RuntimeError(f"group {group_idx}: patches count {n_raw} not multiple of {block_patch_count}")

    valid_blocks = []
    for bi in range(n_raw // block_patch_count):
        start_i = int(bi * block_patch_count)
        end_i = int(start_i + block_patch_count)
        bp = patches_list[start_i:end_i]
        bs = src_pos_list[start_i:end_i]
        bf = src_fid_list[start_i:end_i]
        fid = bs[0][0]
        if fid >= 0:
            is_anchor_block = int(start_i) < int(s_full)
            sort_key = (0 if is_anchor_block else 1, int(fid), int(bs[0][1] // b), int(bs[0][2] // b))
            valid_blocks.append((sort_key, bp, bs, bf))
    valid_blocks.sort(key=lambda x: x[0])

    patches_list = []
    src_pos_list = []
    src_fid_list = []
    for _, bp, bs, bf in valid_blocks:
        pos_in_canvas = len(patches_list) % s_full
        need = (block_patch_count - (pos_in_canvas % block_patch_count)) % block_patch_count
        for _ in range(need):
            patches_list.append(pz)
            src_pos_list.append([-1, -1, -1])
            src_fid_list.append(-1)
        patches_list.extend(bp)
        src_pos_list.extend(bs)
        src_fid_list.extend(bf)

    while len(patches_list) < total_patches:
        patches_list.append(pz)
        src_pos_list.append([-1, -1, -1])
        src_fid_list.append(-1)
    patches_list = patches_list[:total_patches]
    src_pos_list = src_pos_list[:total_patches]
    src_fid_list = src_fid_list[:total_patches]

    patches_arr = np.stack(patches_list, axis=0).astype(np.uint8)
    images_rgb, patch_pos, img_ptr = pack_patches_to_canvases(patches_arr, hb=hb, wb=wb, patch=p, block_size=b)
    meta = {
        "group_idx": int(group_idx),
        "frame_ids": [int(x) for x in group_frame_ids],
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
        "selected_blocks_per_frame": {str(int(k)): int(len(v)) for k, v in selected_blocks_by_frame.items()},
        "selected_block_score_sum_per_frame": {
            str(int(k)): float(sum(float(x) for x in selected_scores_by_frame.get(k, [])))
            for k in selected_blocks_by_frame.keys()
        },
    }
    return (
        meta,
        keep_patch_mask.astype(np.uint8),
        images_rgb.astype(np.uint8),
        patch_pos.astype(np.int32),
        np.asarray(src_pos_list, dtype=np.int32),
        img_ptr.astype(np.int32),
    )


selectors.register("topk_2x2_bitcost", process_group_topk_2x2)
