#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Video processing core - contains process_one_video and related functions."""

import os
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2

try:
    from cv_reader_fast import read_video_fast as _read_video_fast
    from cv_reader_fast import read_video_fast_selected as _read_video_fast_selected
    from cv_reader_fast import read_video_fast_selected_segment as _read_video_fast_selected_segment
except ImportError:
    try:
        from codec_video_prep.cv_reader_fast import read_video_fast as _read_video_fast
        from codec_video_prep.cv_reader_fast import read_video_fast_selected as _read_video_fast_selected
        from codec_video_prep.cv_reader_fast import read_video_fast_selected_segment as _read_video_fast_selected_segment
    except ImportError:
        try:
            from compressed_video_preinfer.cv_reader_fast import read_video_fast as _read_video_fast
            from compressed_video_preinfer.cv_reader_fast import read_video_fast_selected as _read_video_fast_selected
            from compressed_video_preinfer.cv_reader_fast import read_video_fast_selected_segment as _read_video_fast_selected_segment
        except ImportError:
            _read_video_fast = None
            _read_video_fast_selected = None
            _read_video_fast_selected_segment = None
HAS_CV_READER_FAST = _read_video_fast is not None


from .utils import ensure_dir, format_timestamp_ss, smart_resize, _clamp_int, _round_to_multiple
from .frame_utils import (
    frame_is_bad,
    pad_to_multiple_of_bgr,
    pad_to_multiple_of_32_bgr,
    bgr_to_residual_y_u8,
    decode_frame_bgr_at,
    decode_frames_bgr,
    detect_letterbox_bbox_bgr,
    _resize_bgr,
    _resize_gray,
    _resize_mv_and_scale,
    _bgr_to_luma_u8,
)
from .patch_utils import (
    pack_patches_to_canvases,
    save_canvases_as_jpg,
    extract_patch_rgb,
    block_to_4_patches,
    iter_blocks_in_raster,
)
from .video_probe import (
    get_total_frames_fps,
    ffprobe_video_codec_name,
    ffprobe_keyframe_frame_ids,
    ffprobe_sum_pkt_size,
    auto_max_total_patches,
)
from .energy_sampling import (
    pick_peak_frame_ids_from_pkt_size,
    build_variable_length_gops_by_energy,
    sample_frame_ids_by_energy_cdf,
    pick_windows_by_energy,
    allocate_frames_across_windows,
    enforce_time_coverage_frame_ids,
)
from .scoring import (
    mv_res_score_map,
    score_map_to_patch_scores,
    patch_scores_to_block_scores,
)


def _split_frame_ids_for_segments(frame_ids: List[int], segments: int) -> List[List[int]]:
    n = int(len(frame_ids))
    segments = int(max(1, min(int(segments), n)))
    out: List[List[int]] = []
    for i in range(segments):
        a = int(round(i * n / segments))
        b = int(round((i + 1) * n / segments))
        part = [int(x) for x in frame_ids[a:b]]
        if part:
            out.append(part)
    return out


def _cv_reader_segment_worker(args: Tuple[str, List[int], int, int, int, str, int, int, int]) -> Tuple[int, List[Dict[str, Any]]]:
    video_path, part_ids, seek_frame, end_frame, thread_count, thread_type, export_pixels, out_w, out_h = args
    try:
        from cv_reader_fast import read_video_fast_selected_segment
    except ImportError:
        from codec_video_prep.cv_reader_fast import read_video_fast_selected_segment
    try:
        items = read_video_fast_selected_segment(
            str(video_path),
            [int(x) for x in part_ids],
            seek_frame=int(seek_frame),
            end_frame=int(end_frame),
            thread_count=int(thread_count),
            export_bitcost=1,
            thread_type=str(thread_type),
            export_pixels=int(export_pixels),
            out_w=int(out_w),
            out_h=int(out_h),
        )
    except Exception:
        from compressed_video_preinfer.cv_reader_fast import read_video_fast_selected_segment
        items = read_video_fast_selected_segment(
            str(video_path),
            [int(x) for x in part_ids],
            seek_frame=int(seek_frame),
            end_frame=int(end_frame),
            thread_count=int(thread_count),
            export_bitcost=1,
            thread_type=str(thread_type),
            export_pixels=int(export_pixels),
            out_w=int(out_w),
            out_h=int(out_h),
        )
    return int(part_ids[0] if part_ids else -1), list(items)


def _fetch_bitcost_parallel_segments(
    video_path: str,
    frame_ids: List[int],
    segments: int,
    thread_count: int,
    thread_type: str,
    guard_frames: int,
    export_pixels: int = 0,
    out_w: int = 0,
    out_h: int = 0,
) -> List[Dict[str, Any]]:
    if _read_video_fast_selected_segment is None:
        raise RuntimeError("cv_reader_fast segment seek API is not available")
    if int(segments) <= 1 or len(frame_ids) <= 1:
        raise RuntimeError("parallel cv_reader requires at least 2 segments and frames")

    sorted_ids = sorted(int(x) for x in frame_ids)
    parts = _split_frame_ids_for_segments(sorted_ids, int(segments))
    jobs = []
    for part in parts:
        seek_frame = max(0, int(part[0]) - int(guard_frames))
        end_frame = int(part[-1]) + int(guard_frames)
        jobs.append((
            str(video_path), part, seek_frame, end_frame,
            int(thread_count), str(thread_type),
            int(export_pixels), int(out_w), int(out_h),
        ))

    results: List[Tuple[int, List[Dict[str, Any]]]] = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(_cv_reader_segment_worker, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: x[0])
    items: List[Dict[str, Any]] = []
    for _, part_items in results:
        items.extend(part_items)

    got_counts: Dict[int, int] = {}
    want_counts: Dict[int, int] = {}
    for fid in sorted_ids:
        want_counts[int(fid)] = want_counts.get(int(fid), 0) + 1
    for item in items:
        fid = int(item.get("frame_idx", -1))
        got_counts[fid] = got_counts.get(fid, 0) + 1
    if got_counts != want_counts:
        missing = [fid for fid, cnt in want_counts.items() if got_counts.get(fid, 0) != cnt]
        extra = [fid for fid, cnt in got_counts.items() if want_counts.get(fid, 0) != cnt]
        if missing == [0] and not extra:
            # Some codecs do not materialize frame 0 through segment seeking even
            # when seeking from zero. Keep the parallel path and let the normal
            # missing-bitcost backfill below handle this leading frame.
            pass
        else:
            raise RuntimeError(f"segment cv_reader frame mismatch: missing={missing[:8]} extra={extra[:8]}")

    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for item in items:
        by_frame.setdefault(int(item.get("frame_idx", -1)), []).append(item)

    ordered: List[Dict[str, Any]] = []
    for fid in frame_ids:
        bucket = by_frame.get(int(fid))
        if not bucket:
            if int(fid) == 0:
                continue
            raise RuntimeError(f"segment cv_reader missing frame {int(fid)}")
        ordered.append(bucket.pop(0))
    return ordered


def cv_reader_fetch_bitcost(
    video_path: str,
    frame_ids: list,
    bitcost_grid: str = "all",
    export_pixels: int = 0,
    out_w: int = 0,
    out_h: int = 0,
    parallel_segments: int = 0,
    threads_per_segment: int = 4,
    segment_guard_frames: int = 30,
) -> list[dict]:
    """Fetch bit-cost maps aligned with frame_ids using cv_reader_fast."""
    frame_ids = [int(x) for x in frame_ids]
    grid = str(bitcost_grid).lower().strip()
    codec_name = ffprobe_video_codec_name(str(video_path))
    if grid in {"adaptive", "auto"}:
        if codec_name == "hevc":
            grid = "ctu"
        elif codec_name == "vp9":
            grid = "ctu"
        else:
            grid = "mb"
    wanted_keys = {
        "sub": {"sub_mb_bit_cost"},
        "mb": {"mb_bit_cost"},
        "ctu": {"ctu_bit_cost"},
    }.get(grid, {"sub_mb_bit_cost", "mb_bit_cost", "ctu_bit_cost"})

    # Decode sequentially, but only materialize Python/numpy objects for target frames.
    # Frame threading is faster for long sparse VideoMME-style sampling; the C++
    # reader disables decoder-internal target-only accounting for that mode because
    # frame threading drops opaque_ref under the new bitcost_only patch.
    thread_count = int(os.environ.get("CV_READER_FAST_THREAD_COUNT", "1"))
    thread_type = os.environ.get("CV_READER_FAST_THREAD_TYPE", "frame")
    if codec_name in {"hevc", "h264"} and "CV_READER_FAST_THREAD_TYPE" not in os.environ:
        # The patched HEVC bitcost path stores CTU maps correctly under slice
        # threading; frame threading can crash inside FFmpeg's HEVC decoder.
        thread_type = "slice"

    all_frames = None
    if int(parallel_segments) > 1:
        try:
            all_frames = _fetch_bitcost_parallel_segments(
                video_path=str(video_path),
                frame_ids=frame_ids,
                segments=int(parallel_segments),
                thread_count=int(threads_per_segment),
                thread_type=str(thread_type),
                guard_frames=int(segment_guard_frames),
                export_pixels=int(export_pixels),
                out_w=int(out_w),
                out_h=int(out_h),
            )
        except Exception as exc:
            print(f"[warn] parallel cv_reader segment seek failed, fallback to serial: {exc}")

    if all_frames is None:
        all_frames = _read_video_fast_selected(
            str(video_path),
            frame_ids,
            thread_count=thread_count,
            export_bitcost=1,
            thread_type=thread_type,
            export_pixels=export_pixels,
            out_w=out_w,
            out_h=out_h,
        )

    lookup = {int(f["frame_idx"]): f for f in all_frames}
    out = [None] * len(frame_ids)
    last_good = None

    for i, fid in enumerate(frame_ids):
        fid = int(fid)
        item = {"frame_idx": fid}
        fr = lookup.get(fid)
        if fr is not None:
            if fr.get("bitcost") is not None:
                bc = fr["bitcost"]
                if "sub_mb_bit_cost" in wanted_keys and "sub_mb_bit_cost" in bc:
                    item["sub_mb_bit_cost"] = np.asarray(bc["sub_mb_bit_cost"])
                if "mb_bit_cost" in wanted_keys and "mb_bit_cost" in bc:
                    item["mb_bit_cost"] = np.asarray(bc["mb_bit_cost"])
                if "ctu_bit_cost" in wanted_keys and "ctu_bit_cost" in bc:
                    item["ctu_bit_cost"] = np.asarray(bc["ctu_bit_cost"])
            if "pict_type" in fr:
                item["pict_type"] = fr["pict_type"]
            if export_pixels and "pixels" in fr:
                item["pixels"] = fr["pixels"]
            last_good = item
        elif last_good is not None:
            item = dict(last_good)
            item["frame_idx"] = fid

        out[i] = item

    # back-fill leading Nones
    if last_good is not None:
        for i in range(len(out)):
            if out[i] is None or len(out[i]) <= 1:
                out[i] = dict(last_good)
                out[i]["frame_idx"] = frame_ids[i]

    return [x for x in out if x is not None]


def bitcost_item_to_score_map(item: Dict[str, Any], out_h: int, out_w: int,
                              grid: str = "sub", pct: float = 99.0,
                              log_scale: bool = True,
                              codec_name: Optional[str] = None) -> np.ndarray:
    """Convert bitcost item to score map.
    
    Args:
        item: BitCost data from cv_reader
        out_h, out_w: Output dimensions
        grid: "sub", "mb", "ctu", or "adaptive"
        pct: Percentile for normalization
        log_scale: Apply log1p before normalization
        codec_name: "h264" or "hevc" for adaptive grid selection
    """
    # Handle adaptive grid selection based on codec
    effective_grid = grid
    if grid == "adaptive" or grid == "auto":
        codec = str(codec_name).lower().strip() if codec_name else ""
        if codec == "hevc" or codec == "h265":
            # H.265/HEVC: use CTU (64x64) as "macroblock" equivalent
            effective_grid = "ctu"
        elif codec == "vp9":
            # VP9: use SB (64x64) superblock
            effective_grid = "ctu"
        else:
            # H.264/AVC: use MB (16x16)
            effective_grid = "mb"
    
    key_map = {
        "sub": "sub_mb_bit_cost",
        "mb": "mb_bit_cost",
        "ctu": "ctu_bit_cost",
    }
    key = key_map.get(effective_grid, "sub_mb_bit_cost")
    
    if key not in item:
        # Fallback: try available keys in order of preference
        for fallback_key in ["sub_mb_bit_cost", "mb_bit_cost", "ctu_bit_cost"]:
            if fallback_key in item:
                key = fallback_key
                break
        else:
            raise RuntimeError(f"cv_reader result missing bitcost data. Tried: {list(item.keys())}")

    arr = np.asarray(item[key], dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected {key} shape: {arr.shape}")
    arr = np.maximum(arr, 0.0)
    if log_scale:
        arr = np.log1p(arr)
    s = float(np.percentile(arr, pct))
    s = max(s, 1e-6)
    arr = np.clip(arr / s, 0.0, 1.0).astype(np.float32)
    return cv2.resize(arr, (int(out_w), int(out_h)), interpolation=cv2.INTER_NEAREST)

def cv_reader_fetch_mvres(video_path: str, frame_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Fetch mv + residual_y aligned with frame_ids (supports duplicates).
    Requires cv_reader.read_video_cb.
    """
    if not HAS_CV_READER or not hasattr(cv_api, "read_video_cb"):
        raise RuntimeError("cv_reader.read_video_cb not available")

    frame_ids = [int(x) for x in frame_ids]
    pos_map: Dict[int, List[int]] = {}
    for i, fid in enumerate(frame_ids):
        pos_map.setdefault(fid, []).append(i)

    out: List[Optional[Dict[str, Any]]] = [None] * len(frame_ids)
    max_fid = max(frame_ids)

    def all_done() -> bool:
        for q in pos_map.values():
            if q:
                return False
        return True

    def cb(d: Dict[str, Any]):
        idx = int(d.get("frame_idx", -1))
        if idx in pos_map and pos_map[idx]:
            j = pos_map[idx].pop(0)
            mv = np.asarray(d["motion_vector"])
            me = np.asarray(d["motion_energy"]) if "motion_energy" in d else None
            mem = np.asarray(d["motion_energy_median"]) if "motion_energy_median" in d else None
            # prefer residual_y; fallback residual
            if "residual_y" in d:
                ry = np.asarray(d["residual_y"])
            else:
                ry = np.asarray(d["residual"])
                if ry.ndim == 3:
                    ry = cv2.cvtColor(ry, cv2.COLOR_BGR2YUV)[:, :, 0]
            out_item: Dict[str, Any] = {"frame_idx": idx, "motion_vector": mv, "residual_y": ry}
            if me is not None:
                out_item["motion_energy"] = me
            if mem is not None:
                out_item["motion_energy_median"] = mem
            # Add pict_type if available
            if "pict_type" in d:
                out_item["pict_type"] = d["pict_type"]
            out[j] = out_item
        return (not all_done())

    # without_residual=0 means WITH residual (your earlier convention)
    without_residual = 0
    max_frames = int(max_fid) + 1
    # residual_rgb=0: keep residual_y only, skip expensive RGB residual conversion.
    cv_api.read_video_cb(str(video_path), cb, int(without_residual), int(max_frames), frame_ids, -1, 0, 0)

    # fill missing with nearest previous
    last = None
    for i in range(len(out)):
        if out[i] is None:
            if last is not None:
                out[i] = last
        else:
            last = out[i]
    if last is not None:
        for i in range(len(out)):
            if out[i] is None:
                out[i] = last

    return [x for x in out if x is not None]  # type: ignore


# -----------------------------
# main per-video worker
# -----------------------------
def process_one_video(
    video_path: str,
    key: str,
    out_root: str,
    seq_len: int,
    window_len_frames: int,
    num_candidates: int,
    top_k_windows: int,
    max_total_patches: int,
    patch: int,
    mv_unit_div: float,
    mv_pct: float,
    res_pct: float,
    w_mv: float,
    w_res: float,
    num_gops: int = 1,
    gop_anchor_mode: str = "sampled",
    max_total_patches_cap: int = 30000,
    mv_compensate: str = "none",  # {"none","median"}
    mv_dir_mode: str = "l0",
    w_mv_l0: float = 1.0,
    w_mv_l1: float = 1.0,
    fill_to_images: bool = True,
    force_fallback_no_cv_reader: bool = False,
    sample_id: str = "",
    caption: str = "",
    decode_backsearch_max: int = 32,
    per_video_timeout_sec: int = 0,
    group_by_time: bool = False,
    keep_blocks_within_canvas: bool = True,
    min_pixels: int = 56 * 56,
    max_pixels: int = 768 * 768,
    orig_messages: Optional[List[Dict[str, Any]]] = None,
    skip_black_frames: bool = True,
    skip_corrupt_frames: bool = True,
    black_y_mean_thr: float = 8.0,
    black_y_std_thr: float = 6.0,
    corrupt_green_frac_thr: float = 0.35,
    corrupt_g_thr: int = 180,
    corrupt_rb_thr: int = 90,
    ensure_per_second: bool = False,
    sec_stride: float = 1.0,
    min_blocks_per_second: int = 1,
    p_full_per_bucket: int = 3,
    auto_num_gops: bool = True,
    analysis_candidate_multiplier: float = 8.0,
    analysis_max_frames: int = 1024,
    block_selection_frame_penalty: float = 0.35,
    mv_source: str = "vector",
    score_source: str = "mvres",
    bitcost_grid: str = "sub",
    bitcost_pct: float = 99.0,
    bitcost_log_scale: bool = True,
    decode_backend: str = "ffmpeg_native",
    parallel_segments: int = 0,
    threads_per_segment: int = 4,
    segment_guard_frames: int = 30,
    mask_letterbox: bool = True,
    letterbox_dark_thr: float = 16.0,
    collage_patch_order: str = "time",
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Returns (status, message, index_record). status in {"ok","skip","fail"}.
    index_record is returned for:
      - status=="ok": assets index record
      - status=="skip" with unsupported codec: unsupported codec record
    """
    # avoid OpenCV oversubscription inside multiprocessing
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    # Reduce FFmpeg/OpenCV log spam on corrupted streams (can become a major I/O bottleneck).
    # This affects OpenCV's internal FFmpeg backend in many builds.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "loglevel;error")

    vp = str(video_path)
    if not vp or (not Path(vp).exists()):
        return "skip", f"{key} missing video", None

    score_source = str(score_source).lower().strip()
    if score_source not in {"mvres", "bitcost"}:
        score_source = "mvres"
    auto_num_gops = bool(auto_num_gops)
    analysis_candidate_multiplier = float(max(1.0, float(analysis_candidate_multiplier)))
    analysis_max_frames = int(max(64, int(analysis_max_frames)))
    block_selection_frame_penalty = float(max(0.0, float(block_selection_frame_penalty)))
    bitcost_grid = str(bitcost_grid).lower().strip()
    if bitcost_grid not in {"sub", "mb", "ctu", "adaptive", "auto"}:
        bitcost_grid = "sub"
    decode_backend = str(decode_backend).lower().strip()
    if decode_backend == "opencv":
        return "fail", f"{key} opencv decode backend has been removed; use ffmpeg_native", None
    if decode_backend not in {"auto", "ffmpeg_native"}:
        decode_backend = "ffmpeg_native"
    if decode_backend == "auto":
        decode_backend = "ffmpeg_native"

    collage_patch_order = str(collage_patch_order).lower().strip()
    if collage_patch_order == "wh":
        collage_patch_order = "score"
    if collage_patch_order not in {"time", "score"}:
        collage_patch_order = "time"

    codec_name = ffprobe_video_codec_name(vp)
    if codec_name not in {"h264", "hevc"}:
        out_dir = str(Path(out_root) / key)
        rec = {
            "kind": "unsupported_codec",
            "status": "skip",
            "key": str(key),
            "video": str(vp),
            "out_dir": str(out_dir),
            "codec_name": str(codec_name) if codec_name else "unknown",
            "supported_codecs": ["h264", "hevc"],
        }
        return "skip", f"{key} unsupported codec={rec['codec_name']}", rec

    out_dir = str(Path(out_root) / key)
    done_mark = str(Path(out_dir) / "_DONE")
    if Path(done_mark).exists():
        return "skip", f"{key} already done", None

    ensure_dir(out_dir)

    total_frames, fps, H0, W0 = get_total_frames_fps(vp)
    if total_frames <= 0:
        return "fail", f"{key} cannot read total_frames", None

    # -----------------------------
    # Adaptive sampling (Scheme 2 + 3)
    #   - seq_len adapts to output num_images (multiple of 4): seq_len = clamp(4*num_images, 64, 256)
    #   - window_len_frames/num_candidates/top_k_windows adapt to duration for more even coverage
    # This happens automatically so callers do not need to change CLI.
    # -----------------------------
    fps_use = float(fps) if (fps and fps > 0) else 30.0
    duration_sec = float(total_frames) / float(fps_use) if total_frames > 0 else 0.0

    if auto_num_gops or int(num_gops) <= 0:
        num_gops_auto = _clamp_int(round(duration_sec / 8.0), 1, 16)
        num_gops = int(min(num_gops_auto, max(1, total_frames)))
    else:
        num_gops = int(max(1, int(num_gops)))

    # Estimate padded token grid size S_full without decoding.
    # Use smart_resize on original H/W and then pad to (2*patch) to ensure hb/wb even.
    p_est = int(patch)
    pad_base_est = int(2 * p_est)
    try:
        resize_h_est, resize_w_est = smart_resize(
            height=int(H0),
            width=int(W0),
            factor=int(pad_base_est),
            min_pixels=int(min_pixels),
            max_pixels=int(max_pixels),
        )
    except Exception:
        resize_h_est, resize_w_est = int(H0), int(W0)

    pad_bottom_est = (pad_base_est - (int(resize_h_est) % pad_base_est)) % pad_base_est
    pad_right_est = (pad_base_est - (int(resize_w_est) % pad_base_est)) % pad_base_est
    H1_est = int(resize_h_est) + int(pad_bottom_est)
    W1_est = int(resize_w_est) + int(pad_right_est)
    hb_est = int(H1_est // max(1, p_est))
    wb_est = int(W1_est // max(1, p_est))
    # Ensure even grid (should hold by construction)
    if hb_est % 2 != 0:
        hb_est += 1
    if wb_est % 2 != 0:
        wb_est += 1
    S_full_est = int(max(1, hb_est * wb_est))

    # Estimate num_images from budget (auto budget if max_total_patches<=0; otherwise from provided budget).
    cap_total = int(max_total_patches_cap) if int(max_total_patches_cap) > 0 else 30000
    max_total_in = int(max_total_patches)
    if max_total_in <= 0:
        max_total_est, _dbg_est = auto_max_total_patches(
            S_full=S_full_est,
            total_frames=int(total_frames),
            fps=float(fps_use),
            cap_total=int(cap_total),
        )
    else:
        # If user provided a budget, respect it but cap it.
        max_total_est = int(min(int(max_total_in), int(cap_total)))

    num_images_est = int(max(1, int(max_total_est // max(1, S_full_est))))
    # Enforce 4-image grouping and minimum GOP coverage.
    num_images_est = max(4, int(num_images_est // 4) * 4)
    num_images_est = int(max(int(num_gops) * 4 if int(num_gops) > 0 else 4, int(num_images_est)))

    # Scheme 2: seq_len derived from num_images (candidate pool grows with output)
    seq_len_adapt = _clamp_int(float(analysis_candidate_multiplier) * float(num_images_est), 64, int(analysis_max_frames))
    # Do not exceed total_frames too aggressively; duplicates are ok, but keep it bounded.
    if int(total_frames) > 0:
        seq_len_adapt = int(min(int(seq_len_adapt), int(max(16, total_frames))))

    # Scheme 3: window params adapt to duration (more even coverage for long videos)
    if duration_sec <= 15.0:
        win_sec = 6.0
    elif duration_sec <= 60.0:
        win_sec = 10.0
    else:
        win_sec = 12.0

    window_len_frames_adapt = _round_to_multiple(float(fps_use) * float(win_sec), 16)
    window_len_frames_adapt = int(max(32, min(int(window_len_frames_adapt), int(total_frames))))

    num_candidates_adapt = int(max(5, min(20, int(math.ceil(duration_sec / 5.0)))))
    top_k_windows_adapt = int(max(2, min(8, int(math.ceil(duration_sec / 20.0)))))
    top_k_windows_adapt = int(min(int(top_k_windows_adapt), int(num_candidates_adapt)))

    # Override user-provided sampling parameters with adaptive ones.
    seq_len = int(seq_len_adapt)
    window_len_frames = int(window_len_frames_adapt)
    num_candidates = int(num_candidates_adapt)
    top_k_windows = int(top_k_windows_adapt)

    # 1) sample frames on an information-reweighted timeline.
    # High-energy regions get denser candidate frames; low-energy regions get fewer.
    # This is equivalent to constructing a new non-uniform video timeline and then
    # sampling candidate frames from that timeline.
    if int(total_frames) <= 0:
        windows = [(0, 0, 0)]
        frame_ids = [0] * int(seq_len)
        win_dbg = {
            "mode": "uniform_empty_fallback",
            "total_frames": int(total_frames),
            "fps": float(fps),
            "seq_len": int(seq_len),
            "window_len_frames_input": int(window_len_frames),
            "num_candidates_input": int(num_candidates),
            "top_k_windows_input": int(top_k_windows),
            "note": "Empty-video fallback.",
        }
    else:
        windows = [(0, int(total_frames) - 1, 0)]
        frame_ids, energy_cdf_dbg = sample_frame_ids_by_energy_cdf(
            video_path=str(vp),
            total_frames=int(total_frames),
            fps=float(fps_use),
            target_count=int(seq_len),
            bin_sec=0.5,
            smooth_bins=1,
            uniform_mix=0.15,
            max_per_bin=16,
        )
        win_dbg = {
            "mode": "pkt_energy_cdf_timeline",
            "total_frames": int(total_frames),
            "fps": float(fps),
            "seq_len": int(seq_len),
            "window_len_frames_input": int(window_len_frames),
            "num_candidates_input": int(num_candidates),
            "top_k_windows_input": int(top_k_windows),
            "note": "Candidate frames are sampled on a PB packet-size energy reweighted timeline instead of uniformly on the raw timeline.",
            "energy_cdf_debug": energy_cdf_dbg,
        }

    # ---- pkt_size dense peaks (PB only, exclude keyframes) ----
    # After the energy-CDF candidate sampling above, we still inject a small number
    # of local peak-centered frames so sudden bursts are less likely to be missed.
    pkt_dense = True
    pkt_bin_sec = 0.5
    pkt_peaks_cap = 8
    pkt_peaks_per_sec = 0.5  # ~1 peak / 2s
    pkt_peak_neighbor = 1
    pkt_smooth_bins = 1

    pkt_dbg: Optional[Dict[str, Any]] = None
    if bool(pkt_dense) and int(total_frames) > 0:
        try:
            peak_fids, pkt_dbg = pick_peak_frame_ids_from_pkt_size(
                video_path=str(vp),
                fps=float(fps_use),
                total_frames=int(total_frames),
                bin_sec=float(pkt_bin_sec),
                peaks_cap=int(pkt_peaks_cap),
                peaks_per_sec=float(pkt_peaks_per_sec),
                neighbor=int(pkt_peak_neighbor),
                smooth_bins=int(pkt_smooth_bins),
            )
            if peak_fids:
                frame_ids = sorted(set([int(x) for x in frame_ids] + [int(x) for x in peak_fids]))
                win_dbg["pkt_dense"] = True
                win_dbg["pkt_added"] = int(len(peak_fids))
                win_dbg["frame_ids_len_after_pkt"] = int(len(frame_ids))
                win_dbg["frame_ids_preview_after_pkt"] = [int(x) for x in frame_ids[:32]]
            else:
                win_dbg["pkt_dense"] = True
                win_dbg["pkt_added"] = 0
        except Exception as e:
            win_dbg["pkt_dense"] = True
            win_dbg["pkt_error"] = repr(e)

    # Save pkt debug into window_debug so meta.json carries it automatically.
    if pkt_dbg is not None:
        win_dbg["pkt_debug"] = pkt_dbg

    # Cap candidate frames to keep runtime stable; prefer keeping peak frames.
    MAX_CAND = int(max(64, int(analysis_max_frames)))
    if len(frame_ids) > MAX_CAND:
        peak_keep: set = set()
        if isinstance(pkt_dbg, dict):
            try:
                peak_keep = set(int(x) for x in pkt_dbg.get("peak_frame_ids", []) if isinstance(x, (int, float)))
            except Exception:
                peak_keep = set()
        peak_sorted = sorted([x for x in frame_ids if x in peak_keep])
        rest = [x for x in frame_ids if x not in peak_keep]
        need = MAX_CAND - len(peak_sorted)
        if need <= 0:
            frame_ids = peak_sorted[:MAX_CAND]
        else:
            if len(rest) <= need:
                frame_ids = sorted(peak_sorted + rest)
            else:
                idxs = np.linspace(0, len(rest) - 1, need, dtype=np.int32).tolist()
                picked = [rest[int(i)] for i in idxs]
                frame_ids = sorted(peak_sorted + picked)
        win_dbg["frame_ids_len_capped"] = int(len(frame_ids))
        win_dbg["frame_ids_cap"] = int(MAX_CAND)
    win_dbg["frame_ids_preview"] = [int(x) for x in frame_ids[:32]]

    # IMPORTANT: downstream uses seq_len as the final candidate-frame count after
    # information-reweighted sampling, peak injection, and candidate capping.
    seq_len = int(len(frame_ids))

    # 2) optional temporal coverage refinement
    if bool(ensure_per_second):
        frame_ids = enforce_time_coverage_frame_ids(
            frame_ids=frame_ids,
            total_frames=int(total_frames),
            fps=float(fps),
            seq_len=int(seq_len),
            stride_sec=float(sec_stride),
        )
    # ===== GOP anchor pre-alignment (sampled / keyframe / hybrid) =====
    # V2.1: build variable-length GOPs by cumulative PB packet-size energy and use
    # their anchor positions to seed GOP anchors on the sampled frame timeline.
    gop_anchor_mode = str(gop_anchor_mode).lower().strip()
    if gop_anchor_mode not in {"sampled", "keyframe", "hybrid"}:
        gop_anchor_mode = "sampled"

    num_gops_seed = int(min(max(1, int(num_gops)), max(1, len(frame_ids))))

    variable_gop_segments, variable_gop_dbg = build_variable_length_gops_by_energy(
        video_path=str(vp),
        total_frames=int(total_frames),
        fps=float(fps_use),
        target_num_gops=int(num_gops_seed),
        bin_sec=0.5,
        smooth_bins=1,
        min_span_sec=1.5,
        max_span_sec=6.0,
    )

    anchor_t_seed: List[int] = []
    if len(frame_ids) > 0:
        for seg in variable_gop_segments:
            anchor_sec = float(seg.get("anchor_sec", 0.0))
            target_fid = int(round(anchor_sec * float(fps_use)))
            if int(total_frames) > 0:
                target_fid = max(0, min(int(total_frames) - 1, int(target_fid)))
            best_t = min(range(len(frame_ids)), key=lambda ii: abs(int(frame_ids[ii]) - int(target_fid)))
            anchor_t_seed.append(int(best_t))

    if len(anchor_t_seed) == 0:
        anchor_t_seed = [0]
    anchor_t_seed = sorted(set(int(x) for x in anchor_t_seed))
    if 0 not in anchor_t_seed:
        anchor_t_seed = [0] + anchor_t_seed

    num_gops_seed = int(max(1, len(anchor_t_seed)))

    keyframe_align_dbg: Dict[str, Any] = {
        "mode": str(gop_anchor_mode),
        "anchor_t_seed": [int(x) for x in anchor_t_seed],
        "replaced": [],
        "keyframes_found": 0,
        "variable_gop_debug": variable_gop_dbg,
    }

    if gop_anchor_mode in {"keyframe", "hybrid"} and len(frame_ids) > 0:
        kf_fids = ffprobe_keyframe_frame_ids(vp, fps=float(fps), total_frames=int(total_frames))
        keyframe_align_dbg["keyframes_found"] = int(len(kf_fids))

        if len(kf_fids) > 0:
            replacements: List[Tuple[int, int, int]] = []

            if gop_anchor_mode == "keyframe":
                pick_idx = np.linspace(0, len(kf_fids) - 1, len(anchor_t_seed), dtype=np.int32).tolist()
                chosen_kfs = [int(kf_fids[int(i)]) for i in pick_idx]
            else:
                chosen_kfs = []
                kf_arr = np.asarray(kf_fids, dtype=np.int32)
                for t_anchor in anchor_t_seed:
                    target_fid = int(frame_ids[int(t_anchor)])
                    pos = int(np.searchsorted(kf_arr, target_fid, side="left"))
                    cands = []
                    if 0 <= pos < len(kf_arr):
                        cands.append(int(kf_arr[pos]))
                    if pos - 1 >= 0:
                        cands.append(int(kf_arr[pos - 1]))
                    if cands:
                        best = min(cands, key=lambda x: abs(int(x) - int(target_fid)))
                    else:
                        best = int(target_fid)
                    chosen_kfs.append(int(best))

            for t_anchor, kf_fid in zip(anchor_t_seed, chosen_kfs):
                t_anchor = int(t_anchor)
                old_fid = int(frame_ids[t_anchor])
                new_fid = int(max(0, min(int(total_frames) - 1, int(kf_fid)))) if int(total_frames) > 0 else int(kf_fid)
                frame_ids[t_anchor] = int(new_fid)
                replacements.append((int(t_anchor), int(old_fid), int(new_fid)))

            keyframe_align_dbg["replaced"] = [
                {"t": int(t), "old_fid": int(o), "new_fid": int(n)} for (t, o, n) in replacements
            ]
    # ===== end GOP anchor pre-alignment =====
    # 3) derive a stable token grid first. For bitcost mode we can analyze many more
    # candidate frames without decoding RGB up front.
    p = int(patch)
    pad_base = 2 * p
    pz = np.zeros((p, p, 3), dtype=np.uint8)

    resize_h = int(resize_h_est)
    resize_w = int(resize_w_est)
    H_dec = int(H0)
    W_dec = int(W0)
    do_resize = (int(resize_h) != int(H_dec)) or (int(resize_w) != int(W_dec))
    scale_h = float(resize_h) / float(max(1, H_dec))
    scale_w = float(resize_w) / float(max(1, W_dec))
    pad_bottom = int((pad_base - (int(resize_h) % pad_base)) % pad_base)
    pad_right = int((pad_base - (int(resize_w) % pad_base)) % pad_base)
    H1 = int(resize_h) + int(pad_bottom)
    W1 = int(resize_w) + int(pad_right)
    if H1 % p != 0 or W1 % p != 0:
        return "fail", f"{key} padded size not divisible by patch: H1={H1} W1={W1} patch={p}", None

    hb, wb = H1 // p, W1 // p
    if hb % 2 != 0 or wb % 2 != 0:
        return "fail", f"{key} patch grid not even after pad: hb={hb} wb={wb}", None

    frames_bgr_pad: List[np.ndarray] = []
    good_mask = [True] * int(len(frame_ids))

    # 5) fetch scoring features
    use_cv = (HAS_CV_READER and (not force_fallback_no_cv_reader))
    cv_reader_fallback_cause = "none"
    if not HAS_CV_READER:
        cv_reader_fallback_cause = "cv_reader_unavailable"
    elif force_fallback_no_cv_reader:
        cv_reader_fallback_cause = "force_no_cv_reader"
    items: Optional[List[Dict[str, Any]]] = None
    if use_cv:
        try:
            if score_source == "bitcost":
                items = cv_reader_fetch_bitcost(
                    vp, frame_ids,
                    parallel_segments=int(parallel_segments),
                    threads_per_segment=int(threads_per_segment),
                    segment_guard_frames=int(segment_guard_frames),
                )
            else:
                items = cv_reader_fetch_mvres(vp, frame_ids)
        except Exception:
            items = None
            use_cv = False
            cv_reader_fallback_cause = "fetch_exception"
    # Safety: some corrupted streams may cause cv_reader to miss requested frame indices.
    if use_cv and items is not None and len(items) != len(frame_ids):
        items = None
        use_cv = False
        cv_reader_fallback_cause = "len_mismatch"

    residual_fallback_frames = 0
    residual_fallback_by_reason: Dict[str, int] = {
        "cv_reader_unavailable": 0,
        "force_no_cv_reader": 0,
        "fetch_exception": 0,
        "len_mismatch": 0,
        "other_cv_reader_unavailable_or_misaligned": 0,
        "frame_exception": 0,
    }

    # fallback scoring uses residual proxy centered at 128
    fused_maps: List[np.ndarray] = []
    top_lb, bottom_lb, left_lb, right_lb = 0, int(H1), 0, int(W1)
    content_mask = np.ones((int(H1), int(W1)), dtype=np.float32)
    mv_source = str(mv_source).lower().strip()
    if mv_source not in {"vector", "energy", "energy_median", "auto"}:
        mv_source = "vector"
    if score_source != "bitcost":
        frames_bgr = decode_frames_bgr(
            vp,
            frame_ids,
            backsearch_max=int(decode_backsearch_max),
            backend=decode_backend,
        )

        good_mask = []
        for fr in frames_bgr:
            bad = frame_is_bad(
                fr,
                skip_black_frames=bool(skip_black_frames),
                skip_corrupt_frames=bool(skip_corrupt_frames),
                black_y_mean_thr=float(black_y_mean_thr),
                black_y_std_thr=float(black_y_std_thr),
                solid_y_std_thr=4.0,
                solid_color_std_thr=4.0,
                solid_max_range_thr=12.0,
                corrupt_green_frac_thr=float(corrupt_green_frac_thr),
                corrupt_g_thr=int(corrupt_g_thr),
                corrupt_rb_thr=int(corrupt_rb_thr),
            )
            good_mask.append(not bool(bad))

        if (len(good_mask) > 0) and (not good_mask[0]):
            first_good = None
            for i_g in range(1, len(good_mask)):
                if good_mask[i_g]:
                    first_good = i_g
                    break
            if first_good is not None:
                frames_bgr[0], frames_bgr[first_good] = frames_bgr[first_good], frames_bgr[0]
                frame_ids[0], frame_ids[first_good] = int(frame_ids[first_good]), int(frame_ids[0])
                good_mask[0], good_mask[first_good] = good_mask[first_good], good_mask[0]
            else:
                return "skip", f"{key} all decoded frames are black/pure-color/corrupted", None

        frames_bgr_rs: List[np.ndarray] = []
        if do_resize:
            for fr in frames_bgr:
                frames_bgr_rs.append(_resize_bgr(fr, int(resize_h), int(resize_w)))
        else:
            frames_bgr_rs = frames_bgr

        for i, fr in enumerate(frames_bgr_rs):
            frp, (pb, pr) = pad_to_multiple_of_bgr(fr, pad_base)
            frames_bgr_pad.append(frp)
            if i == 0:
                pad_bottom, pad_right = pb, pr

        top_lb, bottom_lb, left_lb, right_lb = detect_letterbox_bbox_bgr(
            frames_bgr_pad[0], dark_thr=float(letterbox_dark_thr)
        )
        content_mask = np.zeros((int(H1), int(W1)), dtype=np.float32)
        content_mask[int(top_lb):int(bottom_lb), int(left_lb):int(right_lb)] = 1.0

    for t in range(len(frame_ids)):
        fr = frames_bgr_pad[t] if (score_source != "bitcost" and t < len(frames_bgr_pad)) else None
        if use_cv and items is not None and t < len(items):
            try:
                if score_source == "bitcost":
                    fused = bitcost_item_to_score_map(
                        items[t],
                        out_h=int(H1),
                        out_w=int(W1),
                        grid=bitcost_grid,
                        pct=float(bitcost_pct),
                        log_scale=bool(bitcost_log_scale),
                        codec_name=codec_name,
                    )
                else:
                    mv = np.asarray(items[t]["motion_vector"])
                    ry = np.asarray(items[t]["residual_y"])
                    me_raw = np.asarray(items[t]["motion_energy"]) if "motion_energy" in items[t] else None
                    mem_raw = np.asarray(items[t]["motion_energy_median"]) if "motion_energy_median" in items[t] else None
                    if mv_source == "energy":
                        me = me_raw
                    elif mv_source == "energy_median":
                        me = mem_raw
                    elif mv_source == "auto":
                        me = mem_raw if mem_raw is not None else me_raw
                    else:
                        me = None

                    # Resize mv/residual to match resized RGB (pixel-aligned assumption)
                    if do_resize:
                        ry = _resize_gray(ry, int(resize_h), int(resize_w))
                        mv = _resize_mv_and_scale(mv, int(resize_h), int(resize_w), scale_h=scale_h, scale_w=scale_w)
                        if me is not None:
                            me = _resize_gray(me.astype(np.float32), max(1, int(resize_h) // 4), max(1, int(resize_w) // 4))

                    # Pad residual_y to match padded RGB
                    if ry.shape[0] != H1 or ry.shape[1] != W1:
                        ry_pad = cv2.copyMakeBorder(
                            ry,
                            0,
                            pad_bottom,
                            0,
                            pad_right,
                            borderType=cv2.BORDER_CONSTANT,
                            value=128,
                        )
                    else:
                        ry_pad = ry
                    fused = mv_res_score_map(
                        mv=mv,
                        res_y=ry_pad,
                        mv_energy=me,
                        mv_unit_div=mv_unit_div,
                        mv_pct=mv_pct,
                        res_pct=res_pct,
                        w_mv=w_mv,
                        w_res=w_res,
                        mv_compensate=mv_compensate,
                        mv_dir_mode=mv_dir_mode,
                        w_mv_l0=w_mv_l0,
                        w_mv_l1=w_mv_l1,
                    )
                if bool(mask_letterbox):
                    fused = (fused * content_mask).astype(np.float32)
            except Exception:
                # Any mismatch -> fall back for this frame
                residual_fallback_frames += 1
                residual_fallback_by_reason["frame_exception"] += 1
                if fr is None:
                    fused = np.zeros((int(H1), int(W1)), dtype=np.float32)
                    if bool(mask_letterbox):
                        fused = (fused * content_mask).astype(np.float32)
                    fused_maps.append(fused.astype(np.float32))
                    continue
                ry = bgr_to_residual_y_u8(fr)
                fused = mv_res_score_map(
                    mv=np.zeros((1, 1, 2), dtype=np.float32),
                    res_y=ry,
                    mv_unit_div=mv_unit_div,
                    mv_pct=mv_pct,
                    res_pct=res_pct,
                    w_mv=0.0,
                    w_res=1.0,
                    mv_compensate="none",
                    mv_dir_mode="l0",
                    w_mv_l0=1.0,
                    w_mv_l1=1.0,
                )
                if bool(mask_letterbox):
                    fused = (fused * content_mask).astype(np.float32)
        else:
            # fallback: use "residual_y proxy" only
            residual_fallback_frames += 1
            if cv_reader_fallback_cause in residual_fallback_by_reason:
                residual_fallback_by_reason[cv_reader_fallback_cause] += 1
            else:
                residual_fallback_by_reason["other_cv_reader_unavailable_or_misaligned"] += 1
            if fr is None:
                fused = np.zeros((int(H1), int(W1)), dtype=np.float32)
                if bool(mask_letterbox):
                    fused = (fused * content_mask).astype(np.float32)
                fused_maps.append(fused.astype(np.float32))
                continue
            ry = bgr_to_residual_y_u8(fr)
            fused = mv_res_score_map(
                mv=np.zeros((1, 1, 2), dtype=np.float32),
                res_y=ry,
                mv_unit_div=mv_unit_div,
                mv_pct=mv_pct,
                res_pct=res_pct,
                w_mv=0.0,
                w_res=1.0,
                mv_compensate="none",
                mv_dir_mode="l0",
                w_mv_l0=1.0,
                w_mv_l1=1.0,
            )
            if bool(mask_letterbox):
                fused = (fused * content_mask).astype(np.float32)
        fused_maps.append(fused.astype(np.float32))

    # 6) decide token budget
    S_full = hb * wb  # patches per full image/canvas

    # How many GOP anchors (each anchor contributes one full I-frame canvas)
    p_full_per_bucket = int(max(0, int(p_full_per_bucket)))
    images_per_bucket = int(1 + p_full_per_bucket)  # I_full + P_full_* canvases
    num_gops_req = int(max(1, len(keyframe_align_dbg.get("anchor_t_seed", []))))

    # To avoid buckets degenerating (especially the last bucket), we require that each bucket
    # has enough non-anchor frames to supply P canvases.
    # Each full P canvas needs `blocks_per_full_canvas = S_full/4` blocks.
    # One frame can supply at most `max_blocks = (hb//2)*(wb//2)` blocks.
    blocks_per_full_canvas = int(S_full // 4)
    max_blocks = int((hb // 2) * (wb // 2))
    needed_blocks_per_bucket = int(p_full_per_bucket * blocks_per_full_canvas)
    min_good_frames_per_bucket = int(math.ceil(float(needed_blocks_per_bucket) / float(max(1, max_blocks))))
    min_good_frames_per_bucket = max(1, int(min_good_frames_per_bucket))

    # Need at least `min_good_frames_per_bucket` frames strictly between anchors (or anchor and end).
    # So anchor gap in t-space must be >= (min_good_frames_per_bucket + 1).
    min_anchor_gap_t = int(min_good_frames_per_bucket + 1)

    # Start from the variable-length GOP anchors built above. If spacing is too tight for
    # the current bucket budget, thin them out while preserving order and the first anchor.
    anchor_t_seed_eff = [int(x) for x in keyframe_align_dbg.get("anchor_t_seed", [0])]
    anchor_t_seed_eff = sorted(set(anchor_t_seed_eff)) if anchor_t_seed_eff else [0]
    filtered_anchor_t_seed: List[int] = []
    last_keep: Optional[int] = None
    for t_anchor in anchor_t_seed_eff:
        t_anchor = int(t_anchor)
        if last_keep is None:
            filtered_anchor_t_seed.append(int(t_anchor))
            last_keep = int(t_anchor)
            continue
        if (int(t_anchor) - int(last_keep)) >= int(min_anchor_gap_t):
            filtered_anchor_t_seed.append(int(t_anchor))
            last_keep = int(t_anchor)
    if len(filtered_anchor_t_seed) == 0:
        filtered_anchor_t_seed = [0]
    keyframe_align_dbg["anchor_t_seed_before_spacing_filter"] = [int(x) for x in anchor_t_seed_eff]
    keyframe_align_dbg["anchor_t_seed_after_spacing_filter"] = [int(x) for x in filtered_anchor_t_seed]

    # Latest valid anchor t so the last bucket still has enough frames:
    # len(frame_ids) - t_anchor - 1 >= min_good_frames_per_bucket
    max_anchor_t = int(max(0, len(frame_ids) - 1 - int(min_good_frames_per_bucket)))

    eligible_anchor_count = int(max(1, (max_anchor_t // max(1, min_anchor_gap_t)) + 1))
    if len(filtered_anchor_t_seed) > 0:
        filtered_anchor_t_seed = [int(x) for x in filtered_anchor_t_seed if int(x) <= int(max_anchor_t)]
        if len(filtered_anchor_t_seed) == 0:
            filtered_anchor_t_seed = [0]
        num_gops_eff = int(min(int(len(filtered_anchor_t_seed)), eligible_anchor_count))
        keyframe_align_dbg["anchor_t_seed_final"] = [int(x) for x in filtered_anchor_t_seed[:num_gops_eff]]
        keyframe_align_dbg["num_gops_eff"] = int(num_gops_eff)

    cap_total = int(max_total_patches_cap) if int(max_total_patches_cap) > 0 else 30000
    max_total = int(max_total_patches)
    auto_budget_dbg: Optional[Dict[str, Any]] = None

    if max_total <= 0:
        # auto budget based on resolution + duration, hard-capped by cap_total
        max_total, auto_budget_dbg = auto_max_total_patches(
            S_full=S_full, total_frames=total_frames, fps=fps, cap_total=cap_total
        )

    # Convert patch budget into image budget. Budget may reduce bucket count, but must
    # not create new buckets beyond the GOP anchors derived above.
    num_images_budget = int(max(1, int(max_total // max(1, S_full))))
    if bool(fill_to_images):
        num_images_budget = max(4, int(math.ceil(float(num_images_budget) / 4.0)) * 4)
    else:
        num_images_budget = max(1, int(num_images_budget))

    if auto_budget_dbg is not None and cap_total > 0:
        num_images_cap = max(1, int(cap_total // max(1, S_full)))
        if bool(fill_to_images):
            num_images_cap = max(4, int(num_images_cap // 4) * 4)
        num_images_budget = min(int(num_images_budget), int(max(1, num_images_cap)))

    # Keep the per-bucket canvas recipe fixed as 1I + p_full_per_bucket * P.
    # Budget only controls how many buckets survive, not how many P canvases a kept
    # bucket receives.
    max_bucket_by_budget = max(1, int(num_images_budget // max(1, images_per_bucket)))
    num_gops_eff = int(min(num_gops_req, max_bucket_by_budget, eligible_anchor_count))

    def build_even_spaced_anchors(n_anchor: int) -> List[int]:
        """Scheme A: anchors approximate equal bucket lengths in t-space.

        For n_anchor buckets over T=len(frame_ids), ideal anchors are at
          t_i ≈ round(i * T / n_anchor), i=0..n_anchor-1
        Then we enforce:
          - anchors are increasing
          - spacing >= min_anchor_gap_t
          - last anchor <= max_anchor_t (so tail bucket has enough frames)
        """
        n_anchor = int(max(1, n_anchor))
        if n_anchor <= 1:
            return [0]

        T = int(len(frame_ids))
        gap = int(max(1, min_anchor_gap_t))

        # 1) ideal equally-sized buckets in t-space
        anchors = [int(round(float(i) * float(T) / float(n_anchor))) for i in range(n_anchor)]
        anchors[0] = 0

        # 2) forward enforce spacing + clamp to max_anchor_t
        out: List[int] = [0]
        for i in range(1, n_anchor):
            t = int(anchors[i])
            # clamp by allowed max anchor
            t = min(int(max_anchor_t), max(0, t))
            # enforce spacing
            t = max(int(out[-1] + gap), int(t))
            # if clamped too hard, keep it at least out[-1]+gap, but not beyond max_anchor_t
            if t > int(max_anchor_t):
                t = int(max_anchor_t)
            out.append(int(t))

        # 3) backward pass to ensure last<=max_anchor_t and spacing
        out[-1] = min(int(out[-1]), int(max_anchor_t))
        for i in range(len(out) - 2, -1, -1):
            if int(out[i + 1]) - int(out[i]) < gap:
                out[i] = int(out[i + 1] - gap)
        out[0] = 0

        # 4) sanitize: clamp to [0, max_anchor_t] and strictly increasing
        cleaned: List[int] = []
        for t in out:
            t = min(int(max_anchor_t), max(0, int(t)))
            if not cleaned:
                cleaned.append(int(t))
            else:
                if int(t) <= int(cleaned[-1]):
                    continue
                cleaned.append(int(t))

        # 5) if we lost anchors due to constraints, fill in additional anchors with gap step
        if len(cleaned) < n_anchor:
            used = set(cleaned)
            # try to insert from left to right
            t_cand = 0
            while len(cleaned) < n_anchor and t_cand <= int(max_anchor_t):
                if t_cand not in used:
                    if not cleaned or (t_cand - cleaned[-1] >= gap):
                        cleaned.append(int(t_cand))
                        used.add(int(t_cand))
                t_cand += gap
            cleaned = sorted(cleaned)

        if not cleaned:
            cleaned = [0]
        if len(cleaned) > n_anchor:
            cleaned = cleaned[: int(n_anchor)]
        if cleaned[0] != 0:
            cleaned[0] = 0
        return cleaned

    def sample_existing_anchors_evenly(anchors: List[int], n_keep: int) -> List[int]:
        """Keep n anchors by uniformly subsampling the existing anchor sequence."""
        anchors = sorted(set(int(x) for x in anchors))
        n_keep = int(max(1, n_keep))
        if not anchors:
            return [0]
        if len(anchors) <= n_keep:
            return anchors
        if n_keep == 1:
            return [int(anchors[0])]

        idxs = np.linspace(0, len(anchors) - 1, n_keep, dtype=np.int32).tolist()
        kept = [int(anchors[int(i)]) for i in idxs]
        kept = sorted(set(kept))

        # Backfill if dedup shrank the list.
        if len(kept) < n_keep:
            used = set(kept)
            for a in anchors:
                if int(a) in used:
                    continue
                kept.append(int(a))
                used.add(int(a))
                if len(kept) >= n_keep:
                    break
            kept = sorted(kept)

        if kept[0] != int(anchors[0]):
            kept[0] = int(anchors[0])
        if len(kept) > n_keep:
            kept = kept[: int(n_keep)]
        return kept

    # Choose effective GOP anchors with guaranteed spacing >= min_anchor_gap_t.
    # Then, if bucket density is too high, reduce bucket count until each bucket
    # has enough non-anchor good frames to form P canvases without repeating blocks.

    # Use filtered variable-length GOP anchors as starting point. When budget forces us
    # to keep fewer buckets, subsample the full anchor set evenly instead of taking a prefix.
    anchor_seed_final = [int(x) for x in keyframe_align_dbg.get("anchor_t_seed_final", [0])]
    anchor_t_indices = sample_existing_anchors_evenly(anchor_seed_final, int(num_gops_eff))
    if len(anchor_t_indices) == 0:
        anchor_t_indices = [0]
    while len(anchor_t_indices) > 1:
        ok_density = True
        for bi, t_anchor in enumerate(anchor_t_indices):
            t_start = int(t_anchor)
            t_end = int(anchor_t_indices[bi + 1]) if (bi + 1) < len(anchor_t_indices) else int(len(frame_ids))
            good_cnt = 0
            for t in range(int(t_start) + 1, int(t_end)):
                if (t < len(good_mask)) and bool(good_mask[t]):
                    good_cnt += 1
            if good_cnt < int(min_good_frames_per_bucket):
                ok_density = False
                break
        if ok_density:
            break
        num_gops_eff = int(max(1, len(anchor_t_indices) - 1))
        anchor_t_indices = sample_existing_anchors_evenly(anchor_seed_final, int(num_gops_eff))
        if len(anchor_t_indices) == 0:
            anchor_t_indices = [0]

    # Post-process tail bucket: if the last bucket is much shorter than the auto-GOP
    # target, merge it back into the previous bucket to avoid tiny tail GOPs.
    if len(anchor_t_indices) >= 2:
        tail_span_t = int(len(frame_ids) - int(anchor_t_indices[-1]))
        tail_span_sec = float(tail_span_t) / float(max(fps_use, 1e-6))
        target_bucket_sec = 8.0 if bool(auto_num_gops) else max(1.5, float(duration_sec) / float(max(1, len(anchor_t_indices))))
        min_tail_merge_sec = max(3.0, 0.5 * float(target_bucket_sec))
        if tail_span_sec < float(min_tail_merge_sec):
            anchor_t_indices = anchor_t_indices[:-1]
            if len(anchor_t_indices) == 0:
                anchor_t_indices = [0]

    anchor_set = set(int(x) for x in anchor_t_indices)

    num_buckets = int(len(anchor_t_indices))
    num_images_budget = int(num_buckets * int(images_per_bucket))

    # available blocks per frame
    max_blocks = int((hb // 2) * (wb // 2))

    # 7) build output token sequence
    patches_list: List[np.ndarray] = []
    src_pos_list: List[List[int]] = []       # [frame_id, src_patch_h, src_patch_w]
    src_frameid_list: List[int] = []         # original frame id per patch
    src_ts_list: List[float] = []            # timestamp (seconds) per patch (pad tokens are -1.0)
    canvas_bucket_ids: List[int] = []
    canvas_roles: List[str] = []
    bucket_ranges_t: List[Dict[str, int]] = []

    fps_use_ts = float(fps) if (fps and fps > 0) else 30.0

    # helper to append one frame's patches (in a specific block order)
    def append_frame_patches(src_img_idx: int, frame_rgb: np.ndarray, blocks: List[Tuple[int, int]]):
        # NOTE: src_patch_position.npy first column stores the ORIGINAL frame_id (not the 0..seq_len-1 index).
        try:
            src_fid = int(frame_ids[int(src_img_idx)])
        except Exception:
            src_fid = -1
        # Place blocks in the order they appear in the input list.
        for (bh, bw) in blocks:
            coords = block_to_4_patches(bh, bw)
            for (ph, pw) in coords:
                patch_rgb = extract_patch_rgb(frame_rgb, ph, pw, patch=p)
                patches_list.append(patch_rgb.astype(np.uint8))
                # src_patch_position: [frame_id, patch_h, patch_w]
                src_pos_list.append([int(src_fid), int(ph), int(pw)])
                # keep src_frame_ids.npy consistent
                src_frameid_list.append(int(src_fid))
                if src_fid >= 0:
                    src_ts_list.append(float(src_fid) / float(fps_use_ts))
                else:
                    src_ts_list.append(-1.0)

    def append_pad_block():
        for _ in range(4):
            patches_list.append(pz)
            src_pos_list.append([-1, -1, -1])
            src_frameid_list.append(-1)
            src_ts_list.append(-1.0)

    # GOP buckets: each bucket contributes one full I canvas + K full P canvases.
    all_blocks = list(iter_blocks_in_raster(hb, wb))
    blocks_per_full_canvas = int(S_full // 4)

    selected_blocks_by_frame: Dict[int, List[Tuple[int, int]]] = {}
    selected_blocks_count_by_frame: Dict[int, int] = {}
    bucket_plans: List[Dict[str, Any]] = []

    def select_bucket_blocks_stratified(
        candidates: List[Tuple[int, int, int, float]],
        t_start: int,
        t_end: int,
        extra_images: int,
        blocks_per_canvas: int,
    ) -> Tuple[List[Tuple[int, int, int, float]], List[Dict[str, Any]]]:
        """Select P blocks using information-weighted temporal focus regions."""
        if extra_images <= 0 or blocks_per_canvas <= 0 or not candidates:
            return [], []

        t_lo = int(t_start) + 1
        t_hi = int(max(t_lo, t_end))
        chosen_all: List[Tuple[int, int, int, float]] = []
        used_keys: set = set()
        frame_mass: Dict[int, float] = {}
        frame_peak: Dict[int, float] = {}

        for (t0, _, _, sc0) in candidates:
            t0 = int(t0)
            sc0 = float(sc0)
            frame_mass[t0] = float(frame_mass.get(t0, 0.0) + max(0.0, sc0))
            frame_peak[t0] = max(float(frame_peak.get(t0, 0.0)), max(0.0, sc0))

        def _take_from_pool(pool: List[Tuple[int, int, int, float]], need: int) -> List[Tuple[int, int, int, float]]:
            out: List[Tuple[int, int, int, float]] = []
            if need <= 0:
                return out
            pool_sorted = sorted(pool, key=lambda x: (-float(x[3]), int(x[0])))
            for rec in pool_sorted:
                key = (int(rec[0]), int(rec[1]), int(rec[2]))
                if key in used_keys:
                    continue
                out.append(rec)
                used_keys.add(key)
                if len(out) >= int(need):
                    break
            return out

        frames_sorted = sorted(int(t) for t in frame_mass.keys() if int(t) >= int(t_lo) and int(t) < int(t_hi))
        focus_ranges: List[Dict[str, Any]] = []
        if not frames_sorted:
            return [], focus_ranges

        weights = np.asarray(
            [max(1e-6, float(frame_mass.get(int(t), 0.0)) + 0.35 * float(frame_peak.get(int(t), 0.0))) for t in frames_sorted],
            dtype=np.float64,
        )
        weight_sum = float(weights.sum())
        if (not np.isfinite(weight_sum)) or weight_sum <= 0.0:
            weights = np.ones((len(frames_sorted),), dtype=np.float64)
            weight_sum = float(weights.sum())
        cdf = np.cumsum(weights) / float(weight_sum)

        for pi in range(int(extra_images)):
            q_lo = float(pi) / float(max(1, extra_images))
            q_hi = float(pi + 1) / float(max(1, extra_images))
            q_mid = 0.5 * (q_lo + q_hi)
            idx_mid = int(np.searchsorted(cdf, q_mid, side="left"))
            idx_mid = max(0, min(len(frames_sorted) - 1, idx_mid))
            idx_lo = int(np.searchsorted(cdf, q_lo, side="left"))
            idx_hi = int(np.searchsorted(cdf, q_hi, side="left"))
            idx_lo = max(0, min(len(frames_sorted) - 1, idx_lo))
            idx_hi = max(idx_lo, min(len(frames_sorted) - 1, idx_hi))

            center_t = int(frames_sorted[idx_mid])
            range_lo_t = int(frames_sorted[idx_lo])
            range_hi_t = int(frames_sorted[idx_hi]) + 1
            local_pool = [
                rec for rec in candidates
                if int(rec[0]) >= int(range_lo_t) and int(rec[0]) < int(range_hi_t)
            ]
            picked = _take_from_pool(local_pool, int(blocks_per_canvas))
            if len(picked) < int(blocks_per_canvas):
                radius = int(max(1, (range_hi_t - range_lo_t)))
                expanded_lo = int(max(t_lo, center_t - radius))
                expanded_hi = int(min(t_hi, center_t + radius + 1))
                fallback_pool = [
                    rec for rec in candidates
                    if int(rec[0]) >= int(expanded_lo) and int(rec[0]) < int(expanded_hi)
                ]
                picked.extend(_take_from_pool(fallback_pool, int(blocks_per_canvas - len(picked))))
            if len(picked) < int(blocks_per_canvas):
                global_pool = [rec for rec in candidates if int(rec[0]) >= int(t_lo) and int(rec[0]) < int(t_hi)]
                picked.extend(_take_from_pool(global_pool, int(blocks_per_canvas - len(picked))))

            chosen_all.extend(picked)
            focus_ranges.append({
                "image_idx": int(pi),
                "range_t_start": int(range_lo_t),
                "range_t_end": int(max(range_lo_t + 1, range_hi_t)),
                "center_t": int(center_t),
                "frame_mass_sum": float(sum(float(frame_mass.get(int(t1), 0.0)) for t1 in frames_sorted if int(t1) >= int(range_lo_t) and int(t1) < int(range_hi_t))),
                "picked_blocks": int(len(picked)),
            })

        return chosen_all, focus_ranges

    def reorder_chosen_for_place(
        chosen_blocks: List[Tuple[int, int, int, float]],
        order_mode: str,
    ) -> List[Tuple[int, int, int, float]]:
        """Order already-selected blocks for placement inside P canvases."""
        if not chosen_blocks:
            return []
        mode = str(order_mode).lower().strip()
        if mode == "time":
            # Place by source frame_id first, then by block top-left in raster order.
            return sorted(
                chosen_blocks,
                key=lambda x: (int(x[0]), int(x[1]), int(x[2]), -float(x[3])),
            )
        # "score": keep strongest blocks first.
        return sorted(
            chosen_blocks,
            key=lambda x: (-float(x[3]), int(x[0]), int(x[1]), int(x[2])),
        )
    for bi, t_anchor in enumerate(anchor_t_indices):
        t_start = int(t_anchor)
        t_end = int(anchor_t_indices[bi + 1]) if (bi + 1) < len(anchor_t_indices) else int(len(frame_ids))
        bucket_ranges_t.append({
            "bucket_id": int(bi),
            "anchor_t": int(t_anchor),
            "p_t_start_exclusive": int(t_start),
            "p_t_end_exclusive": int(t_end),
        })

        # Build P candidates strictly inside this bucket and before next anchor.
        # Order: time asc, score desc (within each frame).
        candidates: List[Tuple[int, int, int, float]] = []  # (t, bh, bw, score)
        frame_pick_count: Dict[int, int] = {}
        bwb = int(wb // 2)
        for t in range(int(t_start) + 1, int(t_end)):
            if int(t) in anchor_set:
                continue
            if (t < len(good_mask)) and (not bool(good_mask[t])):
                continue
            ps = score_map_to_patch_scores(fused_maps[int(t)], patch=p)  # (hb,wb)
            bs = patch_scores_to_block_scores(ps)  # (hb//2, wb//2)
            flat = bs.reshape(-1).astype(np.float32)
            if flat.size <= 0:
                continue
            order = np.argsort(-flat, kind="stable")
            for fid in order.tolist():
                bh = int(int(fid) // bwb)
                bw = int(int(fid) % bwb)
                sc = float(flat[int(fid)])
                used_n = int(frame_pick_count.get(int(t), 0))
                penalty = 1.0 / math.sqrt(1.0 + float(block_selection_frame_penalty) * float(used_n))
                candidates.append((int(t), int(bh), int(bw), float(sc * penalty)))
                frame_pick_count[int(t)] = used_n + 1

        bucket_plans.append({
            "bucket_id": int(bi),
            "anchor_t": int(t_anchor),
            "t_start": int(t_start),
            "t_end": int(t_end),
            "bucket_score": float(sum(float(sc0) for (_, _, _, sc0) in candidates)),
            "candidates": [(int(t0), int(bh0), int(bw0), float(sc0)) for (t0, bh0, bw0, sc0) in candidates],
        })

    bucket_extra_images: Dict[int, int] = {
        int(bp["bucket_id"]): int(p_full_per_bucket)
        for bp in bucket_plans
    }
    bucket_image_counts: Dict[int, int] = {
        int(bp["bucket_id"]): int(images_per_bucket)
        for bp in bucket_plans
    }
    total_block_budget = int(num_buckets * int(p_full_per_bucket) * int(S_full // 4))

    selected_t_needed: set = set(int(x) for x in anchor_t_indices)
    for bucket_plan in bucket_plans:
        bi = int(bucket_plan["bucket_id"])
        extra_images = int(bucket_extra_images.get(int(bi), 0))
        chosen_for_place, focus_ranges = select_bucket_blocks_stratified(
            candidates=[(int(t0), int(bh0), int(bw0), float(sc0)) for (t0, bh0, bw0, sc0) in bucket_plan.get("candidates", [])],
            t_start=int(bucket_plan.get("t_start", int(bucket_plan["anchor_t"]))),
            t_end=int(bucket_plan.get("t_end", int(bucket_plan["anchor_t"]) + 1)),
            extra_images=int(extra_images),
            blocks_per_canvas=int(blocks_per_full_canvas),
        )
        chosen_for_place = reorder_chosen_for_place(chosen_for_place, collage_patch_order)
        bucket_plan["chosen_for_place"] = [(int(t0), int(bh0), int(bw0), float(sc0)) for (t0, bh0, bw0, sc0) in chosen_for_place]
        bucket_plan["sparse_focus_ranges"] = focus_ranges
        if extra_images <= 0:
            continue
        for (t_sel, bh_sel, bw_sel, _) in chosen_for_place:
            selected_t_needed.add(int(t_sel))
            selected_blocks_by_frame.setdefault(int(t_sel), []).append((int(bh_sel), int(bw_sel)))

    for t_sel, blks in selected_blocks_by_frame.items():
        selected_blocks_count_by_frame[int(t_sel)] = int(len(blks))

    frames_bgr_pad_by_t: Dict[int, np.ndarray] = {}
    if score_source == "bitcost":
        selected_t_sorted = sorted(int(x) for x in selected_t_needed)
        selected_frame_ids = [int(frame_ids[int(t)]) for t in selected_t_sorted]
        selected_frames_bgr = decode_frames_bgr(
            vp,
            selected_frame_ids,
            backsearch_max=int(decode_backsearch_max),
            backend=decode_backend,
        )
        good_mask = [False] * int(len(frame_ids))
        for t_sel, fr in zip(selected_t_sorted, selected_frames_bgr):
            bad = frame_is_bad(
                fr,
                skip_black_frames=bool(skip_black_frames),
                skip_corrupt_frames=bool(skip_corrupt_frames),
                black_y_mean_thr=float(black_y_mean_thr),
                black_y_std_thr=float(black_y_std_thr),
                solid_y_std_thr=4.0,
                solid_color_std_thr=4.0,
                solid_max_range_thr=12.0,
                corrupt_green_frac_thr=float(corrupt_green_frac_thr),
                corrupt_g_thr=int(corrupt_g_thr),
                corrupt_rb_thr=int(corrupt_rb_thr),
            )
            good_mask[int(t_sel)] = not bool(bad)
            fr_rs = _resize_bgr(fr, int(resize_h), int(resize_w)) if do_resize else fr
            fr_pad, _ = pad_to_multiple_of_bgr(fr_rs, pad_base)
            frames_bgr_pad_by_t[int(t_sel)] = fr_pad
        if frames_bgr_pad_by_t:
            first_t = min(frames_bgr_pad_by_t.keys())
            top_lb, bottom_lb, left_lb, right_lb = detect_letterbox_bbox_bgr(
                frames_bgr_pad_by_t[int(first_t)], dark_thr=float(letterbox_dark_thr)
            )
    else:
        frames_bgr_pad_by_t = {int(t): fr for t, fr in enumerate(frames_bgr_pad)}

    if not frames_bgr_pad_by_t:
        return "skip", f"{key} no frames available after selection", None

    num_images_actual = int(sum(bucket_image_counts.values())) if bucket_image_counts else int(num_buckets)
    max_total = int(num_images_actual * int(S_full))

    def choose_anchor_t(bucket_plan: Dict[str, Any]) -> int:
        t_anchor0 = int(bucket_plan["anchor_t"])
        if bool(good_mask[t_anchor0]):
            return int(t_anchor0)
        for (t_sel, _, _, _) in bucket_plan["chosen_for_place"]:
            if int(t_sel) < len(good_mask) and bool(good_mask[int(t_sel)]) and int(t_sel) in frames_bgr_pad_by_t:
                return int(t_sel)
        return int(t_anchor0)

    for bucket_plan in bucket_plans:
        bi = int(bucket_plan["bucket_id"])
        t_anchor_use = choose_anchor_t(bucket_plan)
        fr_anchor = frames_bgr_pad_by_t.get(int(t_anchor_use))
        if fr_anchor is not None:
            append_frame_patches(src_img_idx=int(t_anchor_use), frame_rgb=fr_anchor[:, :, ::-1], blocks=all_blocks)
        else:
            for _ in all_blocks:
                append_pad_block()
        canvas_bucket_ids.append(int(bi))
        canvas_roles.append("I_full")

        cpos = 0
        chosen_for_place = bucket_plan["chosen_for_place"]
        extra_images = int(bucket_extra_images.get(int(bi), 0))
        for pi in range(int(extra_images)):
            for _ in range(blocks_per_full_canvas):
                if cpos < len(chosen_for_place):
                    t_sel, bh_sel, bw_sel, _ = chosen_for_place[cpos]
                    fr = frames_bgr_pad_by_t.get(int(t_sel))
                    if (fr is not None) and (int(t_sel) < len(good_mask)) and bool(good_mask[int(t_sel)]):
                        append_frame_patches(src_img_idx=int(t_sel), frame_rgb=fr[:, :, ::-1], blocks=[(int(bh_sel), int(bw_sel))])
                    else:
                        append_pad_block()
                    cpos += 1
                else:
                    append_pad_block()
            canvas_bucket_ids.append(int(bi))
            canvas_roles.append(f"P_full_{int(pi) + 1}")

    # 7.5) pad/truncate to exactly max_total patches so that we can fill whole canvases
    valid_n = int(len(patches_list))
    if valid_n > int(max_total):
        patches_list = patches_list[: int(max_total)]
        src_pos_list = src_pos_list[: int(max_total)]
        src_frameid_list = src_frameid_list[: int(max_total)]
        src_ts_list = src_ts_list[: int(max_total)]
        valid_n = int(max_total)

    while len(patches_list) < int(max_total):
        patches_list.append(pz)
        src_pos_list.append([-1, -1, -1])
        src_frameid_list.append(-1)
        src_ts_list.append(-1.0)

    patches = np.stack(patches_list, axis=0).astype(np.uint8) if patches_list else np.zeros((0, p, p, 3), dtype=np.uint8)
    src_patch_position = np.asarray(src_pos_list, dtype=np.int32) if src_pos_list else np.zeros((0, 3), dtype=np.int32)
    src_frame_ids = np.asarray(src_frameid_list, dtype=np.int32) if src_frameid_list else np.zeros((0,), dtype=np.int32)
    src_timestamps = np.asarray(src_ts_list, dtype=np.float32) if src_ts_list else np.zeros((0,), dtype=np.float32)

    # Pack into full canvases
    images_rgb, patch_position, img_ptr_arr = pack_patches_to_canvases(patches, hb=hb, wb=wb, patch=p)
    frame_ids_arr = np.asarray(frame_ids, dtype=np.int32)

    # Save JPG canvases (do NOT save images.npy / patches.npy)
    jpg_files = save_canvases_as_jpg(images_rgb, out_dir=out_dir, quality=95)

    # Save required arrays/metadata
    np.save(str(Path(out_dir) / "patch_position.npy"), patch_position, allow_pickle=False)
    np.save(str(Path(out_dir) / "img_ptr.npy"), img_ptr_arr, allow_pickle=False)
    np.save(str(Path(out_dir) / "frame_ids.npy"), frame_ids_arr, allow_pickle=False)
    np.save(str(Path(out_dir) / "src_patch_position.npy"), src_patch_position, allow_pickle=False)  # [frame_id, patch_h, patch_w]
    # Compatibility alias: many downstream jsonl schemas expect `patch_positions.npy`
    np.save(str(Path(out_dir) / "patch_positions.npy"), src_patch_position, allow_pickle=False)
    np.save(str(Path(out_dir) / "src_frame_ids.npy"), src_frame_ids, allow_pickle=False)
    np.save(str(Path(out_dir) / "src_timestamps.npy"), src_timestamps, allow_pickle=False)
    # np.save(str(Path(out_dir) / "position.npy"), src_patch_position, allow_pickle=False) 

    meta = {
        "video": vp,
        "key": key,
        "out_dir": str(Path(out_root) / key),
        "mirror_src_root": None,
        "total_frames": int(total_frames),
        "fps": float(fps),
        "orig_hw": [int(H0), int(W0)],
        "pad_bottom_right": [int(pad_bottom), int(pad_right)],
        "padded_hw": [int(H1), int(W1)],
        "patch": int(p),
        "pad_base": int(pad_base),
        "hb_wb": [int(hb), int(wb)],
        "seq_len": int(seq_len),
        "auto_num_gops": bool(auto_num_gops),
        "analysis_candidate_multiplier": float(analysis_candidate_multiplier),
        "analysis_max_frames": int(analysis_max_frames),
        "windows": [{"start": int(a), "end": int(b), "pkt_sum": int(c)} for (a, b, c) in windows],
        "window_debug": win_dbg,
        "w_mv": float(w_mv),
        "w_res": float(w_res),
        "mv_compensate": str(mv_compensate),
        "mv_dir_mode": str(mv_dir_mode),
        "w_mv_l0": float(w_mv_l0),
        "w_mv_l1": float(w_mv_l1),
        "mv_source": str(mv_source),
        "use_cv_reader": bool(use_cv),
        "score_source": str(score_source),
        "bitcost_grid": str(bitcost_grid),
        "bitcost_pct": float(bitcost_pct),
        "bitcost_log_scale": bool(bitcost_log_scale),
        "decode_backend": str(decode_backend),
        "residual_fallback": {
            "used": bool(residual_fallback_frames > 0),
            "frames": int(residual_fallback_frames),
            "ratio": float(residual_fallback_frames) / float(max(1, len(frames_bgr_pad))),
            "proxy": "sobel_luma_magnitude_centered_128",
            "cv_reader_fallback_cause": str(cv_reader_fallback_cause),
            "reasons": {k: int(v) for k, v in residual_fallback_by_reason.items()},
        },
        "max_total_patches": int(max_total),
        "I_frame_patches": int(S_full),
        "num_gops_req": int(num_gops_req),
        "num_gops_eff": int(len(anchor_t_indices)),
        "gop_anchor_t_indices": [int(x) for x in anchor_t_indices],
        "gop_anchor_frame_ids": [int(frame_ids[int(x)]) for x in anchor_t_indices],
        "gop_anchor_mode": str(gop_anchor_mode),
        "images_per_bucket_max": int(images_per_bucket),
        "bucket_image_counts": {str(k): int(v) for k, v in bucket_image_counts.items()},
        "bucket_extra_images": {str(k): int(v) for k, v in bucket_extra_images.items()},
        "num_images_budget": int(num_images_budget),
        "p_full_per_bucket": int(p_full_per_bucket),
        "bucket_ranges_t": bucket_ranges_t,
        "bucket_sparse_focus_ranges": [
            {
                "bucket_id": int(bp.get("bucket_id", -1)),
                "focus_ranges": [
                    {
                        "image_idx": int(fr.get("image_idx", -1)),
                        "range_t_start": int(fr.get("range_t_start", -1)),
                        "range_t_end": int(fr.get("range_t_end", -1)),
                        "center_t": int(fr.get("center_t", -1)),
                        "frame_mass_sum": float(fr.get("frame_mass_sum", 0.0)),
                        "picked_blocks": int(fr.get("picked_blocks", 0)),
                    }
                    for fr in bp.get("sparse_focus_ranges", [])
                ],
            }
            for bp in bucket_plans
        ],
        "canvas_bucket_ids": canvas_bucket_ids,
        "canvas_roles": canvas_roles,
        "keyframe_align": keyframe_align_dbg,
        "total_patches_target": int(max_total),
        "total_patches_valid": int(valid_n),
        "num_images": int(images_rgb.shape[0]),
        "jpg_files": jpg_files,
        "patches_per_image": int(S_full),
        "auto_budget": auto_budget_dbg,
        "global_block_selection": {
            "total_block_budget": int(total_block_budget),
            "max_blocks_per_frame": int(max_blocks),
            "frame_penalty": float(block_selection_frame_penalty),
            "selected_blocks_total": int(sum(selected_blocks_count_by_frame.values()) if selected_blocks_count_by_frame else 0),
            "selected_blocks_per_frame": selected_blocks_count_by_frame,
        },
        "fill_to_images": bool(fill_to_images),
        "group_by_time": bool(group_by_time),
        "keep_blocks_within_canvas": bool(keep_blocks_within_canvas),
        "ensure_per_second": bool(ensure_per_second),
        "sec_stride": float(sec_stride),
        "min_blocks_per_second": int(min_blocks_per_second),
        "mask_letterbox": bool(mask_letterbox),
        "letterbox_dark_thr": float(letterbox_dark_thr),
        "letterbox_bbox_tblr": [int(top_lb), int(bottom_lb), int(left_lb), int(right_lb)],
        # src_patch_position now stores [frame_id, patch_h, patch_w] (pad tokens are -1)
        "src_patch_position_format": "[frame_id, patch_h, patch_w] (pad tokens are -1)",
        "position_npy": "position.npy",
        "sample_id": str(sample_id) if sample_id else "",
        "caption": str(caption) if caption else "",
        "smart_resize": {
            "min_pixels": int(min_pixels),
            "max_pixels": int(max_pixels),
            "factor": int(2 * p),
            "do_resize": bool(do_resize),
            "resize_hw": [int(resize_h), int(resize_w)],
            "scale_hw": [float(scale_h), float(scale_w)],
        },
        "decoded_selected_frames": int(len(frames_bgr_pad_by_t)),
        "has_src_timestamps": True,
        "src_timestamps_npy": "src_timestamps.npy",
    }
    meta["mirror_src_root"] = os.environ.get("LLAVA_CODEC_MIRROR_SRC_ROOT")
    with open(str(Path(out_dir) / "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # A compact per-sample record for fast join in step-2 (avoid per-sample filesystem probing).
    index_rec: Dict[str, Any] = {
        "video": vp,
        "key": str(key),
        "asset_dir": str(out_dir),
        "jpg": list(jpg_files),
        "patch_position": "patch_position.npy",
        "img_ptr": "img_ptr.npy",
        "src_patch_position": "src_patch_position.npy",
        "patch_positions": "patch_positions.npy",
        "src_frame_ids": "src_frame_ids.npy",
        "frame_ids": "frame_ids.npy",
        "position": "position.npy",
        "sample_id": str(sample_id) if sample_id else "",
        "caption": str(caption) if caption else "",
        # prefer meta-derived count; this equals the packed total (including pad patches)
        "num_patches": int(meta.get("max_total_patches", int(patches.shape[0]))),
        "num_images": int(meta.get("num_images", int(images_rgb.shape[0]))),
        "fps": float(meta.get("fps", fps)),
        "patch": int(meta.get("patch", p)),
        "padded_hw": meta.get("padded_hw"),
        "src_timestamps": "src_timestamps.npy",
        "orig_messages": orig_messages if isinstance(orig_messages, list) else [],
    }

    Path(done_mark).write_text("ok\n", encoding="utf-8")
    return "ok", f"{key} ok patches={int(patches.shape[0])} images={int(images_rgb.shape[0])} jpg={len(jpg_files)} use_cv_reader={int(use_cv)}", index_rec
