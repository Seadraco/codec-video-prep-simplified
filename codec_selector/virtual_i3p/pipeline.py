#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main pipeline for Virtual I+3P long video processing."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from codec_selector.codec_patch_gop.frame_utils import (
    _resize_bgr,
    decode_frames_bgr,
    detect_letterbox_bbox_bgr,
    frame_is_bad,
    pad_to_multiple_of_bgr,
)
from codec_selector.codec_patch_gop.patch_utils import save_canvases_as_jpg
from codec_selector.codec_patch_gop.utils import ensure_dir, smart_resize
from codec_selector.codec_patch_gop.video_probe import (
    ffprobe_video_codec_name,
    get_total_frames_fps,
)
from codec_selector.codec_patch_gop.video_processor import cv_reader_fetch_bitcost

from .anchor import select_anchor_frame
from .collage import pack_anchor_fullframe, pack_p_collage
from .p_window import process_p_window_bitcost


def _clamp_int(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def process_video_virtual_i3p(
    video_path: str,
    out_dir: str,
    total_images: int = 384,
    num_segments: int = 96,
    patch_size: int = 14,
    canvas_size: int = 576,
    p_images_per_segment: int = 3,
    anchor_mode: str = "midpoint",
    bitcost_path: Optional[str] = None,
    max_patches_per_p_image: Optional[int] = None,
    min_patches_per_p_image: Optional[int] = None,
    group_size: int = 2,
    decay: float = 0.9,
    use_temporal_accumulation: bool = True,
    use_group_complete: bool = True,
    use_temporal_balance: bool = True,
    temporal_balance_ratio: float = 0.5,
    num_buckets_per_p_window: int = 4,
    overwrite: bool = False,
    save_debug: bool = False,
    bitcost_grid: str = "sub",
    bitcost_pct: float = 99.0,
    bitcost_log_scale: bool = True,
    min_pixels: int = 56 * 56,
    max_pixels: int = 768 * 768,
    skip_black_frames: bool = True,
    skip_corrupt_frames: bool = True,
    decode_backend: str = "ffmpeg_native",
    parallel_segments: int = 0,
    threads_per_segment: int = 4,
    segment_guard_frames: int = 30,
    mask_letterbox: bool = True,
    letterbox_dark_thr: float = 16.0,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Process one video with Virtual I+3P strategy.

    Returns:
        (status, message, metadata_dict)
        status in {"ok", "skip", "fail"}
    """
    # Avoid OpenCV oversubscription
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "loglevel;error")

    vp = str(video_path)
    if not vp or not Path(vp).exists():
        return "skip", "missing video", None

    # Validate codec
    codec_name = ffprobe_video_codec_name(vp)
    if codec_name not in {"h264", "hevc"}:
        return "skip", f"unsupported codec={codec_name}", None

    out_dir_p = Path(out_dir)
    done_mark = out_dir_p / "_DONE"
    if done_mark.exists() and not overwrite:
        return "skip", "already done", None

    ensure_dir(str(out_dir_p / "images"))
    if save_debug:
        ensure_dir(str(out_dir_p / "debug"))

    # Video metadata
    total_frames, fps, H0, W0 = get_total_frames_fps(vp)
    if total_frames <= 0:
        return "fail", "cannot read total_frames", None

    fps_use = float(fps) if fps > 0 else 30.0

    # Compute segment boundaries
    p_images_per_segment = int(max(1, p_images_per_segment))
    expected_num_segments = total_images // (1 + p_images_per_segment)
    num_segments = int(min(expected_num_segments, num_segments))
    num_segments = max(1, num_segments)

    # Determine resize dimensions
    p = int(patch_size)
    pad_base = 2 * p

    try:
        resize_h, resize_w = smart_resize(
            height=int(H0),
            width=int(W0),
            factor=int(pad_base),
            min_pixels=int(min_pixels),
            max_pixels=int(max_pixels),
        )
    except Exception:
        resize_h, resize_w = int(H0), int(W0)

    pad_bottom = (pad_base - (resize_h % pad_base)) % pad_base
    pad_right = (pad_base - (resize_w % pad_base)) % pad_base
    H1 = resize_h + pad_bottom
    W1 = resize_w + pad_right
    hb, wb = H1 // p, W1 // p

    # Ensure even grid
    if hb % 2 != 0:
        hb += 1
        H1 = hb * p
        pad_bottom = H1 - resize_h
    if wb % 2 != 0:
        wb += 1
        W1 = wb * p
        pad_right = W1 - resize_w

    S_full = hb * wb

    # P-image budget
    if max_patches_per_p_image is None:
        max_patches_per_p_image = S_full
    else:
        max_patches_per_p_image = int(max_patches_per_p_image)

    # Segment frame IDs
    segment_boundaries = []
    for i in range(num_segments + 1):
        fid = _clamp_int(round(i * total_frames / num_segments), 0, total_frames - 1)
        segment_boundaries.append(fid)

    segments: List[Dict[str, Any]] = []
    for i in range(num_segments):
        start_fid = segment_boundaries[i]
        end_fid = segment_boundaries[i + 1]
        # Collect all frame IDs in this segment
        seg_frame_ids = list(range(start_fid, end_fid))
        if not seg_frame_ids:
            seg_frame_ids = [start_fid]
        segments.append({
            "idx": i,
            "start_fid": start_fid,
            "end_fid": end_fid,
            "frame_ids": seg_frame_ids,
        })

    # Fetch all bitcost data at once (more efficient)
    all_frame_ids: List[int] = []
    for seg in segments:
        # For P windows, we need frames within each segment
        # Divide segment into p_images_per_segment sub-windows
        seg_fids = seg["frame_ids"]
        n = len(seg_fids)
        if n <= p_images_per_segment:
            # Use all frames
            all_frame_ids.extend(seg_fids)
        else:
            # Divide into p_images_per_segment windows
            window_size = max(1, n // p_images_per_segment)
            for w in range(p_images_per_segment):
                w_start = w * window_size
                w_end = (w + 1) * window_size if w < p_images_per_segment - 1 else n
                window_fids = seg_fids[w_start:w_end]
                all_frame_ids.extend(window_fids)

    # Deduplicate and sort
    all_frame_ids = sorted(set(int(x) for x in all_frame_ids))

    # Fetch bitcost
    bitcost_items: List[Dict[str, Any]] = []
    try:
        bitcost_items = cv_reader_fetch_bitcost(
            vp,
            [int(x) for x in all_frame_ids],
            bitcost_grid=str(bitcost_grid),
            parallel_segments=int(parallel_segments),
            threads_per_segment=int(threads_per_segment),
            segment_guard_frames=int(segment_guard_frames),
        )
    except Exception as e:
        return "fail", f"bitcost fetch failed: {e}", None

    # Map frame_id -> bitcost_item
    bitcost_by_fid: Dict[int, Dict[str, Any]] = {}
    for item in bitcost_items:
        fid = int(item.get("frame_idx", -1))
        if fid >= 0:
            bitcost_by_fid[fid] = item

    # Build output images
    images_list: List[np.ndarray] = []
    image_metadata: List[Dict[str, Any]] = []
    debug_summary: List[Dict[str, Any]] = []
    all_selected_patches: List[Dict[str, Any]] = []

    # For debug video: collect per-segment frame -> patches mapping
    segment_debug_data: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}

    image_idx = 0
    b = int(max(1, int(group_size)))

    for seg in segments:
        seg_idx = seg["idx"]
        seg_fids = seg["frame_ids"]

        # ---- Anchor ----
        anchor_fid, anchor_frame_bgr = select_anchor_frame(
            vp,
            seg_fids,
            mode=str(anchor_mode),
            backend=str(decode_backend),
        )

        if anchor_frame_bgr is not None and not frame_is_bad(anchor_frame_bgr):
            # Resize and pad
            if resize_h != H0 or resize_w != W0:
                anchor_rs = _resize_bgr(anchor_frame_bgr, resize_h, resize_w)
            else:
                anchor_rs = anchor_frame_bgr
            anchor_pad, _ = pad_to_multiple_of_bgr(anchor_rs, pad_base)
            anchor_rgb = anchor_pad[:, :, ::-1]

            # Pack full-frame anchor
            anchor_img, _, _ = pack_anchor_fullframe(anchor_rgb, hb, wb, p, group_size=b)
            images_list.append(anchor_img[0])
            image_metadata.append({
                "image_idx": image_idx,
                "type": "anchor",
                "segment_idx": seg_idx,
                "frame_id": int(anchor_fid),
            })

            # Anchor saved in debug video later
        else:
            # Blank canvas as fallback
            images_list.append(np.zeros((H1, W1, 3), dtype=np.uint8))
            image_metadata.append({
                "image_idx": image_idx,
                "type": "anchor",
                "segment_idx": seg_idx,
                "frame_id": int(anchor_fid) if anchor_fid >= 0 else seg_fids[len(seg_fids) // 2],
            })

        image_idx += 1

        # ---- P Windows ----
        n = len(seg_fids)
        if n <= p_images_per_segment:
            p_windows = [seg_fids]
        else:
            window_size = max(1, n // p_images_per_segment)
            p_windows = []
            for w in range(p_images_per_segment):
                w_start = w * window_size
                w_end = (w + 1) * window_size if w < p_images_per_segment - 1 else n
                p_windows.append(seg_fids[w_start:w_end])

        for p_win_idx, p_win_fids in enumerate(p_windows):
            # Get bitcost items for this window
            win_bitcost = []
            for fid in p_win_fids:
                item = bitcost_by_fid.get(int(fid))
                if item is not None:
                    win_bitcost.append(item)

            if not win_bitcost or not p_win_fids:
                # Blank P-collage
                images_list.append(np.zeros((H1, W1, 3), dtype=np.uint8))
                image_metadata.append({
                    "image_idx": image_idx,
                    "type": "p_collage",
                    "segment_idx": seg_idx,
                    "p_window_idx": p_win_idx,
                    "selected_patches": [],
                })
                image_idx += 1
                continue

            # Process P-window
            p_result = process_p_window_bitcost(
                video_path=vp,
                window_frame_ids=[int(x) for x in p_win_fids],
                out_h=int(H1),
                out_w=int(W1),
                patch_size=int(p),
                group_size=int(b),
                max_patches_per_p_image=int(max_patches_per_p_image),
                bitcost_grid=str(bitcost_grid),
                bitcost_pct=float(bitcost_pct),
                bitcost_log_scale=bool(bitcost_log_scale),
                codec_name=str(codec_name),
                use_temporal_accumulation=bool(use_temporal_accumulation),
                decay=float(decay),
                use_temporal_balance=bool(use_temporal_balance),
                temporal_balance_ratio=float(temporal_balance_ratio),
                num_buckets_per_p_window=int(num_buckets_per_p_window),
                bitcost_items=win_bitcost if win_bitcost else None,
                decode_backend=str(decode_backend),
                min_patches_per_p_image=int(min_patches_per_p_image) if min_patches_per_p_image else None,
            )

            selected_patches = p_result["selected_patches"]

            # Decode selected frames for this P-window
            selected_fids = sorted(set(int(p["frame_id"]) for p in selected_patches))
            frame_dict: Dict[int, np.ndarray] = {}
            if selected_fids:
                frames_bgr = decode_frames_bgr(
                    vp,
                    selected_fids,
                    backsearch_max=32,
                    backend=str(decode_backend),
                )
                for fid, fr in zip(selected_fids, frames_bgr):
                    if fr is not None and not frame_is_bad(fr):
                        if resize_h != H0 or resize_w != W0:
                            fr_rs = _resize_bgr(fr, resize_h, resize_w)
                        else:
                            fr_rs = fr
                        fr_pad, _ = pad_to_multiple_of_bgr(fr_rs, pad_base)
                        frame_dict[int(fid)] = fr_pad[:, :, ::-1]  # BGR -> RGB

            # Pack collage
            if selected_patches and frame_dict:
                p_img, _, _ = pack_p_collage(
                    frame_dict,
                    selected_patches,
                    hb,
                    wb,
                    p,
                    group_size=b,
                )
                if p_img.shape[0] > 0:
                    images_list.append(p_img[0])
                else:
                    images_list.append(np.zeros((H1, W1, 3), dtype=np.uint8))
            else:
                images_list.append(np.zeros((H1, W1, 3), dtype=np.uint8))

            image_metadata.append({
                "image_idx": image_idx,
                "type": "p_collage",
                "segment_idx": seg_idx,
                "p_window_idx": p_win_idx,
                "selected_patches": selected_patches,
            })

            # Collect debug data per segment
            if save_debug and selected_patches:
                if seg_idx not in segment_debug_data:
                    segment_debug_data[seg_idx] = {}
                for patch in selected_patches:
                    fid = int(patch["frame_id"])
                    if fid not in segment_debug_data[seg_idx]:
                        segment_debug_data[seg_idx][fid] = []
                    segment_debug_data[seg_idx][fid].append(patch)

            # Debug summary
            if selected_patches:
                fids = [p["frame_id"] for p in selected_patches]
                scores = [p.get("score", 0.0) for p in selected_patches]
                debug_summary.append({
                    "segment_idx": seg_idx,
                    "p_window_idx": p_win_idx,
                    "num_selected_groups": p_result.get("num_selected_groups", 0),
                    "num_selected_patches": len(selected_patches),
                    "min_frame_id": min(fids),
                    "max_frame_id": max(fids),
                    "mean_score": float(np.mean(scores)) if scores else 0.0,
                    "max_score": float(np.max(scores)) if scores else 0.0,
                })

            image_idx += 1

        # ---- Generate debug video for this segment ----
        if save_debug and seg_idx in segment_debug_data and segment_debug_data[seg_idx]:
            seg_fids_to_decode = sorted(segment_debug_data[seg_idx].keys())
            if seg_fids_to_decode:
                try:
                    seg_frames_bgr = decode_frames_bgr(
                        vp,
                        seg_fids_to_decode,
                        backsearch_max=32,
                        backend=str(decode_backend),
                    )
                    seg_frames_dict: Dict[int, np.ndarray] = {}
                    for fid, fr in zip(seg_fids_to_decode, seg_frames_bgr):
                        if fr is not None and not frame_is_bad(fr):
                            seg_frames_dict[int(fid)] = fr

                    if seg_frames_dict:
                        from .debug import generate_segment_debug_video
                        generate_segment_debug_video(
                            frames_bgr_dict=seg_frames_dict,
                            frame_to_patches=segment_debug_data[seg_idx],
                            patch_size=int(p),
                            segment_idx=int(seg_idx),
                            debug_dir=out_dir_p / "debug",
                            fps=float(fps_use),
                        )
                except Exception as e:
                    print(f"[warn] debug video generation failed for segment {seg_idx}: {e}")

    # Save images
    images_arr = np.stack(images_list, axis=0).astype(np.uint8) if images_list else np.zeros((0, H1, W1, 3), dtype=np.uint8)
    jpg_files = save_canvases_as_jpg(images_arr, out_dir=str(out_dir_p / "images"), quality=95)

    # Rename to image_000000.jpg format
    for i, old_fn in enumerate(jpg_files):
        old_fp = out_dir_p / "images" / old_fn
        new_fn = f"image_{i:06d}.jpg"
        new_fp = out_dir_p / "images" / new_fn
        old_fp.rename(new_fp)
        jpg_files[i] = new_fn

    # Save metadata
    meta = {
        "video_path": str(vp),
        "total_images": int(total_images),
        "num_segments": int(num_segments),
        "p_images_per_segment": int(p_images_per_segment),
        "anchor_mode": str(anchor_mode),
        "patch_size": int(p),
        "canvas_size": int(canvas_size),
        "group_size": int(b),
        "use_temporal_accumulation": bool(use_temporal_accumulation),
        "decay": float(decay),
        "use_group_complete": bool(use_group_complete),
        "use_temporal_balance": bool(use_temporal_balance),
        "temporal_balance_ratio": float(temporal_balance_ratio),
        "num_buckets_per_p_window": int(num_buckets_per_p_window),
        "bitcost_only": True,
        "bitcost_grid": str(bitcost_grid),
        "bitcost_pct": float(bitcost_pct),
        "bitcost_log_scale": bool(bitcost_log_scale),
        "total_frames": int(total_frames),
        "fps": float(fps_use),
        "orig_hw": [int(H0), int(W0)],
        "padded_hw": [int(H1), int(W1)],
        "hb_wb": [int(hb), int(wb)],
        "num_images_output": int(len(images_list)),
        "jpg_files": jpg_files,
        "images": image_metadata,
    }

    with open(str(out_dir_p / "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Save numpy arrays if needed by downstream
    patch_positions = []
    for img_meta in image_metadata:
        if img_meta["type"] == "anchor":
            patch_positions.append([img_meta["image_idx"], -1, -1])
        else:
            for p in img_meta.get("selected_patches", []):
                patch_positions.append([
                    img_meta["image_idx"],
                    int(p["patch_y"]),
                    int(p["patch_x"]),
                ])

    if patch_positions:
        np.save(
            str(out_dir_p / "patch_positions.npy"),
            np.asarray(patch_positions, dtype=np.int32),
            allow_pickle=False,
        )

    # Frame IDs array
    frame_ids_arr = np.asarray(all_frame_ids, dtype=np.int32)
    np.save(str(out_dir_p / "frame_ids.npy"), frame_ids_arr, allow_pickle=False)

    # Visible indices (all frames used)
    visible = np.ones(len(all_frame_ids), dtype=np.int32)
    np.save(str(out_dir_p / "visible_indices.npy"), visible, allow_pickle=False)

    # Debug summary
    if save_debug and debug_summary:
        from .debug import write_debug_summary
        write_debug_summary(debug_summary, out_dir_p / "debug")

    # Done mark
    done_mark.write_text("ok\n", encoding="utf-8")

    return (
        "ok",
        f"virtual_i3p ok images={len(images_list)} segments={num_segments}",
        meta,
    )
