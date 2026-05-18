#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main entry point for codec patch GOP processing."""

import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Pool

import numpy as np

from .utils import (
    ensure_dir,
    format_timestamp_ss,
    rewrite_user_content_image_tags,
    extract_video_path_from_item,
    infer_key_from_video,
    extract_caption_from_item,
    mirror_key_from_video,
    iter_jsonl,
)
from .video_processor import process_one_video


def _worker(job: Dict[str, Any]) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Worker function for multiprocessing pool."""
    key = job.get("key", "unknown")

    # Optional hard timeout per job
    tmo = int(job.get("per_video_timeout_sec", 0) or 0)
    if tmo > 0:
        try:
            import signal

            def _alarm_handler(signum, frame):
                raise TimeoutError(f"timeout>{tmo}s")

            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(tmo)
        except Exception:
            tmo = 0

    try:
        return process_one_video(**job)
    except Exception as e:
        return "fail", f"{key} err={repr(e)}", None
    finally:
        if tmo > 0:
            try:
                import signal
                signal.alarm(0)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(
        description="Generate codec-aware patch dataset with GOP-based sampling"
    )

    # Input/output
    ap.add_argument("--jsonl", required=True, help="jsonl lines containing at least {video, key, exists}")
    ap.add_argument("--out_root", required=True, help="Output root directory")
    ap.add_argument("--num_workers", type=int, default=8, help="Number of worker processes")

    # Filtering
    ap.add_argument("--max_lines", type=int, default=0, help="Only scan first N lines (0=all)")
    ap.add_argument("--max_jobs", type=int, default=0, help="Only process first N jobs (0=all)")
    ap.add_argument("--assume_exists", action="store_true", help="Skip Path.exists() check")
    ap.add_argument("--response_ok_only", action="store_true", help="Keep only error==null responses")

    # Sampling parameters
    ap.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    ap.add_argument("--num_gops", type=int, default=0, help="Number of GOP anchors (0=auto)")
    ap.add_argument("--p_full_per_bucket", type=int, default=3, help="Full P canvases per bucket")
    ap.add_argument("--no_auto_num_gops", action="store_true", help="Disable duration-adaptive GOP count")
    ap.add_argument("--analysis_candidate_multiplier", type=float, default=8.0, help="Analysis candidate frames ~= multiplier * num_images")
    ap.add_argument("--analysis_max_frames", type=int, default=1024, help="Hard cap on analysis candidate frames")
    ap.add_argument("--block_selection_frame_penalty", type=float, default=0.35, help="Per-frame score decay strength during block selection")
    ap.add_argument(
        "--gop_anchor_mode",
        type=str,
        default="sampled",
        choices=["sampled", "keyframe", "hybrid"],
        help="How to choose GOP anchors",
    )
    ap.add_argument("--window_len_frames", type=int, default=512, help="Window length in frames")
    ap.add_argument("--num_candidates", type=int, default=5, help="Number of candidate windows")
    ap.add_argument("--top_k_windows", type=int, default=2, help="Top K windows to select")

    # Patch parameters
    ap.add_argument("--patch", type=int, default=16, help="Patch size")
    ap.add_argument("--max_total_patches", type=int, default=0, help="Patch token budget (0=auto)")
    ap.add_argument("--max_total_patches_cap", type=int, default=30000, help="Hard cap for auto budget")

    # Scoring weights
    ap.add_argument("--mv_unit_div", type=float, default=4.0, help="MV unit divisor")
    ap.add_argument("--mv_pct", type=float, default=95.0, help="MV percentile")
    ap.add_argument("--res_pct", type=float, default=95.0, help="Residual percentile")
    ap.add_argument("--w_mv", type=float, default=1.0, help="MV weight")
    ap.add_argument("--w_res", type=float, default=1.0, help="Residual weight")
    ap.add_argument("--mv_compensate", type=str, default="none", choices=["none", "median"])
    ap.add_argument("--mv_dir_mode", type=str, default="l0", choices=["l0", "l1", "max", "sum", "biw"])
    ap.add_argument("--w_mv_l0", type=float, default=1.0)
    ap.add_argument("--w_mv_l1", type=float, default=1.0)
    ap.add_argument(
        "--mv_source",
        type=str,
        default="vector",
        choices=["vector", "energy", "energy_median", "auto"],
    )
    ap.add_argument(
        "--score_source",
        type=str,
        default="mvres",
        choices=["mvres", "bitcost"],
        help="Patch scoring source",
    )
    ap.add_argument(
        "--bitcost_grid",
        type=str,
        default="sub",
        choices=["sub", "mb", "ctu", "adaptive"],
        help="Bit-cost grid used when --score_source=bitcost (default: sub). "
             "'sub'=4x4 sub-macroblock (recommended, consistent for H.264/H.265), "
             "'mb'=16x16 macroblock, "
             "'ctu'=64x64 CTU (H.265 only), "
             "'adaptive'=auto select mb/ctu by codec (not recommended, inconsistent)",
    )
    ap.add_argument("--bitcost_pct", type=float, default=99.0, help="Percentile normalization for bit-cost score")
    ap.add_argument("--bitcost_log_scale", action="store_true", help="Apply log1p to bit-cost score before normalization")
    ap.add_argument(
        "--decode_backend",
        type=str,
        default="ffmpeg_native",
        choices=["ffmpeg_native", "auto"],
        help="Frame decode backend for final RGB patch extraction. 'auto' is kept as a compatibility alias for ffmpeg_native.",
    )

    # NEW: Collage patch order
    ap.add_argument(
        "--collage_patch_order",
        type=str,
        default="time",
        choices=["time", "score", "wh"],
        help="P-block placement order: 'time' sorts by frame_id then block position, 'score' keeps score-first order; 'wh' is kept as a compatibility alias for 'score'",
    )

    # Letterbox masking
    ap.add_argument("--no_mask_letterbox", action="store_true", help="Disable letterbox masking")
    ap.add_argument("--letterbox_dark_thr", type=float, default=16.0)

    # Frame filtering
    ap.add_argument("--no_skip_black_frames", action="store_true")
    ap.add_argument("--no_skip_corrupt_frames", action="store_true")
    ap.add_argument("--black_y_mean_thr", type=float, default=8.0)
    ap.add_argument("--black_y_std_thr", type=float, default=6.0)
    ap.add_argument("--corrupt_green_frac_thr", type=float, default=0.35)
    ap.add_argument("--corrupt_g_thr", type=int, default=180)
    ap.add_argument("--corrupt_rb_thr", type=int, default=90)

    # Resize
    ap.add_argument("--min_pixels", type=int, default=56 * 56)
    ap.add_argument("--max_pixels", type=int, default=768 * 768)

    # Mirror output structure
    ap.add_argument("--mirror_src_root", type=str, default="")
    ap.add_argument("--mirror_keep_ext", action="store_true")

    # Assets index
    ap.add_argument("--assets_index", type=str, default="")
    ap.add_argument("--assets_index_flush_every", type=int, default=200)
    ap.add_argument("--out_train_jsonl", type=str, default="")
    ap.add_argument("--empty_messages", action="store_true")

    # Misc
    ap.add_argument("--decode_backsearch_max", type=int, default=32)
    ap.add_argument("--per_video_timeout_sec", type=int, default=0)
    ap.add_argument("--group_by_time", action="store_true")
    ap.add_argument("--no_keep_blocks_within_canvas", action="store_true")
    ap.add_argument("--no_fill_to_images", action="store_true")
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--force_no_cv_reader", action="store_true")
    ap.add_argument("--scan_log_every", type=int, default=5000)

    # Time coverage
    ap.add_argument("--ensure_per_second", action="store_true")
    ap.add_argument("--sec_stride", type=float, default=1.0)
    ap.add_argument("--min_blocks_per_second", type=int, default=1)

    args = ap.parse_args()

    if args.mirror_src_root:
        os.environ["LLAVA_CODEC_MIRROR_SRC_ROOT"] = str(args.mirror_src_root)

    ensure_dir(args.out_root)

    print(f"[scan] jsonl={args.jsonl} max_lines={args.max_lines} max_jobs={args.max_jobs}")
    print(
        f"[config] collage_patch_order={args.collage_patch_order} "
        f"score_source={args.score_source} bitcost_grid={args.bitcost_grid} "
        f"decode_backend={args.decode_backend}"
    )

    # Job generator
    def job_iter():
        n_missing = 0
        n_skipped_schema = 0
        n_skipped_resp = 0
        yielded = 0

        for ln, it in enumerate(iter_jsonl(args.jsonl), start=1):
            if args.max_lines and ln > int(args.max_lines):
                break

            if args.response_ok_only:
                err = it.get("error")
                if err not in (None, "null"):
                    n_skipped_resp += 1
                    continue
                resp = it.get("response")
                if isinstance(resp, dict):
                    sc = resp.get("status_code")
                    if sc is not None and int(sc) != 200:
                        n_skipped_resp += 1
                        continue

            if it.get("exists") is False:
                n_missing += 1
                continue

            # Get video list
            video_list = []
            imgs_src = it.get("images_source")
            if isinstance(imgs_src, list) and imgs_src:
                for v in imgs_src:
                    if isinstance(v, str) and v:
                        video_list.append(str(v))
            else:
                v1 = extract_video_path_from_item(it)
                if isinstance(v1, str) and v1:
                    video_list.append(str(v1))

            if not video_list:
                n_skipped_schema += 1
                continue

            if not args.assume_exists:
                vv_ok = []
                for v in video_list:
                    if Path(v).exists():
                        vv_ok.append(v)
                    else:
                        n_missing += 1
                video_list = vv_ok
                if not video_list:
                    continue

            sample_id0 = it.get("id", "")
            if not isinstance(sample_id0, str):
                sample_id0 = ""
            caption = extract_caption_from_item(it)
            orig_messages = it.get("messages", [])
            if not isinstance(orig_messages, list):
                orig_messages = []

            for vid_i, video in enumerate(video_list):
                key = None
                if args.mirror_src_root:
                    key = mirror_key_from_video(
                        video_path=str(video),
                        mirror_src_root=str(args.mirror_src_root),
                        strip_ext=(not bool(args.mirror_keep_ext)),
                    )
                if not key:
                    key = infer_key_from_video(str(video), it)

                sample_id = sample_id0
                if sample_id and len(video_list) > 1:
                    sample_id = f"{sample_id}__v{vid_i}"

                yield {
                    "video_path": str(video),
                    "key": str(key),
                    "out_root": str(args.out_root),
                    "seq_len": int(args.seq_len),
                    "window_len_frames": int(args.window_len_frames),
                    "num_candidates": int(args.num_candidates),
                    "top_k_windows": int(args.top_k_windows),
                    "max_total_patches": int(args.max_total_patches),
                    "max_total_patches_cap": int(args.max_total_patches_cap),
                    "patch": int(args.patch),
                    "mv_unit_div": float(args.mv_unit_div),
                    "mv_pct": float(args.mv_pct),
                    "res_pct": float(args.res_pct),
                    "w_mv": float(args.w_mv),
                    "w_res": float(args.w_res),
                    "num_gops": int(args.num_gops),
                    "gop_anchor_mode": str(args.gop_anchor_mode),
                    "p_full_per_bucket": int(args.p_full_per_bucket),
                    "auto_num_gops": (not bool(args.no_auto_num_gops)),
                    "analysis_candidate_multiplier": float(args.analysis_candidate_multiplier),
                    "analysis_max_frames": int(args.analysis_max_frames),
                    "block_selection_frame_penalty": float(args.block_selection_frame_penalty),
                    "mv_compensate": str(args.mv_compensate),
                    "mv_dir_mode": str(args.mv_dir_mode),
                    "w_mv_l0": float(args.w_mv_l0),
                    "w_mv_l1": float(args.w_mv_l1),
                    "mv_source": str(args.mv_source),
                    "score_source": str(args.score_source),
                    "bitcost_grid": str(args.bitcost_grid),
                    "bitcost_pct": float(args.bitcost_pct),
                    "bitcost_log_scale": bool(args.bitcost_log_scale),
                    "decode_backend": str(args.decode_backend),
                    "fill_to_images": (not bool(args.no_fill_to_images)),
                    "force_fallback_no_cv_reader": bool(args.force_no_cv_reader),
                    "mask_letterbox": (not bool(args.no_mask_letterbox)),
                    "letterbox_dark_thr": float(args.letterbox_dark_thr),
                    "sample_id": str(sample_id),
                    "caption": str(caption),
                    "decode_backsearch_max": int(args.decode_backsearch_max),
                    "per_video_timeout_sec": int(args.per_video_timeout_sec),
                    "group_by_time": bool(args.group_by_time),
                    "keep_blocks_within_canvas": (not bool(args.no_keep_blocks_within_canvas)),
                    "min_pixels": int(args.min_pixels),
                    "max_pixels": int(args.max_pixels),
                    "orig_messages": orig_messages,
                    "skip_black_frames": (not bool(args.no_skip_black_frames)),
                    "skip_corrupt_frames": (not bool(args.no_skip_corrupt_frames)),
                    "black_y_mean_thr": float(args.black_y_mean_thr),
                    "black_y_std_thr": float(args.black_y_std_thr),
                    "corrupt_green_frac_thr": float(args.corrupt_green_frac_thr),
                    "corrupt_g_thr": int(args.corrupt_g_thr),
                    "corrupt_rb_thr": int(args.corrupt_rb_thr),
                    "ensure_per_second": bool(args.ensure_per_second),
                    "sec_stride": float(args.sec_stride),
                    "min_blocks_per_second": int(args.min_blocks_per_second),
                    "collage_patch_order": str(args.collage_patch_order),
                }

                yielded += 1
                if args.max_jobs and yielded >= int(args.max_jobs):
                    return

            if args.scan_log_every and (ln % int(args.scan_log_every) == 0):
                print(f"[scan] ln={ln} yielded={yielded} missing={n_missing} schema_skip={n_skipped_schema} resp_skip={n_skipped_resp}")

        print(f"[scan_done] yielded={yielded} missing={n_missing} schema_skip={n_skipped_schema} resp_skip={n_skipped_resp}")

    # Processing
    print(f"[info] out_root={args.out_root} workers={args.num_workers}")

    ok = skip = fail = 0
    t0 = time.time()

    index_fp = None
    index_ok = 0
    if args.assets_index:
        idx_p = Path(args.assets_index)
        idx_p.parent.mkdir(parents=True, exist_ok=True)
        index_fp = open(idx_p, "a", encoding="utf-8")

    train_fp = None
    if args.out_train_jsonl:
        tp = Path(args.out_train_jsonl)
        tp.parent.mkdir(parents=True, exist_ok=True)
        train_fp = open(tp, "a", encoding="utf-8", buffering=1)

    with Pool(processes=int(args.num_workers)) as pool:
        for i, (st, msg, idx_rec) in enumerate(pool.imap_unordered(_worker, job_iter(), chunksize=int(args.chunksize)), start=1):
            if st == "ok":
                ok += 1
                if index_fp is not None and isinstance(idx_rec, dict) and idx_rec:
                    index_fp.write(json.dumps(idx_rec, ensure_ascii=False) + "\n")
                    index_ok += 1
                    if args.assets_index_flush_every and (index_ok % int(args.assets_index_flush_every) == 0):
                        index_fp.flush()
                
                if train_fp is not None and isinstance(idx_rec, dict) and idx_rec:
                    asset_dir = str(idx_rec.get("asset_dir", ""))
                    asset_dir_abs = str(Path(asset_dir).resolve()) if asset_dir else ""
                    jpg_list = idx_rec.get("jpg") or []
                    images = [str((Path(asset_dir_abs) / str(fn)).resolve()) for fn in jpg_list]
                    
                    pp_path = Path(asset_dir_abs) / "patch_positions.npy"
                    if not pp_path.exists():
                        pp_path = Path(asset_dir_abs) / "src_patch_position.npy"
                    patch_positions = [str(pp_path.resolve())]

                    # Build timestamp dict
                    ts_dict = {}
                    try:
                        sfid = np.load(str(Path(asset_dir) / "src_frame_ids.npy"))
                        sts = np.load(str(Path(asset_dir) / "src_timestamps.npy"))
                        if sfid.shape[0] == sts.shape[0] and sfid.ndim == 1:
                            mask = (sfid >= 0)
                            sfid_v = sfid[mask].astype(np.int64, copy=False)
                            sts_v = sts[mask].astype(np.float64, copy=False)
                            order = np.argsort(sfid_v, kind="stable")
                            sfid_v = sfid_v[order]
                            sts_v = sts_v[order]
                            last_fid = None
                            for fid_i, sec_i in zip(sfid_v.tolist(), sts_v.tolist()):
                                if last_fid is None or int(fid_i) != int(last_fid):
                                    ts_dict[str(int(fid_i))] = format_timestamp_ss(float(sec_i))
                                    last_fid = int(fid_i)
                    except Exception:
                        pass

                    n_img = max(1, len(images))
                    img_prefix = "".join(["<image>\n" for _ in range(n_img)])

                    # Build messages
                    messages_out = []
                    if not bool(args.empty_messages):
                        msgs_in = idx_rec.get("orig_messages")
                        if isinstance(msgs_in, list) and len(msgs_in) > 0:
                            prefixed = False
                            for m in msgs_in:
                                if not isinstance(m, dict):
                                    continue
                                role = m.get("role")
                                content = m.get("content")
                                role_s = str(role) if isinstance(role, str) else ""
                                content_s = str(content) if isinstance(content, str) else ""
                                if role_s == "user" and (not prefixed):
                                    if content_s.lstrip().startswith("<image>"):
                                        messages_out.append({"role": "user", "content": content_s})
                                    else:
                                        messages_out.append({"role": "user", "content": img_prefix + content_s})
                                    prefixed = True
                                elif role_s in ("user", "assistant"):
                                    messages_out.append({"role": role_s, "content": content_s})
                                else:
                                    messages_out.append({"role": role_s or "user", "content": content_s})
                        else:
                            cap = str(idx_rec.get("caption", "") or "")
                            if cap:
                                messages_out = [
                                    {"role": "user", "content": img_prefix},
                                    {"role": "assistant", "content": cap},
                                ]
                            else:
                                messages_out = [{"role": "user", "content": img_prefix}]

                    try:
                        fps_val = float(idx_rec.get("fps", 0.0))
                    except Exception:
                        fps_val = 0.0

                    # Ensure first user message has correct image tags
                    try:
                        if isinstance(messages_out, list):
                            num_imgs = len(images)
                            for _m in messages_out:
                                if isinstance(_m, dict) and _m.get("role") == "user":
                                    _m["content"] = rewrite_user_content_image_tags(_m.get("content", ""), num_imgs)
                                    break
                    except Exception:
                        pass

                    out_item = {
                        "id": str(idx_rec.get("sample_id", "")) or str(idx_rec.get("key", "")),
                        "messages": messages_out,
                        "images": images,
                        "patch_positions": patch_positions,
                        "fps": [fps_val],
                        "images_source": [str(Path(str(idx_rec.get("video", ""))).resolve())] if idx_rec.get("video") else [],
                    }
                    train_fp.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            
            elif st == "skip":
                skip += 1
                if index_fp is not None and isinstance(idx_rec, dict) and idx_rec.get("kind") == "unsupported_codec":
                    index_fp.write(json.dumps(idx_rec, ensure_ascii=False) + "\n")
                    index_ok += 1
                    if args.assets_index_flush_every and (index_ok % int(args.assets_index_flush_every) == 0):
                        index_fp.flush()
            else:
                fail += 1

            if i % 20 == 0 or st == "fail":
                rate = i / max(1e-6, (time.time() - t0))
                print(f"[prog] done={i} ok={ok} skip={skip} fail={fail} rate={rate:.2f}/s last={msg}")

    if index_fp is not None:
        index_fp.flush()
        index_fp.close()

    if train_fp is not None:
        train_fp.flush()
        train_fp.close()

    print(f"[done] ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
