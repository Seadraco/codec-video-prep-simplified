"""Command line entrypoint for codec-aware video preprocessing."""

from __future__ import annotations

import argparse
import json

from .api import run_preinfer


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run optimized codec-aware video preprocessing.")
    # basic params
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_sampled_frames", type=int, default=1024)
    ap.add_argument("--group_size", type=int, default=32)
    ap.add_argument("--target_canvas", type=int, default=0,
                    help="Target total canvas count for adaptive GOP ablation; 0=unbounded")
    ap.add_argument("--images_per_group", type=int, default=4)
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--max_pixels", type=int, default=153664)
    ap.add_argument("--block_size", type=int, default=2,
                    help="Block size for patch grouping and scoring")
    ap.add_argument("--min_group_frames", type=int, default=8)
    ap.add_argument("--max_group_frames", type=int, default=64)
    ap.add_argument("--thread_type", default="slice", choices=["auto", "slice", "frame"],
                    help="FFmpeg decoder thread type for cv_reader_fast")
    ap.add_argument("--thread_count", type=int, default=1,
                    help="FFmpeg decoder thread count for cv_reader_fast")
    ap.add_argument("--disable_target_only", action="store_true",
                    help="Disable target-only bitcost pruning for legacy-compatible bitcost maps")
    ap.add_argument("--bitcost_grid", default="adaptive", choices=["sub", "mb", "ctu", "adaptive"])
    ap.add_argument("--canvas_format", default="jpg", choices=["jpg", "png", "npy"])
    ap.add_argument("--grouping_mode", default="readiness", choices=["fixed", "readiness"],
                    help="fixed=uniform group_size; readiness=dynamic by content")
    ap.add_argument("--frame_sampling_mode", default="uniform_count",
                    choices=["uniform_count", "fps", "pkt_peak"],
                    help="Frame sampling strategy")
    ap.add_argument("--sample_fps", type=float, default=4.0,
                    help="Target sample fps when frame_sampling_mode=fps")

    # readiness threshold
    ap.add_argument("--readiness_sum_threshold", type=float, default=0.0,
                    help="Readiness threshold; 0=auto estimate")
    ap.add_argument("--readiness_sum_threshold_mode", default="legacy",
                    choices=["legacy", "auto", "clamped_sqrt_bpppf", "fixed"],
                    help="How to compute readiness threshold (ignored when grouping_mode=fixed)")
    ap.add_argument("--readiness_norm_sum_threshold", type=float, default=2250000.0,
                    help="Normalization threshold for readiness")
    ap.add_argument("--iframe_score_clip_mode", default="none", choices=["none", "non_i_percentile"],
                    help="Clip I-frame score maps before readiness grouping")
    ap.add_argument("--iframe_score_clip_percentile", type=float, default=95.0,
                    help="Non-I score percentile used as I-frame clip cap")

    # misc
    ap.add_argument("--avoid_keyframes", action="store_true", default=True,
                    help="Avoid keyframes in sampling")
    ap.add_argument("--no_avoid_keyframes", action="store_true",
                    help="Disable keyframe avoidance")
    ap.add_argument("--save_mask_video", action="store_true",
                    help="Save side-by-side mask visualization video")
    ap.add_argument("--verbose", action="store_true",
                    help="Print debug logs")
    ap.add_argument("--adaptive_anchor", action="store_true",
                    help="Select each group's full-frame anchor adaptively")
    ap.add_argument("--adaptive_anchor_w_center", type=float, default=0.7,
                    help="Adaptive anchor temporal-center weight")
    ap.add_argument("--adaptive_anchor_w_low_bc", type=float, default=0.3,
                    help="Adaptive anchor low bit-cost weight")
    ap.add_argument("--adaptive_gop", action="store_true",
                    help="Split groups adaptively using frame bit-cost as predictive-cost proxy")
    ap.add_argument("--adaptive_gop_gamma", type=float, default=0.0,
                    help="Adaptive GOP split threshold; 0 uses percentile")
    ap.add_argument("--adaptive_gop_percentile", type=float, default=75.0,
                    help="Auto threshold percentile for adaptive GOP")
    ap.add_argument("--event_aggregation", action="store_true",
                    help="Reserve transition block budget across temporal event bins")
    ap.add_argument("--event_aggregation_bins", type=int, default=4,
                    help="Temporal bins used for event aggregation")
    ap.add_argument("--event_aggregation_min_blocks", type=int, default=8,
                    help="Minimum transition blocks reserved per non-empty event bin")
    ap.add_argument("--parallel_decode_cv_reader", action="store_true",
                    help="Parallelize decode and cv_reader")
    ap.add_argument("--decode_backend", default="cv_reader_pixels",
                    choices=["ffmpeg_native", "cv_reader_pixels"],
                    help="Decode backend")
    ap.add_argument("--parallel_segments", type=int, default=0,
                    help="Number of parallel decode segments (0=disabled)")
    ap.add_argument("--threads_per_segment", type=int, default=4,
                    help="FFmpeg threads per segment worker")
    ap.add_argument("--segment_guard_frames", type=int, default=30,
                    help="Extra frames before/after each segment for seek accuracy")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    kwargs = vars(args).copy()

    # handle avoid_keyframes flag pair
    if kwargs.pop("no_avoid_keyframes", False):
        kwargs["avoid_keyframes"] = False
    elif kwargs.get("avoid_keyframes") is None:
        kwargs["avoid_keyframes"] = True

    result = run_preinfer(**kwargs)
    print(json.dumps({"out_dir": result.out_dir, "meta_path": result.meta_path, "timings": result.timings}, indent=2))


if __name__ == "__main__":
    main()
