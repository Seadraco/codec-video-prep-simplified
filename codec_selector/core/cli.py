#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI adapter for the bitcost readiness selector engine."""

import argparse

from .config import BitcostReadinessConfig


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the reusable bitcost codec patch selector")
    ap.add_argument("--video", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--frame_sampling_mode", default="fps", choices=["fps", "uniform_count", "pkt_size_peak", "fps_plus_pkt_size_peak", "all_frames"])
    ap.add_argument("--sample_fps", default=4.0, type=float)
    ap.add_argument("--num_sampled_frames", default=0, type=int)
    ap.add_argument("--pkt_peak_bin_sec", default=0.5, type=float)
    ap.add_argument("--pkt_peak_cap", default=128, type=int)
    ap.add_argument("--pkt_peak_per_sec", default=1.0, type=float)
    ap.add_argument("--pkt_peak_neighbor", default=0, type=int)
    ap.add_argument("--pkt_peak_smooth_bins", default=1, type=int)
    ap.add_argument("--group_size", default=32, type=int)
    ap.add_argument("--target_canvas", default=0, type=int)
    ap.add_argument("--grouping_mode", default="fixed", choices=["fixed", "readiness"])
    ap.add_argument("--images_per_group", default=4, type=int)
    ap.add_argument("--patch", default=14, type=int)
    ap.add_argument("--block_size", default=2, type=int)
    ap.add_argument("--max_dim", default=616, type=int)
    ap.add_argument("--max_pixels", default=150000, type=int)
    ap.add_argument("--no_resize", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--bitcost_grid", default="sub", choices=["sub", "mb", "ctu", "adaptive"])
    ap.add_argument("--bitcost_pct", default=99.0, type=float)
    ap.add_argument("--bitcost_log_scale", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--frame_score_norm_mode", default="none", choices=["none", "frame_mean_floor"])
    ap.add_argument("--frame_score_norm_floor_ratio", default=0.5, type=float)
    ap.add_argument("--frame_score_norm_allow_boost", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--iframe_score_clip_mode", default="none", choices=["none", "non_i_percentile"])
    ap.add_argument("--iframe_score_clip_percentile", default=95.0, type=float)
    ap.add_argument("--avoid_keyframes", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--avoid_keyframe_offset", default=1, type=int)
    ap.add_argument("--canvas_format", default="jpg", choices=["jpg", "png"])
    ap.add_argument("--save_mask_video", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--video_fps", default=0.0, type=float)
    ap.add_argument("--verbose", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--adaptive_anchor", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--adaptive_anchor_w_center", default=0.7, type=float)
    ap.add_argument("--adaptive_anchor_w_low_bc", default=0.3, type=float)
    ap.add_argument("--adaptive_gop", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--adaptive_gop_gamma", default=0.0, type=float)
    ap.add_argument("--adaptive_gop_percentile", default=75.0, type=float)
    ap.add_argument("--event_aggregation", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--event_aggregation_bins", default=4, type=int)
    ap.add_argument("--event_aggregation_min_blocks", default=8, type=int)
    ap.add_argument("--selector_mode", default="topk_2x2_bitcost", choices=["topk_2x2_bitcost", "diverse_mixed_simple"])
    ap.add_argument("--dedup_descriptor", default="pooled4", choices=["pooled4", "full"])
    ap.add_argument("--readiness_sum_threshold", default=0.0, type=float)
    ap.add_argument("--readiness_sum_threshold_mode", default="legacy", choices=["legacy", "fixed", "auto", "clamped_sqrt_bpppf"])
    ap.add_argument("--readiness_norm_sum_threshold", default=1050000.0, type=float)
    ap.add_argument("--bpppf_clamp_min", default=0.015, type=float)
    ap.add_argument("--bpppf_clamp_max", default=0.09, type=float)
    ap.add_argument("--min_group_frames", default=8, type=int)
    ap.add_argument("--max_group_frames", default=64, type=int)
    ap.add_argument("--min_group_sec", default=0.0, type=float)
    ap.add_argument("--max_group_sec", default=0.0, type=float)
    ap.add_argument("--readiness_coverage_bins", default=3, type=int)
    ap.add_argument("--readiness_delta_ratio", default=0.05, type=float)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    from .pipeline import run_bitcost_readiness

    cfg = BitcostReadinessConfig(**vars(args))
    result = run_bitcost_readiness(cfg)
    print(f"[done] sampled={result.summary.get('sampled_frames')} groups={result.summary.get('num_groups')} out_dir={result.out_dir}")


if __name__ == "__main__":
    main()
