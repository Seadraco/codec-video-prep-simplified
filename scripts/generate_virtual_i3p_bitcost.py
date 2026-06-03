#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virtual I+3P for Long Video — Bitcost-based patch selection.

This script implements a new experimental strategy for long video input construction:
  - Divide video into virtual segments (e.g. 96 segments for 384 total images)
  - Each segment: 1 full-frame anchor + 3 P-collages based on bitcost

Output format remains I+3P style, where I is a full-frame semantic anchor (not
necessarily a real codec I-frame), and P-collages use bitcost-based sparse patches.

Example:
    python scripts/generate_virtual_i3p_bitcost.py \
        --video_path /path/to/video.mp4 \
        --out_dir /path/to/output \
        --total_images 128 \
        --num_segments 32 \
        --patch_size 14 \
        --canvas_size 576 \
        --p_images_per_segment 3 \
        --anchor_mode midpoint \
        --group_size 2 \
        --decay 0.9 \
        --use_temporal_accumulation \
        --use_group_complete \
        --save_debug
"""

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Virtual I+3P long video patch selection using bitcost",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--video_path", type=str, required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for images and metadata",
    )

    # Core parameters
    parser.add_argument(
        "--total_images", type=int, default=128,
        help="Total number of output images (must be divisible by 1 + p_images_per_segment)",
    )
    parser.add_argument(
        "--num_segments", type=int, default=32,
        help="Number of virtual segments. Default: total_images // (1 + p_images_per_segment)",
    )
    parser.add_argument(
        "--patch_size", type=int, default=14,
        help="ViT patch size in pixels (14 or 16)",
    )
    parser.add_argument(
        "--canvas_size", type=int, default=576,
        help="Target canvas size (used for smart_resize factor hint)",
    )
    parser.add_argument(
        "--p_images_per_segment", type=int, default=3,
        help="Number of P-collage images per segment",
    )

    # Anchor selection
    parser.add_argument(
        "--anchor_mode", type=str, default="midpoint",
        choices=["midpoint", "quality"],
        help="Anchor frame selection mode: midpoint or quality-based",
    )

    # Bitcost parameters
    parser.add_argument(
        "--bitcost_path", type=str, default=None,
        help="Optional pre-computed bitcost pickle/npy path (not yet supported, will be fetched on-the-fly)",
    )
    parser.add_argument(
        "--bitcost_grid", type=str, default="sub",
        choices=["sub", "mb", "ctu", "adaptive", "auto"],
        help="Bitcost grid resolution",
    )
    parser.add_argument(
        "--bitcost_pct", type=float, default=99.0,
        help="Percentile for bitcost normalization",
    )
    parser.add_argument(
        "--bitcost_log_scale", action="store_true", default=True,
        help="Apply log1p to bitcost before normalization",
    )
    parser.add_argument(
        "--no_bitcost_log_scale", action="store_false", dest="bitcost_log_scale",
        help="Disable log1p on bitcost",
    )

    # Group and patch selection
    parser.add_argument(
        "--max_patches_per_p_image", type=int, default=None,
        help="Maximum patches per P-collage image (default: full canvas)",
    )
    parser.add_argument(
        "--min_patches_per_p_image", type=int, default=None,
        help="Minimum patches per P-collage image. If not set, defaults to 20%% of canvas. "
             "Groups are filled with connectivity-aware expansion to avoid fragmentation.",
    )
    parser.add_argument(
        "--group_size", type=int, default=2,
        help="Group size for group-complete selection (e.g. 2 for 2x2)",
    )

    # Temporal accumulation
    parser.add_argument(
        "--use_temporal_accumulation", action="store_true", default=False,
        help="Enable temporal score accumulation with decay",
    )
    parser.add_argument(
        "--decay", type=float, default=0.9,
        help="Decay factor for temporal accumulation",
    )

    # Group-complete
    parser.add_argument(
        "--use_group_complete", action="store_true", default=False,
        help="Enable group-complete expansion (2x2 groups)",
    )

    # Temporal balance
    parser.add_argument(
        "--use_temporal_balance", action="store_true", default=False,
        help="Enable temporal balance across sub-buckets",
    )
    parser.add_argument(
        "--temporal_balance_ratio", type=float, default=0.5,
        help="Ratio of budget allocated to per-bucket selection",
    )
    parser.add_argument(
        "--num_buckets_per_p_window", type=int, default=4,
        help="Number of sub-buckets per P-window for temporal balance",
    )

    # Misc
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite existing output",
    )
    parser.add_argument(
        "--save_debug", action="store_true", default=False,
        help="Save debug visualizations",
    )
    parser.add_argument(
        "--skip_black_frames", action="store_true", default=True,
        help="Skip black frames",
    )
    parser.add_argument(
        "--skip_corrupt_frames", action="store_true", default=True,
        help="Skip corrupted frames",
    )
    parser.add_argument(
        "--decode_backend", type=str, default="ffmpeg_native",
        choices=["ffmpeg_native", "auto"],
        help="Frame decode backend",
    )
    parser.add_argument(
        "--parallel_segments", type=int, default=0,
        help="Number of parallel segments for bitcost fetching (0 = serial)",
    )
    parser.add_argument(
        "--threads_per_segment", type=int, default=4,
        help="Threads per segment for parallel bitcost fetching",
    )
    parser.add_argument(
        "--segment_guard_frames", type=int, default=30,
        help="Guard frames around each segment for parallel fetching",
    )
    parser.add_argument(
        "--mask_letterbox", action="store_true", default=True,
        help="Mask letterbox borders in scoring",
    )
    parser.add_argument(
        "--letterbox_dark_thr", type=float, default=16.0,
        help="Darkness threshold for letterbox detection",
    )
    parser.add_argument(
        "--min_pixels", type=int, default=56 * 56,
        help="Minimum pixel count for smart resize",
    )
    parser.add_argument(
        "--max_pixels", type=int, default=768 * 768,
        help="Maximum pixel count for smart resize",
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Validate
    if args.total_images % (1 + args.p_images_per_segment) != 0:
        print(
            f"[warn] total_images={args.total_images} not divisible by "
            f"(1 + p_images_per_segment)={1 + args.p_images_per_segment}. "
            f"Adjusting num_segments.",
            file=sys.stderr,
        )

    expected_segments = args.total_images // (1 + args.p_images_per_segment)
    if args.num_segments != expected_segments:
        print(
            f"[info] Using num_segments={min(args.num_segments, expected_segments)} "
            f"(requested={args.num_segments}, expected from total_images={expected_segments})",
        )

    from codec_selector.virtual_i3p.pipeline import process_video_virtual_i3p

    status, msg, meta = process_video_virtual_i3p(
        video_path=str(args.video_path),
        out_dir=str(args.out_dir),
        total_images=int(args.total_images),
        num_segments=int(args.num_segments),
        patch_size=int(args.patch_size),
        canvas_size=int(args.canvas_size),
        p_images_per_segment=int(args.p_images_per_segment),
        anchor_mode=str(args.anchor_mode),
        bitcost_path=str(args.bitcost_path) if args.bitcost_path else None,
        max_patches_per_p_image=int(args.max_patches_per_p_image) if args.max_patches_per_p_image else None,
        min_patches_per_p_image=int(args.min_patches_per_p_image) if args.min_patches_per_p_image else None,
        group_size=int(args.group_size),
        decay=float(args.decay),
        use_temporal_accumulation=bool(args.use_temporal_accumulation),
        use_group_complete=bool(args.use_group_complete),
        use_temporal_balance=bool(args.use_temporal_balance),
        temporal_balance_ratio=float(args.temporal_balance_ratio),
        num_buckets_per_p_window=int(args.num_buckets_per_p_window),
        overwrite=bool(args.overwrite),
        save_debug=bool(args.save_debug),
        bitcost_grid=str(args.bitcost_grid),
        bitcost_pct=float(args.bitcost_pct),
        bitcost_log_scale=bool(args.bitcost_log_scale),
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
        skip_black_frames=bool(args.skip_black_frames),
        skip_corrupt_frames=bool(args.skip_corrupt_frames),
        decode_backend=str(args.decode_backend),
        parallel_segments=int(args.parallel_segments),
        threads_per_segment=int(args.threads_per_segment),
        segment_guard_frames=int(args.segment_guard_frames),
        mask_letterbox=bool(args.mask_letterbox),
        letterbox_dark_thr=float(args.letterbox_dark_thr),
    )

    print(f"[{status}] {msg}")
    if meta:
        print(f"  Output: {meta.get('out_dir', args.out_dir)}")
        print(f"  Images: {meta.get('num_images_output', 'N/A')}")
        print(f"  Segments: {meta.get('num_segments', 'N/A')}")

    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
