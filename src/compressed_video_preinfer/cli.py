"""Command line entrypoint for compressed-video pre-infer."""

from __future__ import annotations

import argparse
import json

from .api import run_preinfer


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run optimized compressed-video pre-infer.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_sampled_frames", type=int, default=1024)
    ap.add_argument("--group_size", type=int, default=32)
    ap.add_argument("--images_per_group", type=int, default=4)
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--max_pixels", type=int, default=153664)
    ap.add_argument("--min_group_frames", type=int, default=8)
    ap.add_argument("--max_group_frames", type=int, default=64)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    result = run_preinfer(**vars(args))
    print(json.dumps({"out_dir": result.out_dir, "meta_path": result.meta_path, "timings": result.timings}, indent=2))


if __name__ == "__main__":
    main()

