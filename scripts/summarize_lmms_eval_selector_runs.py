#!/usr/bin/env python3
"""Summarize lmms-eval accuracy, preprocessing, and selector metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def parse_sample(row: dict[str, Any], source: Path) -> dict[str, Any]:
    if isinstance(row.get("videomme_perception_score"), dict):
        payload = row["videomme_perception_score"]
        return {
            "score": float(payload.get("score", 0.0)),
            "duration": str(payload.get("duration", "unknown")),
            "question_type": str(payload.get("task_category", "unknown")),
            "video_id": str(payload.get("videoID", "unknown")),
        }
    if isinstance(row.get("avg_accuracy"), dict):
        payload = row["avg_accuracy"]
        return {
            "score": float(payload.get("rating", payload.get("match_success", 0.0))),
            "duration": "short",
            "question_type": str(payload.get("dim", "unknown")),
            "video_id": str(payload.get("video_id", "unknown")),
        }
    raise ValueError(f"unsupported sample schema: {source}")


def score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = float(sum(row["score"] for row in rows))
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
    }


def grouped_scores(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: score_summary(value) for key, value in sorted(groups.items())}


def load_samples(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*samples*.jsonl")):
        rows.extend(parse_sample(row, path) for row in read_jsonl(path))
    if not rows:
        raise FileNotFoundError(f"no lmms-eval sample JSONL found under {run_dir}")
    return rows


def load_preprocess(run_dir: Path) -> dict[str, Any]:
    per_video: dict[str, float] = {}
    duplicate_rows = 0
    for path in sorted((run_dir / "codec_preprocess_audit").glob("*.jsonl")):
        for row in read_jsonl(path):
            video = str(row.get("video", "unknown"))
            value = float(row.get("subprocess_ms", 0.0))
            if video in per_video:
                duplicate_rows += 1
                per_video[video] = max(per_video[video], value)
            else:
                per_video[video] = value
    values = list(per_video.values())
    return {
        "videos": len(values),
        "duplicate_audit_rows": duplicate_rows,
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "mean_ms": statistics.fmean(values) if values else None,
        "sum_ms": sum(values),
    }


def load_selector(run_dir: Path) -> dict[str, Any] | None:
    summaries: list[dict[str, Any]] = []
    for path in sorted((run_dir / "online_codec_cache").glob("*/meta.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = payload.get("selector_summary")
        if isinstance(summary, dict):
            summaries.append(summary)
    if not summaries:
        return None

    group_count = sum(int(item.get("group_count", 0)) for item in summaries)
    target_blocks = sum(int(item.get("target_blocks", 0)) for item in summaries)
    mad_count = sum(int(item.get("adjacent_mad_count", 0)) for item in summaries)
    adjacent_pairs = sum(
        int(item.get("adjacent_same_position_pairs", 0)) for item in summaries
    )
    adjacent_duplicates = sum(
        int(item.get("adjacent_same_position_duplicates", 0)) for item in summaries
    )

    def weighted(field: str, weight_field: str = "target_blocks") -> float:
        denominator = sum(int(item.get(weight_field, 0)) for item in summaries)
        if denominator == 0:
            return 0.0
        return sum(
            float(item.get(field, 0.0)) * int(item.get(weight_field, 0))
            for item in summaries
        ) / denominator

    cdf_thresholds = sorted(
        {
            threshold
            for item in summaries
            for threshold in (item.get("adjacent_mad_cdf") or {})
        }
    )
    return {
        "videos": len(summaries),
        "groups": group_count,
        "target_blocks": target_blocks,
        "bitcost_selected": sum(
            int(item.get("bitcost_selected", 0)) for item in summaries
        ),
        "diversity_selected": sum(
            int(item.get("diversity_selected", 0)) for item in summaries
        ),
        "backfill_selected": sum(
            int(item.get("backfill_selected", 0)) for item in summaries
        ),
        "dedup_rejected": sum(
            int(item.get("dedup_rejected", 0)) for item in summaries
        ),
        "dedup_rejected_unique": sum(
            int(item.get("dedup_rejected_unique", 0)) for item in summaries
        ),
        "dedup_comparisons": sum(
            int(item.get("dedup_comparisons", 0)) for item in summaries
        ),
        "mean_unique_source_frames_per_group": (
            sum(
                float(item.get("mean_unique_source_frames_per_group", 0.0))
                * int(item.get("group_count", 0))
                for item in summaries
            )
            / group_count
            if group_count
            else 0.0
        ),
        "mean_unique_spatial_positions_per_group": (
            sum(
                float(item.get("mean_unique_spatial_positions_per_group", 0.0))
                * int(item.get("group_count", 0))
                for item in summaries
            )
            / group_count
            if group_count
            else 0.0
        ),
        "mean_temporal_distribution_entropy_normalized": weighted(
            "mean_temporal_distribution_entropy_normalized"
        ),
        "mean_max_blocks_per_frame_fraction": weighted(
            "mean_max_blocks_per_frame_fraction"
        ),
        "adjacent_same_position_pairs": adjacent_pairs,
        "adjacent_same_position_duplicates": adjacent_duplicates,
        "adjacent_same_position_duplicate_rate": (
            adjacent_duplicates / adjacent_pairs if adjacent_pairs else 0.0
        ),
        "selected_bitcost_mean": weighted("selected_bitcost_mean"),
        "selected_novelty_mean": weighted("selected_novelty_mean"),
        "selected_edge_mean": weighted("selected_edge_mean"),
        "adjacent_mad_count": mad_count,
        "adjacent_mad_fraction_le_threshold": weighted(
            "mean_adjacent_mad_fraction_le_threshold",
            "adjacent_mad_count",
        ),
        "adjacent_mad_cdf": {
            threshold: (
                sum(
                    float((item.get("adjacent_mad_cdf") or {}).get(threshold, 0.0))
                    * int(item.get("adjacent_mad_count", 0))
                    for item in summaries
                )
                / mad_count
            )
            for threshold in cdf_thresholds
        }
        if mad_count
        else {},
        "selector_timing_sec": {
            field: sum(
                float((item.get("timing_sec") or {}).get(field, 0.0))
                for item in summaries
            )
            for field in ("descriptor", "score", "dedup_map", "selection", "total")
        },
    }


def summarize_run(label: str, run_dir: Path) -> dict[str, Any]:
    rows = load_samples(run_dir)
    return {
        "label": label,
        "path": str(run_dir.resolve()),
        "scores": {
            "overall": score_summary(rows),
            "by_duration": grouped_scores(rows, "duration"),
            "by_question_type": grouped_scores(rows, "question_type"),
        },
        "preprocess": load_preprocess(run_dir),
        "selector": load_selector(run_dir),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Simplified Codec selector experiments",
        "",
        "## Accuracy",
        "",
        "| Run | Overall | Short | Long |",
        "|---|---:|---:|---:|",
    ]
    for run in report["runs"]:
        scores = run["scores"]

        def accuracy(group: dict[str, Any] | None) -> str:
            if not group or group["accuracy"] is None:
                return "n/a"
            return f"{100.0 * group['accuracy']:.2f}% ({group['correct']:.0f}/{group['n']})"

        lines.append(
            "| {label} | {overall} | {short} | {long} |".format(
                label=run["label"],
                overall=accuracy(scores["overall"]),
                short=accuracy(scores["by_duration"].get("short")),
                long=accuracy(scores["by_duration"].get("long")),
            )
        )
    lines.extend(
        [
            "",
            "## Preprocessing and selector",
            "",
            "| Run | Preprocess P50/P90 | Unique frames/group | Entropy | "
            "Max-frame share | Dedup rejected | Backfill | Mean bit-cost |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in report["runs"]:
        prep = run["preprocess"]
        selector = run["selector"] or {}
        lines.append(
            f"| {run['label']} | {prep['p50_ms'] or 0:.1f}/{prep['p90_ms'] or 0:.1f} ms "
            f"| {selector.get('mean_unique_source_frames_per_group', 0):.2f} "
            f"| {selector.get('mean_temporal_distribution_entropy_normalized', 0):.4f} "
            f"| {selector.get('mean_max_blocks_per_frame_fraction', 0):.4f} "
            f"| {selector.get('dedup_rejected_unique', 0)} "
            f"| {selector.get('backfill_selected', 0)} "
            f"| {selector.get('selected_bitcost_mean', 0):.4f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeat for every lmms-eval run",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = []
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"--run must be LABEL=PATH, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        runs.append(summarize_run(label, Path(raw_path)))
    report = {"runs": runs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(markdown(report), encoding="utf-8")

    rows = []
    for run in runs:
        for question_type, score in run["scores"]["by_question_type"].items():
            rows.append(
                {
                    "run": run["label"],
                    "question_type": question_type,
                    **score,
                }
            )
    with (args.output_dir / "question_type.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run", "question_type", "n", "correct", "accuracy"),
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
