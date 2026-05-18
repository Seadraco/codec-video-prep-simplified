#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Energy-based sampling and window selection algorithms."""

import math
from typing import Tuple, List, Dict, Any
import numpy as np

from .video_probe import ffprobe_packets_pb_energy_bins, ffprobe_sum_pkt_size


def pick_peak_frame_ids_from_pkt_size(
    video_path: str,
    fps: float,
    total_frames: int,
    bin_sec: float = 0.5,
    peaks_cap: int = 8,
    peaks_per_sec: float = 0.5,
    neighbor: int = 1,
    smooth_bins: int = 1,
) -> Tuple[List[int], Dict[str, Any]]:
    """Select peak frame ids using PB packet-size energy bins (exclude keyframes)."""
    fps_use = float(fps) if (fps and fps > 0) else 30.0
    total_frames = int(max(0, total_frames))

    centers, energy, dbg_bins = ffprobe_packets_pb_energy_bins(
        video_path=str(video_path),
        bin_sec=float(bin_sec),
        smooth_bins=int(smooth_bins),
    )

    dbg: Dict[str, Any] = {
        "bin_sec": float(bin_sec),
        "peaks_cap": int(peaks_cap),
        "peaks_per_sec": float(peaks_per_sec),
        "neighbor": int(neighbor),
        "smooth_bins": int(smooth_bins),
        "bins": dbg_bins,
        "peak_bins": [],
        "peak_times_sec": [],
        "peak_frame_ids": [],
    }

    if energy.size == 0 or centers.size == 0:
        dbg["error"] = "no_energy"
        return [], dbg

    duration_sec = float(total_frames) / float(fps_use) if total_frames > 0 else float(dbg_bins.get("duration_est_sec", 0.0))
    duration_sec = max(duration_sec, 0.0)

    k = int(max(1, round(duration_sec * float(peaks_per_sec)))) if duration_sec > 0 else 1
    k = int(min(int(peaks_cap), k, int(energy.size)))

    idx = np.argsort(-energy)[:k]
    idx = [int(i) for i in idx.tolist()]
    dbg["peak_bins"] = idx

    peak_times = [float(centers[i]) for i in idx]
    dbg["peak_times_sec"] = peak_times

    out_set: set = set()
    nb = int(max(0, int(neighbor)))
    for t in peak_times:
        fid0 = int(round(float(t) * float(fps_use)))
        if total_frames > 0:
            fid0 = max(0, min(total_frames - 1, fid0))
        for d in range(-nb, nb + 1):
            fid = int(fid0 + d)
            if total_frames > 0:
                fid = max(0, min(total_frames - 1, fid))
            out_set.add(int(fid))

    out = sorted(out_set)
    dbg["peak_frame_ids"] = out
    return out, dbg


def build_variable_length_gops_by_energy(
    video_path: str,
    total_frames: int,
    fps: float,
    target_num_gops: int,
    bin_sec: float = 0.5,
    smooth_bins: int = 1,
    min_span_sec: float = 1.5,
    max_span_sec: float = 6.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build variable-length GOPs by cumulative PB packet-size energy.

    V2.2 implementation:
    - compute PB-only packet-size energy bins
    - accumulate smoothed energy over time
    - cut a GOP when cumulative energy reaches a target threshold, subject to
      min/max span constraints
    - prefer a local low-energy valley near the threshold crossing so cut points
      land on more stable boundaries
    - each GOP records a start / end / anchor time
    """
    total_frames = int(max(0, total_frames))
    fps_use = float(fps) if (fps and fps > 0) else 30.0
    target_num_gops = int(max(1, int(target_num_gops)))
    bin_sec = float(bin_sec) if (bin_sec and bin_sec > 1e-6) else 0.5
    smooth_bins = int(max(0, int(smooth_bins)))
    min_span_sec = float(max(0.25, float(min_span_sec)))
    max_span_sec = float(max(min_span_sec, float(max_span_sec)))

    dbg: Dict[str, Any] = {
        "mode": "variable_gop_cumulative_energy",
        "target_num_gops": int(target_num_gops),
        "bin_sec": float(bin_sec),
        "smooth_bins": int(smooth_bins),
        "min_span_sec": float(min_span_sec),
        "max_span_sec": float(max_span_sec),
        "segments": [],
        "bins": {},
    }

    if total_frames <= 0:
        seg = {
            "gop_idx": 0,
            "start_bin": 0,
            "end_bin": 0,
            "start_sec": 0.0,
            "end_sec": 0.0,
            "anchor_sec": 0.0,
            "energy_sum": 0.0,
        }
        dbg["segments"] = [seg]
        dbg["segment_count"] = 1
        dbg["cut_energy_threshold"] = 0.0
        return [seg], dbg

    centers, energy, bins_dbg = ffprobe_packets_pb_energy_bins(
        video_path=str(video_path),
        bin_sec=float(bin_sec),
        smooth_bins=int(smooth_bins),
    )
    dbg["bins"] = bins_dbg

    duration_sec = float(total_frames) / float(fps_use) if total_frames > 0 else 0.0
    duration_sec = max(duration_sec, 0.0)

    if energy.size == 0 or centers.size == 0:
        segs: List[Dict[str, Any]] = []
        n = int(max(1, target_num_gops))
        sec_edges = np.linspace(0.0, max(duration_sec, bin_sec), n + 1, dtype=np.float64).tolist()
        for gi in range(n):
            st_sec = float(sec_edges[gi])
            ed_sec = float(sec_edges[gi + 1])
            anchor_sec = 0.5 * (st_sec + ed_sec)
            segs.append({
                "gop_idx": int(gi),
                "start_bin": int(gi),
                "end_bin": int(gi),
                "start_sec": float(st_sec),
                "end_sec": float(ed_sec),
                "anchor_sec": float(anchor_sec),
                "energy_sum": 0.0,
            })
        dbg["segment_count"] = int(len(segs))
        dbg["cut_energy_threshold"] = 0.0
        dbg["fallback"] = "uniform_time"
        dbg["segments"] = segs
        return segs, dbg

    energy = np.asarray(energy, dtype=np.float64)
    energy = np.maximum(energy, 0.0)
    nb = int(len(energy))

    total_energy = float(energy.sum())
    if (not np.isfinite(total_energy)) or total_energy <= 1e-12:
        total_energy = float(nb)
        energy = np.ones((nb,), dtype=np.float64)

    cut_energy_threshold = float(total_energy) / float(max(1, target_num_gops))
    min_bins = int(max(1, math.ceil(float(min_span_sec) / float(bin_sec))))
    max_bins = int(max(min_bins, math.ceil(float(max_span_sec) / float(bin_sec))))

    dbg["total_energy"] = float(total_energy)
    dbg["cut_energy_threshold"] = float(cut_energy_threshold)
    dbg["min_bins"] = int(min_bins)
    dbg["max_bins"] = int(max_bins)
    dbg["cut_points"] = []

    segs: List[Dict[str, Any]] = []
    start_bin = 0
    acc = 0.0
    gi = 0
    valley_search_radius = int(max(1, min(3, max_bins // 3)))

    def _best_cut_bin(start_idx: int, current_idx: int) -> int:
        lo = int(max(int(start_idx) + int(min_bins) - 1, int(current_idx) - int(valley_search_radius)))
        hi = int(min(int(start_idx) + int(max_bins) - 1, int(nb) - 1, int(current_idx) + int(valley_search_radius)))
        if hi < lo:
            return int(current_idx)
        best = int(current_idx)
        best_key = (float(energy[best]), abs(int(best) - int(current_idx)))
        for bi in range(int(lo), int(hi) + 1):
            cand_key = (float(energy[bi]), abs(int(bi) - int(current_idx)))
            if cand_key < best_key:
                best = int(bi)
                best_key = cand_key
        return int(best)

    i = 0
    while i < nb:
        acc += float(energy[i])
        span_bins = int(i - start_bin + 1)
        reach_min = bool(span_bins >= min_bins)
        reach_max = bool(span_bins >= max_bins)
        reach_energy = bool(acc >= cut_energy_threshold)

        should_cut = False
        if reach_max:
            should_cut = True
        elif reach_min and reach_energy:
            should_cut = True

        if should_cut:
            cut_bin = _best_cut_bin(int(start_bin), int(i))
            st_sec = float(start_bin) * float(bin_sec)
            ed_sec = min(duration_sec, float(cut_bin + 1) * float(bin_sec))
            anchor_bin = int((start_bin + cut_bin) // 2)
            anchor_sec = float(centers[max(0, min(nb - 1, anchor_bin))])
            seg_energy = float(energy[start_bin:cut_bin + 1].sum())
            segs.append({
                "gop_idx": int(gi),
                "start_bin": int(start_bin),
                "end_bin": int(cut_bin),
                "start_sec": float(st_sec),
                "end_sec": float(ed_sec),
                "anchor_sec": float(anchor_sec),
                "energy_sum": float(seg_energy),
            })
            dbg["cut_points"].append({
                "gop_idx": int(gi),
                "threshold_bin": int(i),
                "cut_bin": int(cut_bin),
                "cut_sec": float(ed_sec),
            })
            gi += 1
            start_bin = int(cut_bin + 1)
            acc = 0.0
            i = int(cut_bin + 1)
            continue
        i += 1

    if start_bin < nb:
        st_sec = float(start_bin) * float(bin_sec)
        ed_sec = min(duration_sec, float(nb) * float(bin_sec))
        anchor_bin = int((start_bin + (nb - 1)) // 2)
        anchor_sec = float(centers[max(0, min(nb - 1, anchor_bin))])
        tail_energy = float(energy[start_bin:nb].sum())
        segs.append({
            "gop_idx": int(gi),
            "start_bin": int(start_bin),
            "end_bin": int(nb - 1),
            "start_sec": float(st_sec),
            "end_sec": float(ed_sec),
            "anchor_sec": float(anchor_sec),
            "energy_sum": float(tail_energy),
        })

    # Merge last segment if too short
    if len(segs) >= 2:
        last = segs[-1]
        span_last = float(last["end_sec"]) - float(last["start_sec"])
        if span_last < float(min_span_sec) * 0.5:
            prev = segs[-2]
            prev["end_bin"] = int(last["end_bin"])
            prev["end_sec"] = float(last["end_sec"])
            prev["energy_sum"] = float(prev["energy_sum"]) + float(last["energy_sum"])
            prev_anchor_bin = int((int(prev["start_bin"]) + int(prev["end_bin"])) // 2)
            prev["anchor_sec"] = float(centers[max(0, min(nb - 1, prev_anchor_bin))])
            segs = segs[:-1]

    for gi, seg in enumerate(segs):
        seg["gop_idx"] = int(gi)

    dbg["segment_count"] = int(len(segs))
    dbg["segments"] = [
        {
            "gop_idx": int(seg["gop_idx"]),
            "start_bin": int(seg["start_bin"]),
            "end_bin": int(seg["end_bin"]),
            "start_sec": float(seg["start_sec"]),
            "end_sec": float(seg["end_sec"]),
            "anchor_sec": float(seg["anchor_sec"]),
            "energy_sum": float(seg["energy_sum"]),
        }
        for seg in segs
    ]
    return segs, dbg


def sample_frame_ids_by_energy_cdf(
    video_path: str,
    total_frames: int,
    fps: float,
    target_count: int,
    bin_sec: float = 0.5,
    smooth_bins: int = 1,
    uniform_mix: float = 0.15,
    max_per_bin: int = 16,
) -> Tuple[List[int], Dict[str, Any]]:
    """Sample frame ids with density proportional to PB packet-size energy.

    Intuition:
      - high-energy time bins get more sampled frames
      - low-energy time bins get fewer sampled frames
      - a small uniform prior prevents static-but-important segments from being starved

    This effectively builds a new non-uniform timeline where "information-dense"
    regions are sampled more densely.
    """
    total_frames = int(max(0, total_frames))
    target_count = int(max(0, target_count))
    fps_use = float(fps) if (fps and fps > 0) else 30.0
    bin_sec = float(bin_sec) if (bin_sec and bin_sec > 1e-6) else 0.5
    smooth_bins = int(max(0, int(smooth_bins)))
    uniform_mix = float(max(0.0, min(1.0, float(uniform_mix))))
    max_per_bin = int(max(1, int(max_per_bin)))

    dbg: Dict[str, Any] = {
        "mode": "pkt_energy_cdf",
        "target_count": int(target_count),
        "bin_sec": float(bin_sec),
        "smooth_bins": int(smooth_bins),
        "uniform_mix": float(uniform_mix),
        "max_per_bin": int(max_per_bin),
        "bins": {},
        "bin_count": 0,
        "selected_before_pad": 0,
        "selected_after_pad": 0,
        "per_bin_counts": [],
    }

    if total_frames <= 0 or target_count <= 0:
        dbg["error"] = "empty_input"
        return [], dbg

    centers, energy, bins_dbg = ffprobe_packets_pb_energy_bins(
        video_path=str(video_path),
        bin_sec=float(bin_sec),
        smooth_bins=int(smooth_bins),
    )
    dbg["bins"] = bins_dbg

    if energy.size == 0 or centers.size == 0:
        dbg["error"] = "no_energy"
        if total_frames == 1:
            return [0] * target_count, dbg
        out = np.linspace(0, total_frames - 1, target_count, dtype=np.int32).tolist()
        dbg["selected_before_pad"] = int(len(out))
        dbg["selected_after_pad"] = int(len(out))
        return [int(x) for x in out], dbg

    nb = int(len(energy))
    dbg["bin_count"] = int(nb)

    energy = np.asarray(energy, dtype=np.float64)
    energy = np.maximum(energy, 0.0)

    mean_e = float(energy.mean()) if energy.size > 0 else 0.0
    if (not np.isfinite(mean_e)) or mean_e <= 0.0:
        mean_e = 1.0

    # Mix in a uniform prior so static-but-important regions still get sampled.
    weights = (1.0 - uniform_mix) * energy + uniform_mix * mean_e
    weights = np.maximum(weights, 1e-12)
    weights = weights / float(weights.sum())

    ideal = weights * float(target_count)
    counts = np.floor(ideal).astype(np.int32)
    frac = ideal - counts.astype(np.float64)

    # Respect a per-bin cap so a single bursty region does not absorb everything.
    counts = np.minimum(counts, int(max_per_bin)).astype(np.int32)
    cur = int(counts.sum())

    if cur < target_count:
        order = np.argsort(-frac)
        for idx in order.tolist():
            if cur >= target_count:
                break
            if int(counts[idx]) >= int(max_per_bin):
                continue
            counts[idx] += 1
            cur += 1

    if cur < target_count:
        order = np.argsort(-weights)
        ptr = 0
        while cur < target_count and len(order) > 0:
            idx = int(order[ptr % len(order)])
            if int(counts[idx]) < int(max_per_bin):
                counts[idx] += 1
                cur += 1
            ptr += 1
            if ptr > int(target_count * 8 + nb * 8):
                break

    frame_ids: List[int] = []
    per_bin_counts: List[Tuple[int, int]] = []
    for bi in range(nb):
        c = int(counts[bi])
        if c <= 0:
            continue
        per_bin_counts.append((int(bi), int(c)))

        t0 = float(bi) * float(bin_sec)
        t1 = float(bi + 1) * float(bin_sec)
        f0 = int(math.floor(t0 * fps_use))
        f1 = int(math.ceil(t1 * fps_use)) - 1
        if total_frames > 0:
            f0 = max(0, min(total_frames - 1, f0))
            f1 = max(0, min(total_frames - 1, f1))
        if f1 < f0:
            f1 = f0

        if c == 1 or f1 == f0:
            picked = [int(round((f0 + f1) * 0.5))]
        else:
            picked = np.linspace(f0, f1, c, dtype=np.int32).tolist()
        frame_ids.extend(int(x) for x in picked)

    frame_ids = sorted(int(x) for x in frame_ids)
    dbg["per_bin_counts"] = [{"bin": int(b), "count": int(c)} for (b, c) in per_bin_counts]
    dbg["selected_before_pad"] = int(len(frame_ids))

    if len(frame_ids) == 0:
        if total_frames == 1:
            frame_ids = [0] * target_count
        else:
            frame_ids = np.linspace(0, total_frames - 1, target_count, dtype=np.int32).tolist()

    if len(frame_ids) < target_count:
        if total_frames == 1:
            extra = [0] * (target_count - len(frame_ids))
        else:
            extra = np.linspace(0, total_frames - 1, target_count - len(frame_ids), dtype=np.int32).tolist()
        frame_ids = sorted([int(x) for x in frame_ids] + [int(x) for x in extra])

    if len(frame_ids) > target_count:
        idxs = np.linspace(0, len(frame_ids) - 1, target_count, dtype=np.int32).tolist()
        frame_ids = [int(frame_ids[int(i)]) for i in idxs]

    dbg["selected_after_pad"] = int(len(frame_ids))
    return [int(x) for x in frame_ids], dbg


def pick_windows_by_energy(
    video_path: str,
    total_frames: int,
    fps: float,
    window_len_frames: int,
    num_candidates: int,
    top_k: int,
) -> Tuple[List[Tuple[int, int, int]], Dict[str, Any]]:
    """Return list of selected windows: [(start_f, end_f, score_pkt_sum), ...]
    plus debug dict.
    """
    total_frames = int(total_frames)
    window_len_frames = int(window_len_frames)
    num_candidates = int(num_candidates)
    top_k = int(top_k)

    dbg: Dict[str, Any] = {
        "total_frames": total_frames,
        "fps": float(fps),
        "window_len_frames": window_len_frames,
        "num_candidates": num_candidates,
        "top_k": top_k,
        "candidates": [],
        "selected": [],
    }

    if total_frames <= 0:
        return [(0, 0, 0)], dbg

    if window_len_frames <= 0 or window_len_frames >= total_frames:
        # treat whole video as one window
        return [(0, total_frames - 1, 0)], dbg

    # candidate window starts evenly spaced in [0, total_frames - window_len]
    max_start = max(0, total_frames - window_len_frames)
    if num_candidates <= 1:
        starts = [0]
    else:
        starts = np.linspace(0, max_start, num_candidates, dtype=np.int32).tolist()

    fps_use = fps if fps and fps > 0 else 30.0
    dur_sec = float(window_len_frames) / float(fps_use)

    cand = []
    for st in starts:
        st = int(st)
        ed = int(min(total_frames - 1, st + window_len_frames - 1))
        start_sec = float(st) / float(fps_use)
        score = ffprobe_sum_pkt_size(video_path, start_sec, dur_sec)
        cand.append((st, ed, int(score)))
    cand_sorted = sorted(cand, key=lambda x: x[2], reverse=True)

    dbg["candidates"] = cand_sorted

    chosen = cand_sorted[: max(1, top_k)]
    dbg["selected"] = chosen
    return chosen, dbg


def allocate_frames_across_windows(
    windows: List[Tuple[int, int, int]],
    seq_len: int,
) -> List[int]:
    """Allocate seq_len frames across windows proportional to scores (fallback equal),
    sample uniformly within each window, then merge & pad/truncate to seq_len.
    """
    seq_len = int(seq_len)
    if seq_len <= 0:
        return []

    # if only one window
    if len(windows) == 1:
        st, ed, _ = windows[0]
        return np.linspace(st, ed, seq_len, dtype=np.int32).tolist()

    scores = np.array([max(0, w[2]) for w in windows], dtype=np.float64)
    if scores.sum() <= 0:
        # equal allocation
        weights = np.ones_like(scores) / float(len(scores))
    else:
        weights = scores / scores.sum()

    # initial per-window counts (at least 1)
    counts = np.maximum(1, np.floor(weights * seq_len).astype(int))
    # adjust to sum == seq_len
    while counts.sum() > seq_len:
        i = int(np.argmax(counts))
        if counts[i] > 1:
            counts[i] -= 1
        else:
            break
    while counts.sum() < seq_len:
        i = int(np.argmax(weights))
        counts[i] += 1

    frame_ids: List[int] = []
    for (st, ed, _), c in zip(windows, counts.tolist()):
        if c <= 0:
            continue
        frame_ids.extend(np.linspace(int(st), int(ed), int(c), dtype=np.int32).tolist())

    frame_ids = sorted([int(x) for x in frame_ids])

    # dedup but keep order (dedup can reduce length)
    dedup = []
    last = None
    for x in frame_ids:
        if last is None or x != last:
            dedup.append(x)
        last = x
    frame_ids = dedup

    # pad/truncate to seq_len
    if len(frame_ids) == 0:
        frame_ids = [0] * seq_len
    if len(frame_ids) < seq_len:
        frame_ids = (frame_ids + [frame_ids[-1]] * (seq_len - len(frame_ids)))[:seq_len]
    else:
        frame_ids = frame_ids[:seq_len]
    return frame_ids


def enforce_time_coverage_frame_ids(
    frame_ids: List[int],
    total_frames: int,
    fps: float,
    seq_len: int,
    stride_sec: float = 1.0,
) -> List[int]:
    """Best-effort ensure at least one sampled frame per time-bin.

    We create mandatory frame ids at roughly the center of each `stride_sec` bin,
    merge with existing `frame_ids`, then truncate/pad to `seq_len`.

    Note: If duration_bins > seq_len, it is impossible to cover all bins. In that
    case we cover as many earliest bins as possible.
    """
    seq_len = int(seq_len)
    if seq_len <= 0:
        return []

    fps_use = float(fps) if (fps and fps > 0) else 30.0
    total_frames = int(max(0, total_frames))
    if total_frames <= 0:
        return frame_ids[:seq_len]

    stride = float(stride_sec) if (stride_sec and stride_sec > 0) else 1.0
    duration_sec = float(total_frames) / float(fps_use)
    n_bins = int(math.floor(duration_sec / stride + 1e-9))
    n_bins = max(1, n_bins)

    # If bins exceed seq_len, we can only cover the first `seq_len` bins.
    n_bins_cover = min(int(n_bins), int(seq_len))

    mandatory: List[int] = []
    for b in range(n_bins_cover):
        t_center = (float(b) + 0.5) * stride
        fid = int(round(t_center * fps_use))
        if total_frames > 0:
            fid = max(0, min(total_frames - 1, fid))
        mandatory.append(fid)

    merged = sorted(set(mandatory + list(frame_ids)))

    if len(merged) < seq_len:
        # pad with duplicates of the last frame
        merged = merged + [merged[-1]] * (seq_len - len(merged))
    elif len(merged) > seq_len:
        # truncate, but try to keep the mandatory ones
        kept = []
        mset = set(mandatory)
        for x in merged:
            if len(kept) >= seq_len:
                break
            kept.append(x)
        merged = kept

    return [int(x) for x in merged[:seq_len]]
