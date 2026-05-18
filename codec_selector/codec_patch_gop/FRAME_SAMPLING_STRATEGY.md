# Frame Sampling Strategy

This document describes the current frame sampling strategy used by `tool/codec_patch_gop`.

## Goal

The sampler is designed to:

- allocate more candidate frames to information-dense regions
- keep a minimum amount of temporal coverage
- adapt bucket/GOP count for long videos
- separate cheap analysis-stage sampling from expensive RGB decode when possible
- reduce over-concentration on a few high-score frames during patch selection

## High-Level Pipeline

The current pipeline is:

1. probe video metadata
2. estimate output token grid and patch budget
3. choose GOP count
4. choose analysis candidate frame count
5. sample candidate frame ids on a packet-size-energy-reweighted timeline
6. inject extra peak-centered frame ids
7. optionally enforce coarse time coverage
8. build variable-length GOPs from packet energy and map them onto sampled frames
9. score candidate frames
10. choose patch blocks inside each GOP bucket with per-frame diversity penalty
11. decode only needed RGB frames for final patch extraction

## 1. Metadata And Budget Estimation

The sampler first reads:

- `total_frames`
- `fps`
- original height and width

Then it estimates the resized and padded frame shape using `smart_resize()` and patch-aligned padding. This gives an estimated full-canvas token count:

- `S_full_est = hb_est * wb_est`

This estimate is used to derive the output image budget before any real frame decode.

If `max_total_patches <= 0`, the system uses `auto_max_total_patches()` to choose an automatic token budget from:

- resolution tier
- video duration
- hard cap `max_total_patches_cap`

## 2. GOP Count

The current default is adaptive GOP count.

If `auto_num_gops=True` or `num_gops<=0`, then:

- `num_gops_auto = clamp(round(duration_sec / 8.0), 1, 16)`

This means:

- short videos usually keep 1 to 2 GOP buckets
- longer videos get more buckets
- very long videos are capped at 16 buckets

This is important because long-video coverage is mainly improved by increasing bucket count, not only by increasing frame count.

## 3. Analysis Candidate Frame Count

The candidate frame pool is not fixed anymore.

The system first estimates:

- `num_images_est = max(budget / S_full_est, num_gops * 4)`

Then it derives analysis-stage candidate count:

- `seq_len_adapt = clamp(analysis_candidate_multiplier * num_images_est, 64, analysis_max_frames)`

Default values:

- `analysis_candidate_multiplier = 8.0`
- `analysis_max_frames = 1024`

So for long videos, analysis can consider many more frames than before.

## 4. Energy-CDF Timeline Sampling

Candidate frames are sampled with `sample_frame_ids_by_energy_cdf()`.

The energy source is:

- PB packet size only
- keyframes are excluded
- energy is aggregated in time bins

Current defaults:

- `bin_sec = 0.5`
- `smooth_bins = 1`
- `uniform_mix = 0.15`
- `max_per_bin = 16`

The logic is:

- high-energy bins get more sampled frames
- low-energy bins get fewer sampled frames
- a small uniform prior avoids starving static but important regions
- a per-bin cap avoids one bursty region taking everything

This effectively constructs a non-uniform timeline for candidate-frame sampling.

## 5. Peak Injection

After the energy-CDF sampling step, the sampler injects a small set of extra frames around local packet-energy peaks using `pick_peak_frame_ids_from_pkt_size()`.

Current defaults:

- `pkt_bin_sec = 0.5`
- `pkt_peaks_cap = 8`
- `pkt_peaks_per_sec = 0.5`
- `pkt_peak_neighbor = 1`

This is meant to reduce the chance of missing short bursts that may be underrepresented by the CDF allocation alone.

## 6. Candidate Cap

After peak injection, candidate frame ids are hard-capped:

- `MAX_CAND = analysis_max_frames`

If the candidate set exceeds the cap:

- peak frames are kept preferentially
- the remaining frames are downsampled evenly

This keeps runtime stable while preserving burst-sensitive samples.

## 7. Optional Time Coverage

If `ensure_per_second=True`, the sampler applies `enforce_time_coverage_frame_ids()`.

This adds mandatory frame ids at roughly one frame per `sec_stride` interval, then truncates or pads back to `seq_len`.

This is a safety mechanism for coarse temporal coverage when energy-based sampling becomes too concentrated.

## 8. Variable-Length GOP Construction

GOP buckets are built by `build_variable_length_gops_by_energy()`.

The algorithm:

- computes PB-only packet-size energy bins
- accumulates energy over time
- cuts a GOP when cumulative energy reaches its target threshold
- also enforces span constraints

Current defaults:

- `bin_sec = 0.5`
- `smooth_bins = 1`
- `min_span_sec = 1.5`
- `max_span_sec = 6.0`

Each GOP segment stores:

- start time
- end time
- anchor time
- segment energy

Then each GOP anchor time is mapped to the nearest sampled frame index.

Depending on `gop_anchor_mode`, anchors may then be:

- left as sampled anchors
- replaced by keyframes
- aligned to nearby keyframes in hybrid mode

## 9. Analysis Vs Decode

There are now two practical modes.

### `score_source=mvres`

This path still decodes all sampled candidate frames first, because MV/residual scoring and bad-frame filtering are tied to those decoded frames.

### `score_source=bitcost`

This path now separates cheap analysis from expensive decode:

- many candidate frames are analyzed with bitcost maps first
- patch blocks are selected from those scores
- only anchor frames and actually selected P-block source frames are decoded to RGB

This is the main change that makes long-video sampling scale better.

## 10. Bucket-Inside Block Selection

Inside each GOP bucket:

- anchor frame contributes one full I canvas
- non-anchor frames provide candidate P blocks

For each non-anchor frame:

1. convert score map to patch scores
2. convert patch scores to 2x2 block scores
3. sort blocks by score

Then the system applies a per-frame diversity penalty during candidate construction:

- `penalty = 1 / sqrt(1 + block_selection_frame_penalty * used_n)`

where:

- `used_n` is how many high-ranked blocks from that frame have already been emitted into the candidate list
- `block_selection_frame_penalty` defaults to `0.35`

Effect:

- a strong frame can still contribute many blocks
- but the marginal gain decreases
- long videos are less likely to collapse onto a few extreme frames

## 11. Bad-Frame Handling

Decoded frames used for final extraction are filtered for:

- black frames
- near-solid-color frames
- corrupted green frames

In `bitcost` mode, this check runs only on the selected frames that are actually decoded for final output.

If an anchor frame is bad, the system tries to swap to a good selected frame within the same bucket.

## Important Parameters

The most important knobs for current sampling behavior are:

- `num_gops`
- `auto_num_gops`
- `analysis_candidate_multiplier`
- `analysis_max_frames`
- `ensure_per_second`
- `sec_stride`
- `gop_anchor_mode`
- `score_source`
- `block_selection_frame_penalty`

## Practical Summary

The current strategy is best understood as:

- use packet-size energy to decide where to look
- use adaptive GOP buckets to decide how to cover long videos
- use a larger cheap analysis pool for long videos
- only decode RGB frames that are truly needed
- apply a diversity penalty so selected patches are spread across more frames

That is the current behavior implemented in `video_processor.py`, `energy_sampling.py`, and `video_probe.py`.
