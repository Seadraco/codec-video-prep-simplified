# Changelog

## 0.2.5.post4

- Add an adaptive `sample_stride` gate: use the exact public Top-K path for
  densely sampled videos and enable the research selector only when the median
  sampled-frame interval reaches a configurable threshold.
- Replace the dataset-sensitive absolute deduplication threshold with an
  optional per-group adjacent-MAD quantile.
- Set the opt-in research profile to 30% diversity, 15% group-quantile
  deduplication, and a 5-second activation threshold.
- Validate the activation threshold across 8/16/32/64 Canvas budgets; the
  stricter threshold preserves sparse-video gains while avoiding a Rapid c64
  regression.
- Record source FPS, sampled-frame interval, activation decision, and control
  reason in every group of `meta.json`.
- Include all adaptive-selector parameters in the bundled LLaVA cache key.
- Support `--version` on both the public CLI and the legacy compatibility
  alias so an installed wheel can be audited without running preprocessing.
- Preserve `topk_2x2_bitcost` as the package default and keep the disabled
  research path byte-identical to the public selector.

## 0.2.5.post3

- Set the opt-in research selector's validated default mix to 90% public
  bit-cost and 10% diversity.
- Keep the public `topk_2x2_bitcost` selector as the package default.
- Add exact paired McNemar statistics and offline-cache metadata discovery to
  the experiment summarizer.
- Make `codec-video-prep-doctor` validate the actual `slice` thread default.
- Document the RapidVideoQA-200 and TempCompass-MC staged ablations, including
  the non-significant uncertainty of the observed accuracy gain.

## 0.2.5.post2

- Expose the five simplified selector controls through Python, CLI, and
  environment variables.
- Support independent Diversity-only and dedup-only ablations.
- Add per-group and top-level Canvas selection diagnostics to `meta.json`.
- Include all selector controls in the bundled LLaVA integration cache key.
- Keep `topk_2x2_bitcost` as the unchanged default control path.

## 0.2.5.post1

- Add the opt-in `diverse_mixed_simple` Block selector.
- Keep 75% of the non-Anchor budget on public bit-cost ranking.
- Allocate 25% to Anchor-relative novelty and edge ranking.
- Deduplicate adjacent same-position Blocks with `pooled4` or `full` descriptors.
- Backfill by bit-cost order to preserve the exact Canvas and token budget.
- Preserve the public selector as the default and leave Anchor behavior unchanged.
- Build eight Linux wheels for Python 3.10–3.13 on x86_64 and aarch64.
- Use conditional NumPy dependencies so the Python 3.13 wheels install with
  NumPy 2.x while Python 3.10–3.12 retain the public NumPy 1.x dependency.
- Constrain Pillow below 12 so aarch64 installations use a published
  manylinux2014 wheel instead of attempting a source build.
- Provide the `codec-video-prep-legacy-exact` command alias expected by the
  official LLaVA-OneVision-2 `lmms-eval` wrapper.
