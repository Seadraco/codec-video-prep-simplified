# Changelog

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
