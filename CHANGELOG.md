# Changelog

## 0.2.5.post1

- Add the opt-in `diverse_mixed_simple` Block selector.
- Keep 75% of the non-Anchor budget on public bit-cost ranking.
- Allocate 25% to Anchor-relative novelty and edge ranking.
- Deduplicate adjacent same-position Blocks with `pooled4` or `full` descriptors.
- Backfill by bit-cost order to preserve the exact Canvas and token budget.
- Preserve the public selector as the default and leave Anchor behavior unchanged.
