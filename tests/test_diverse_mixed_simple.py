from __future__ import annotations

import numpy as np
import pytest

from codec_selector.core.config import BitcostReadinessConfig
from codec_selector.plugins.selectors.diverse_mixed_simple import (
    _adjacent_diff_full,
    _adjacent_diff_pooled4,
    _pooled4_descriptors,
    _rank01,
    process_group_diverse_mixed_simple,
)
from codec_selector.plugins.selectors.topk_2x2_bitcost import process_group_topk_2x2


def _frame(height: int = 56, width: int = 56, shift: int = 0) -> np.ndarray:
    y, x = np.indices((height, width))
    blue = (x * 3 + y + shift) % 256
    green = (x + y * 2 + shift * 3) % 256
    red = (x * 2 + y * 3 + shift * 5) % 256
    return np.stack([blue, green, red], axis=-1).astype(np.uint8)


def _selector_inputs() -> tuple[list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    frames = [_frame() for _ in range(4)]
    frame_ids = [10, 20, 30, 40]
    scores = [np.ones((56, 56), dtype=np.float32) for _ in frames]
    block_scores = [
        np.asarray([[1, 2], [3, 4]], dtype=np.float32),
        np.asarray([[20, 19], [18, 17]], dtype=np.float32),
        np.asarray([[16, 15], [14, 13]], dtype=np.float32),
        np.asarray([[12, 11], [10, 9]], dtype=np.float32),
    ]
    return frame_ids, frames, scores, block_scores


def test_rank01_preserves_ties() -> None:
    ranks = _rank01(np.asarray([1.0, 1.0, 3.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(ranks, [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0, 1.0])


def test_descriptor_modes_distinguish_native_detail() -> None:
    y, x = np.indices((28, 28))
    checker = ((x + y) % 2 * 255).astype(np.uint8)
    inverse = (255 - checker).astype(np.uint8)
    first = np.repeat(checker[..., None], 3, axis=-1)
    second = np.repeat(inverse[..., None], 3, axis=-1)
    frames = [first, second]
    pooled = _pooled4_descriptors(frames, block_rows=1, block_cols=1)
    pooled_diff = _adjacent_diff_pooled4(pooled)
    full_diff = _adjacent_diff_full(frames, block_rows=1, block_cols=1)
    assert float(pooled_diff[1, 0, 0]) < 0.025
    assert float(full_diff[1, 0, 0]) > 0.9


@pytest.mark.parametrize("descriptor", ["pooled4", "full"])
def test_selector_keeps_budget_and_metadata_aligned(descriptor: str) -> None:
    frame_ids, frames, scores, block_scores = _selector_inputs()
    meta, keep_mask, canvases, patch_pos, src_pos, img_ptr = process_group_diverse_mixed_simple(
        group_idx=0,
        group_frame_ids=list(frame_ids),
        group_frames_bgr=[frame.copy() for frame in frames],
        group_scores=[score.copy() for score in scores],
        images_per_group=3,
        patch=14,
        block_size=2,
        group_block_scores=[score.copy() for score in block_scores],
        good_mask=[True] * 4,
        dedup_descriptor=descriptor,
    )
    assert canvases.shape == (3, 56, 56, 3)
    assert patch_pos.shape == (48, 3)
    assert src_pos.shape == (48, 3)
    assert img_ptr.tolist() == [0, 16, 32, 48]
    assert keep_mask.shape == (4, 4, 4)
    assert int(keep_mask[0].sum()) == 16
    assert meta["selected_blocks"] == 8
    assert meta["selector"]["target_blocks"] == 8
    assert meta["selector"]["dedup_descriptor"] == descriptor
    assert meta["selector"]["backfill_selected"] > 0
    assert meta["selector"]["dedup_rejected"] > 0
    assert meta["selector"]["adjacent_mad_count"] > 0
    assert set(meta["selector"]["adjacent_mad_quantiles"]) == {
        "p05",
        "p10",
        "p20",
        "p50",
        "p80",
        "p90",
        "p95",
    }
    assert "0.025" in meta["selector"]["adjacent_mad_cdf"]
    assert 0.0 <= meta["selector"]["adjacent_mad_fraction_le_threshold"] <= 1.0

    frame_by_id = {frame_id: frame for frame_id, frame in zip(frame_ids, frames)}
    for source, destination in zip(src_pos, patch_pos):
        source_frame, patch_h, patch_w = source.tolist()
        if source_frame < 0:
            continue
        canvas_idx, canvas_h, canvas_w = destination.tolist()
        source_patch = frame_by_id[source_frame][
            patch_h * 14:(patch_h + 1) * 14,
            patch_w * 14:(patch_w + 1) * 14,
            ::-1,
        ]
        canvas_patch = canvases[
            canvas_idx,
            canvas_h * 14:(canvas_h + 1) * 14,
            canvas_w * 14:(canvas_w + 1) * 14,
        ]
        np.testing.assert_array_equal(source_patch, canvas_patch)

    valid_frame_ids = src_pos[src_pos[:, 0] >= 0, 0]
    assert np.all(valid_frame_ids[:16] == 10)
    assert np.all(np.diff(valid_frame_ids[16:]) >= 0)


def test_anchor_only_output_matches_public_selector() -> None:
    frame_ids, frames, scores, block_scores = _selector_inputs()
    common = {
        "group_idx": 0,
        "group_frame_ids": list(frame_ids),
        "group_frames_bgr": [frame.copy() for frame in frames],
        "group_scores": [score.copy() for score in scores],
        "images_per_group": 1,
        "patch": 14,
        "block_size": 2,
        "group_block_scores": [score.copy() for score in block_scores],
        "good_mask": [False, True, True, True],
        "anchor_idx": 0,
        "anchor_strategy": "fixed_anchor",
    }
    baseline = process_group_topk_2x2(**common)
    candidate = process_group_diverse_mixed_simple(
        **{
            **common,
            "group_frame_ids": list(frame_ids),
            "group_frames_bgr": [frame.copy() for frame in frames],
            "group_scores": [score.copy() for score in scores],
            "group_block_scores": [score.copy() for score in block_scores],
            "good_mask": [False, True, True, True],
        },
        dedup_descriptor="pooled4",
    )
    for baseline_value, candidate_value in zip(baseline[1:], candidate[1:]):
        np.testing.assert_array_equal(baseline_value, candidate_value)


def test_config_keeps_public_default_and_validates_new_mode() -> None:
    default = BitcostReadinessConfig(video="in.mp4", out_dir="out").normalized()
    assert default.selector_mode == "topk_2x2_bitcost"
    assert default.diversity_fraction == 0.10
    assert default.dedup_descriptor == "pooled4"

    configured = BitcostReadinessConfig(
        video="in.mp4",
        out_dir="out",
        selector_mode="DIVERSE_MIXED_SIMPLE",
        diversity_fraction=0.4,
        novelty_weight=1.0,
        dedup_enabled=False,
        dedup_descriptor="FULL",
        dedup_threshold=0.05,
    ).normalized()
    assert configured.selector_mode == "diverse_mixed_simple"
    assert configured.diversity_fraction == 0.4
    assert configured.novelty_weight == 1.0
    assert configured.dedup_enabled is False
    assert configured.dedup_descriptor == "full"
    assert configured.dedup_threshold == 0.05

    with pytest.raises(ValueError):
        BitcostReadinessConfig(
            video="in.mp4",
            out_dir="out",
            selector_mode="diverse_mixed_simple",
            event_aggregation=True,
        ).normalized()

    for name, value in (("diversity_fraction", -0.1), ("novelty_weight", 1.1)):
        with pytest.raises(ValueError):
            BitcostReadinessConfig(
                video="in.mp4",
                out_dir="out",
                selector_mode="diverse_mixed_simple",
                **{name: value},
            ).normalized()


@pytest.mark.parametrize(
    ("diversity_fraction", "dedup_enabled"),
    [(0.25, False), (0.0, True), (0.25, True)],
)
def test_ablation_modes_preserve_budget(
    diversity_fraction: float,
    dedup_enabled: bool,
) -> None:
    frame_ids, frames, scores, block_scores = _selector_inputs()
    meta, _, canvases, patch_pos, src_pos, _ = process_group_diverse_mixed_simple(
        group_idx=0,
        group_frame_ids=list(frame_ids),
        group_frames_bgr=[frame.copy() for frame in frames],
        group_scores=[score.copy() for score in scores],
        images_per_group=3,
        patch=14,
        block_size=2,
        group_block_scores=[score.copy() for score in block_scores],
        good_mask=[True] * 4,
        diversity_fraction=diversity_fraction,
        dedup_enabled=dedup_enabled,
    )
    selector = meta["selector"]
    assert canvases.shape == (3, 56, 56, 3)
    assert patch_pos.shape == src_pos.shape == (48, 3)
    assert selector["target_blocks"] == 8
    assert (
        selector["bitcost_selected"]
        + selector["diversity_selected"]
        + selector["backfill_selected"]
        == 8
    )
    if not dedup_enabled:
        assert selector["dedup_rejected"] == 0
        assert selector["backfill_selected"] == 0
    if diversity_fraction == 0.0:
        assert selector["diversity_selected"] == 0


def test_zero_diversity_without_dedup_matches_public() -> None:
    frame_ids, frames, scores, block_scores = _selector_inputs()
    common = {
        "group_idx": 0,
        "group_frame_ids": list(frame_ids),
        "group_frames_bgr": [frame.copy() for frame in frames],
        "group_scores": [score.copy() for score in scores],
        "images_per_group": 3,
        "patch": 14,
        "block_size": 2,
        "group_block_scores": [score.copy() for score in block_scores],
        "good_mask": [True] * 4,
    }
    baseline = process_group_topk_2x2(**common)
    candidate = process_group_diverse_mixed_simple(
        **{
            **common,
            "group_frame_ids": list(frame_ids),
            "group_frames_bgr": [frame.copy() for frame in frames],
            "group_scores": [score.copy() for score in scores],
            "group_block_scores": [score.copy() for score in block_scores],
            "good_mask": [True] * 4,
        },
        diversity_fraction=0.0,
        dedup_enabled=False,
    )
    for baseline_value, candidate_value in zip(baseline[1:], candidate[1:]):
        np.testing.assert_array_equal(baseline_value, candidate_value)


@pytest.mark.parametrize(
    ("descriptor", "expected"),
    [("pooled4", 0.025), ("full", 0.035)],
)
def test_default_and_explicit_dedup_thresholds(
    descriptor: str,
    expected: float,
) -> None:
    frame_ids, frames, scores, block_scores = _selector_inputs()
    common = {
        "group_idx": 0,
        "group_frame_ids": list(frame_ids),
        "group_frames_bgr": [frame.copy() for frame in frames],
        "group_scores": [score.copy() for score in scores],
        "images_per_group": 3,
        "patch": 14,
        "group_block_scores": [score.copy() for score in block_scores],
        "good_mask": [True] * 4,
        "dedup_descriptor": descriptor,
    }
    default = process_group_diverse_mixed_simple(**common)[0]["selector"]
    explicit = process_group_diverse_mixed_simple(
        **{
            **common,
            "group_frame_ids": list(frame_ids),
            "group_frames_bgr": [frame.copy() for frame in frames],
            "group_scores": [score.copy() for score in scores],
            "group_block_scores": [score.copy() for score in block_scores],
            "good_mask": [True] * 4,
        },
        dedup_threshold=0.123,
    )[0]["selector"]
    assert default["dedup_threshold"] == expected
    assert explicit["dedup_threshold"] == 0.123


def test_group_quantile_resolves_from_local_distribution() -> None:
    frame_ids, frames, scores, block_scores = _selector_inputs()
    frames = [
        _frame(shift=0),
        _frame(shift=2),
        _frame(shift=20),
        _frame(shift=80),
    ]
    selector = process_group_diverse_mixed_simple(
        group_idx=0,
        group_frame_ids=list(frame_ids),
        group_frames_bgr=[frame.copy() for frame in frames],
        group_scores=[score.copy() for score in scores],
        images_per_group=3,
        patch=14,
        block_size=2,
        group_block_scores=[score.copy() for score in block_scores],
        good_mask=[True] * 4,
        dedup_threshold_mode="group_quantile",
        dedup_quantile=0.20,
    )[0]["selector"]
    assert selector["dedup_threshold_mode"] == "group_quantile"
    assert selector["dedup_quantile"] == 0.20
    assert selector["dedup_threshold_fallback"] is False
    assert selector["dedup_threshold"] == pytest.approx(
        selector["adjacent_mad_quantiles"]["p20"]
    )
    assert selector["adjacent_mad_fraction_le_threshold"] == pytest.approx(
        0.20,
        abs=0.05,
    )
