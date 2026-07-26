from __future__ import annotations

import numpy as np

from codec_selector.core.common_cache import (
    common_cache_key,
    load_common_cache,
    save_common_cache,
)


def test_common_cache_round_trip_without_pickle(tmp_path) -> None:
    payload = {
        "video": "/tmp/example.mp4",
        "frame_ids": [1, 9],
        "prepared_hw": [28, 28],
    }
    key = common_cache_key(payload)
    cache_dir = tmp_path / key[:2] / key
    frames = [
        np.full((28, 28, 3), 7, dtype=np.uint8),
        np.full((28, 28, 3), 19, dtype=np.uint8),
    ]
    bitcost = [
        {
            "frame_idx": 1,
            "pict_type": "P",
            "sub_mb_bit_cost": np.arange(16, dtype=np.int32).reshape(4, 4),
        },
        {
            "frame_idx": 9,
            "pict_type": "B",
            "sub_mb_bit_cost": np.arange(16, dtype=np.int32).reshape(4, 4) + 3,
        },
    ]

    save_common_cache(
        cache_dir,
        key_payload=payload,
        frame_ids=[1, 9],
        frames_bgr=frames,
        bitcost_items=bitcost,
    )
    loaded = load_common_cache(
        cache_dir,
        expected_key_payload=payload,
        expected_frame_ids=[1, 9],
    )
    assert loaded is not None
    loaded_frames, loaded_bitcost, meta = loaded
    assert meta["format_version"] == 1
    for expected, actual in zip(frames, loaded_frames):
        np.testing.assert_array_equal(actual, expected)
    for expected, actual in zip(bitcost, loaded_bitcost):
        assert actual["frame_idx"] == expected["frame_idx"]
        assert actual["pict_type"] == expected["pict_type"]
        np.testing.assert_array_equal(
            actual["sub_mb_bit_cost"],
            expected["sub_mb_bit_cost"],
        )


def test_common_cache_rejects_mismatched_payload(tmp_path) -> None:
    payload = {"video": "/tmp/example.mp4", "frame_ids": [1]}
    key = common_cache_key(payload)
    cache_dir = tmp_path / key
    save_common_cache(
        cache_dir,
        key_payload=payload,
        frame_ids=[1],
        frames_bgr=[np.zeros((28, 28, 3), dtype=np.uint8)],
        bitcost_items=[
            {
                "frame_idx": 1,
                "mb_bit_cost": np.zeros((2, 2), dtype=np.int32),
            }
        ],
    )
    assert (
        load_common_cache(
            cache_dir,
            expected_key_payload={**payload, "frame_ids": [2]},
            expected_frame_ids=[2],
        )
        is None
    )
