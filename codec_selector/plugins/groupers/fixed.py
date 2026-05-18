#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-size grouping plugin."""

from typing import List

from codec_selector.core.config import GroupSpec
from codec_selector.core.registry import groupers


def build_fixed_groups(frame_ids: List[int], group_size: int) -> List[GroupSpec]:
    groups: List[GroupSpec] = []
    step = max(1, int(group_size))
    for group_idx, start in enumerate(range(0, len(frame_ids), step)):
        end = min(len(frame_ids), start + step)
        groups.append(
            GroupSpec(
                group_idx=int(group_idx),
                start=int(start),
                end=int(end),
                frame_count=int(end - start),
                stop_reason="fixed_group_size",
                readiness={},
            )
        )
    return groups


groupers.register("fixed", build_fixed_groups)
