#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Virtual I+3P for Long Video - codec-aware patch selection pipeline.

This module provides a new experimental strategy for long video input construction:
- Divide video into virtual segments (e.g. 96 segments for 384 total images)
- Each segment: 1 full-frame anchor + 3 P-collages based on bitcost
"""

from .pipeline import process_video_virtual_i3p
from .anchor import select_anchor_frame
from .p_window import process_p_window_bitcost

__all__ = [
    "process_video_virtual_i3p",
    "select_anchor_frame",
    "process_p_window_bitcost",
]
