#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Video metadata helpers backed by ffprobe."""

import json
import subprocess
from fractions import Fraction
from typing import Any, Dict

from .config import VideoMetadata


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _parse_fps(value: Any) -> float:
    text = str(value or "").strip()
    if not text or text == "0/0":
        return 0.0
    try:
        return float(Fraction(text))
    except Exception:
        return _to_float(text)


def probe_video(video_path: str) -> VideoMetadata:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,bit_rate,avg_frame_rate,nb_frames:format=bit_rate,duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    data: Dict[str, Any] = json.loads(proc.stdout or "{}")
    streams = data.get("streams") if isinstance(data.get("streams"), list) else []
    stream = streams[0] if streams and isinstance(streams[0], dict) else {}
    fmt = data.get("format") if isinstance(data.get("format"), dict) else {}

    fps = _parse_fps(stream.get("avg_frame_rate"))
    duration = _to_float(fmt.get("duration"))
    total_frames = int(_to_float(stream.get("nb_frames")))
    if total_frames <= 0 and fps > 0 and duration > 0:
        total_frames = int(round(float(fps) * float(duration)))

    return VideoMetadata(
        video=str(video_path),
        total_frames=max(0, int(total_frames)),
        fps=max(0.0, float(fps)),
        height=int(stream.get("height", 0) or 0),
        width=int(stream.get("width", 0) or 0),
        codec_name=str(stream.get("codec_name", "") or "").lower(),
        stream_bit_rate=_to_float(stream.get("bit_rate")),
        format_bit_rate=_to_float(fmt.get("bit_rate")),
        duration=duration,
    )
