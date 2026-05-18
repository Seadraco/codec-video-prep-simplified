#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for codec patch GOP processing."""

import os
import re
import json
import math
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Regex for rewriting image tags in user content
_IMAGE_PREFIX_RE = re.compile(r"^\s*(?:<image>\s*)+", flags=re.IGNORECASE)


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 768 * 768,
) -> Tuple[int, int]:
    """Resize rule copied from run_cut_frames.py.

    - Output H/W are multiples of `factor`.
    - If area > max_pixels: shrink with floor.
    - If area < min_pixels: enlarge with ceil.
    - Otherwise: round to nearest multiple of factor.

    Returns: (resized_h, resized_w)
    """
    h = int(height)
    w = int(width)
    f = int(max(1, factor))

    if h <= 0 or w <= 0:
        return 0, 0

    if max(h, w) / max(1, min(h, w)) > 200:
        raise ValueError(f"Extreme aspect ratio: h={h}, w={w}")

    # round to nearest multiple of factor
    h_bar = int(round(h / f) * f)
    w_bar = int(round(w / f) * f)

    area = float(h) * float(w)
    if area > float(max_pixels):
        beta = math.sqrt(area / float(max_pixels))
        h_bar = int(math.floor((h / beta) / f) * f)
        w_bar = int(math.floor((w / beta) / f) * f)
    elif area < float(min_pixels):
        beta = math.sqrt(float(min_pixels) / max(1.0, area))
        h_bar = int(math.ceil((h * beta) / f) * f)
        w_bar = int(math.ceil((w * beta) / f) * f)

    h_bar = max(f, int(h_bar))
    w_bar = max(f, int(w_bar))
    return int(h_bar), int(w_bar)


def sha1_8(s: str) -> str:
    """Return first 8 chars of SHA1 hex digest."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def format_timestamp_ss(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS string."""
    s = max(0.0, float(seconds))
    hh = int(s // 3600)
    mm = int((s % 3600) // 60)
    ss = int(s % 60)
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def rewrite_user_content_image_tags(content: str, num_images: int) -> str:
    """Replace leading <image> tags with <image>\n prefix for each image."""
    if not isinstance(content, str):
        return ""
    # Remove existing leading image tags
    cleaned = _IMAGE_PREFIX_RE.sub("", content)
    prefix = "<image>\n" * max(0, int(num_images))
    return prefix + cleaned


def ensure_dir(p: str) -> None:
    """Ensure directory exists."""
    Path(p).mkdir(parents=True, exist_ok=True)


def _clamp_int(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def _round_to_multiple(x: float, base: int) -> int:
    return int(round(float(x) / base) * base)


# -----------------------------
# JSONL utilities
# -----------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load all lines from jsonl file."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def iter_jsonl(path: str):
    """Iterate over jsonl file line by line (generator)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


# -----------------------------
# Path/Key utilities
# -----------------------------

def extract_video_path_from_item(it: Dict[str, Any]) -> Optional[str]:
    """Extract video path from jsonl item."""
    for k in ("video", "path", "video_path"):
        v = it.get(k)
        if isinstance(v, str) and v:
            return v
    # OpenAI-like result format
    custom = it.get("custom")
    if isinstance(custom, dict):
        for kk in ("video", "path", "video_path"):
            v = custom.get(kk)
            if isinstance(v, str) and v:
                return v
    return None


def infer_key_from_video(video_path: str, it: Dict[str, Any]) -> str:
    """Infer a stable key for output directory."""
    k = it.get("key")
    if isinstance(k, str) and k:
        return k

    stem = Path(video_path).stem
    vid = it.get("id")
    if isinstance(vid, str) and vid:
        return f"{stem}__{sha1_8(video_path)}__{vid[:8]}"

    return f"{stem}__{sha1_8(video_path)}"


def extract_caption_from_item(it: Dict[str, Any]) -> str:
    """Extract caption / assistant text from OpenAI-like results jsonl.
    Expected schema:
      it["response"]["body"]["choices"][0]["message"]["content"]
    """
    resp = it.get("response")
    if not isinstance(resp, dict):
        return ""
    body = resp.get("body")
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    ch0 = choices[0]
    if not isinstance(ch0, dict):
        return ""
    msg = ch0.get("message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    return content if isinstance(content, str) else ""


# -----------------------------
# Mirror key helper
# -----------------------------

def mirror_key_from_video(video_path: str, mirror_src_root: str, strip_ext: bool = True) -> Optional[str]:
    """Map an absolute video path to a relative directory key under `mirror_src_root`.

    Example:
      video_path=/ov2/dataset_mid_source/batch1_5millions/a/b/c.mp4
      mirror_src_root=/ov2/dataset_mid_source/batch1_5millions
      -> key=a/b/c  (if strip_ext)

    Returns None if video_path is not under mirror_src_root.
    """
    try:
        vp = Path(video_path)
        root = Path(mirror_src_root)
        vp_res = vp.resolve()
        root_res = root.resolve()
        try:
            rel = vp_res.relative_to(root_res)
        except Exception:
            vp_s = str(vp)
            root_s = str(mirror_src_root)
            if not vp_s.startswith(root_s.rstrip("/") + "/") and vp_s != root_s:
                return None
            rel = Path(os.path.relpath(vp_s, root_s))

        if strip_ext:
            return str(rel.with_suffix(""))
        return str(rel)
    except Exception:
        return None
