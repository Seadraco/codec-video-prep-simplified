#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collage output helpers."""

import subprocess
from pathlib import Path
from typing import List

import cv2
import numpy as np

from codec_selector.core.registry import packers


def save_canvases(
    images_rgb: np.ndarray,
    out_dir: str,
    start_idx: int = 0,
    fmt: str = "jpg",
    quality: int = 95,
) -> List[str]:
    out: List[str] = []
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if images_rgb is None or images_rgb.size == 0:
        return out

    fmt_use = str(fmt).lower().strip()
    if fmt_use not in {"jpg", "png"}:
        raise ValueError(f"unsupported canvas format: {fmt}")
    q = int(max(1, min(100, int(quality))))
    for i in range(int(images_rgb.shape[0])):
        fn = f"canvas_{int(start_idx) + i:03d}.{fmt_use}"
        fp = out_p / fn
        bgr = images_rgb[i][:, :, ::-1]
        if fmt_use == "png":
            ok = cv2.imwrite(str(fp), bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        else:
            ok = cv2.imwrite(str(fp), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError(f"failed to write image: {str(fp)}")
        out.append(fn)
    return out


def save_mask_video(path: Path, fps: float, frames_bgr: List[np.ndarray]) -> Path:
    if not frames_bgr:
        raise RuntimeError("save_mask_video received empty frames")

    h, w = frames_bgr[0].shape[:2]
    for fr in frames_bgr:
        if fr.shape[:2] != (h, w):
            raise RuntimeError("all frames must share the same shape for ffmpeg encoding")

    out_path = path.with_suffix(".mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{float(fps):.6f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for fr in frames_bgr:
            proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        if proc.stdout is not None:
            proc.stdout.read()
        proc.wait()
    except Exception:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        proc.kill()
        proc.wait()
        raise

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="ignore") if stderr is not None else ""
        raise RuntimeError(f"ffmpeg encode failed for {str(out_path)}: {err}")
    return out_path


packers.register("collage", save_canvases)
