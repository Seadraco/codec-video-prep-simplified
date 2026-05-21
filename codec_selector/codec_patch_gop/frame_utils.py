#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frame processing utilities including resize, letterbox detection, and bad frame filtering."""

from typing import Tuple, Optional, List
import subprocess
import numpy as np
import cv2


def _resize_bgr(frame_bgr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize BGR frame with area when shrinking and linear when enlarging."""
    H, W = frame_bgr.shape[:2]
    if int(H) == int(out_h) and int(W) == int(out_w):
        return frame_bgr
    shrink = (out_h < H) or (out_w < W)
    interp = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    return cv2.resize(frame_bgr, (int(out_w), int(out_h)), interpolation=interp)


def _resize_gray(gray: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize single-channel image with area when shrinking and linear when enlarging."""
    H, W = gray.shape[:2]
    if int(H) == int(out_h) and int(W) == int(out_w):
        return gray
    shrink = (out_h < H) or (out_w < W)
    interp = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    return cv2.resize(gray, (int(out_w), int(out_h)), interpolation=interp)


def _resize_mv_and_scale(mv: np.ndarray, out_h: int, out_w: int, scale_h: float, scale_w: float) -> np.ndarray:
    """Resize MV field to (out_h,out_w) and scale vector components.

    Assumption (as per user): mv/res are pixel-aligned. When spatially resizing the
    image, motion in pixel units should be scaled by the same factors.

    Supports mv shape (...,2) or (...,4) where channels are (x,y) or (l0x,l0y,l1x,l1y).
    """
    if mv.ndim != 3 or mv.shape[2] < 2:
        return mv

    Hm, Wm = mv.shape[:2]
    if int(Hm) == int(out_h) and int(Wm) == int(out_w):
        mv_rs = mv.astype(np.float32, copy=False)
    else:
        # resize per-channel
        mv_f = mv.astype(np.float32, copy=False)
        ch = mv_f.shape[2]
        mv_rs = np.zeros((int(out_h), int(out_w), ch), dtype=np.float32)
        for c in range(ch):
            mv_rs[:, :, c] = cv2.resize(mv_f[:, :, c], (int(out_w), int(out_h)), interpolation=cv2.INTER_LINEAR)

    # scale vector components
    mv_rs[:, :, 0] *= float(scale_w)
    mv_rs[:, :, 1] *= float(scale_h)
    if mv_rs.shape[2] >= 4:
        mv_rs[:, :, 2] *= float(scale_w)
        mv_rs[:, :, 3] *= float(scale_h)

    return mv_rs


def detect_letterbox_bbox_bgr(frame_bgr: np.ndarray, dark_thr: float = 16.0) -> Tuple[int, int, int, int]:
    """Detect content bbox [top,bottom,left,right] by scanning dark borders."""
    if frame_bgr is None or frame_bgr.size == 0:
        return 0, 0, 0, 0

    h, w = frame_bgr.shape[:2]
    y = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)
    row_mean = y.mean(axis=1)
    col_mean = y.mean(axis=0)

    max_tb = int(h * 0.45)
    max_lr = int(w * 0.45)

    top = 0
    while top < min(h, max_tb) and float(row_mean[top]) < float(dark_thr):
        top += 1

    bottom = h
    while bottom - 1 >= max(0, h - max_tb) and float(row_mean[bottom - 1]) < float(dark_thr):
        bottom -= 1

    left = 0
    while left < min(w, max_lr) and float(col_mean[left]) < float(dark_thr):
        left += 1

    right = w
    while right - 1 >= max(0, w - max_lr) and float(col_mean[right - 1]) < float(dark_thr):
        right -= 1

    if top >= bottom or left >= right:
        return 0, h, 0, w
    return int(top), int(bottom), int(left), int(right)


def _bgr_to_luma_u8(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 frame to luma (Y) uint8."""
    if frame_bgr is None or frame_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if frame_bgr.ndim == 2:
        return frame_bgr.astype(np.uint8, copy=False)
    if frame_bgr.shape[2] == 1:
        return frame_bgr[:, :, 0].astype(np.uint8, copy=False)
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
    return yuv[:, :, 0].astype(np.uint8, copy=False)


def is_black_frame(
    frame_bgr: np.ndarray,
    y_mean_thr: float = 8.0,
    y_std_thr: float = 6.0,
) -> bool:
    """Detect near-black (contentless) frames.

    Heuristic:
      - mean(Y) very low AND std(Y) very low.

    This filters pure black fades/blank segments.
    """
    try:
        y = _bgr_to_luma_u8(frame_bgr)
        if y.size == 0:
            return True
        m = float(y.mean())
        s = float(y.std())
        return (m <= float(y_mean_thr)) and (s <= float(y_std_thr))
    except Exception:
        return False


def is_solid_color_frame(
    frame_bgr: np.ndarray,
    y_std_thr: float = 4.0,
    color_std_thr: float = 4.0,
    max_range_thr: float = 12.0,
) -> bool:
    """Detect nearly pure-color / flat frames.

    This is broader than black-frame detection and catches frames that are almost
    entirely one color, such as pure white / gray / green screens or flat fades.

    Heuristic:
      - luma std is very low
      - per-channel std is very low
      - dynamic range (max-min) of each channel is small
    """
    try:
        if frame_bgr is None or frame_bgr.size == 0:
            return True
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            y = _bgr_to_luma_u8(frame_bgr)
            if y.size == 0:
                return True
            return bool((float(y.std()) <= float(y_std_thr)) and ((float(y.max()) - float(y.min())) <= float(max_range_thr)))

        y = _bgr_to_luma_u8(frame_bgr)
        if y.size == 0:
            return True

        b = frame_bgr[:, :, 0].astype(np.float32, copy=False)
        g = frame_bgr[:, :, 1].astype(np.float32, copy=False)
        r = frame_bgr[:, :, 2].astype(np.float32, copy=False)

        y_std = float(y.std())
        b_std = float(b.std())
        g_std = float(g.std())
        r_std = float(r.std())

        b_rng = float(b.max() - b.min())
        g_rng = float(g.max() - g.min())
        r_rng = float(r.max() - r.min())

        return bool(
            (y_std <= float(y_std_thr))
            and (b_std <= float(color_std_thr))
            and (g_std <= float(color_std_thr))
            and (r_std <= float(color_std_thr))
            and (b_rng <= float(max_range_thr))
            and (g_rng <= float(max_range_thr))
            and (r_rng <= float(max_range_thr))
        )
    except Exception:
        return False


def is_corrupted_green_frame(
    frame_bgr: np.ndarray,
    green_frac_thr: float = 0.35,
    g_thr: int = 180,
    rb_thr: int = 90,
) -> bool:
    """Detect typical FFmpeg/OpenCV decode corruption that manifests as large green blocks.

    Heuristic:
      - Count pixels with very strong G while R/B are low.
      - If the fraction is high, treat frame as corrupted.

    This catches the common 'green macroblock' artifacts (like the screenshot).
    """
    try:
        if frame_bgr is None or frame_bgr.size == 0:
            return True
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            return False
        b = frame_bgr[:, :, 0].astype(np.uint8, copy=False)
        g = frame_bgr[:, :, 1].astype(np.uint8, copy=False)
        r = frame_bgr[:, :, 2].astype(np.uint8, copy=False)
        mask = (g >= int(g_thr)) & (r <= int(rb_thr)) & (b <= int(rb_thr))
        frac = float(mask.mean())
        if frac >= float(green_frac_thr):
            return True

        # Secondary check: overall green dominance with low texture (often corrupted fill)
        mg = float(g.mean()); mr = float(r.mean()); mb = float(b.mean())
        sg = float(g.std())
        if (mg > 120.0) and (mg - max(mr, mb) > 80.0) and (sg < 50.0):
            return True
        return False
    except Exception:
        return False


def frame_is_bad(
    frame_bgr: np.ndarray,
    skip_black_frames: bool = True,
    skip_corrupt_frames: bool = True,
    black_y_mean_thr: float = 8.0,
    black_y_std_thr: float = 6.0,
    solid_y_std_thr: float = 4.0,
    solid_color_std_thr: float = 4.0,
    solid_max_range_thr: float = 12.0,
    corrupt_green_frac_thr: float = 0.35,
    corrupt_g_thr: int = 180,
    corrupt_rb_thr: int = 90,
) -> bool:
    """Unified bad-frame predicate used to avoid selecting blank/corrupted segments.

    Optimized: compute luma once, reuse channels, avoid redundant astype copies.
    """
    try:
        if frame_bgr is None or frame_bgr.size == 0:
            return True

        # Compute luma once
        y = _bgr_to_luma_u8(frame_bgr)
        if y.size == 0:
            return True

        y_mean = float(y.mean())
        y_std = float(y.std())

        # Black frame check
        if skip_black_frames and (y_mean <= float(black_y_mean_thr)) and (y_std <= float(black_y_std_thr)):
            return True

        # Solid color check (re-use y_std)
        if frame_bgr.ndim == 3 and frame_bgr.shape[2] >= 3:
            b = frame_bgr[:, :, 0]
            g = frame_bgr[:, :, 1]
            r = frame_bgr[:, :, 2]
            b_std = float(b.std())
            g_std = float(g.std())
            r_std = float(r.std())
            b_rng = float(b.max() - b.min())
            g_rng = float(g.max() - g.min())
            r_rng = float(r.max() - r.min())
            if (
                y_std <= float(solid_y_std_thr)
                and b_std <= float(solid_color_std_thr)
                and g_std <= float(solid_color_std_thr)
                and r_std <= float(solid_color_std_thr)
                and b_rng <= float(solid_max_range_thr)
                and g_rng <= float(solid_max_range_thr)
                and r_rng <= float(solid_max_range_thr)
            ):
                return True
        else:
            if (y_std <= float(solid_y_std_thr)) and ((float(y.max()) - float(y.min())) <= float(solid_max_range_thr)):
                return True

        # Corrupted green check (re-use b, g, r)
        if skip_corrupt_frames and frame_bgr.ndim == 3 and frame_bgr.shape[2] >= 3:
            mask = (g >= int(corrupt_g_thr)) & (r <= int(corrupt_rb_thr)) & (b <= int(corrupt_rb_thr))
            if float(mask.mean()) >= float(corrupt_green_frac_thr):
                return True
            mg = float(g.mean())
            mr = float(r.mean())
            mb = float(b.mean())
            sg = float(g.std())
            if (mg > 120.0) and (mg - max(mr, mb) > 80.0) and (sg < 50.0):
                return True

        return False
    except Exception:
        return False


def pad_to_multiple_of_bgr(frame_bgr: np.ndarray, base: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad bottom/right with zeros so that H and W are multiples of `base`.

    For patch-based 2x2 blocks, you typically want base = 2 * patch.

    Returns padded frame and (pad_bottom, pad_right).
    """
    base = int(max(1, base))
    H, W = frame_bgr.shape[:2]
    pad_bottom = (base - (H % base)) % base
    pad_right = (base - (W % base)) % base
    if pad_bottom == 0 and pad_right == 0:
        return frame_bgr, (0, 0)
    out = cv2.copyMakeBorder(
        frame_bgr,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return out, (pad_bottom, pad_right)


def pad_to_multiple_of_32_bgr(frame_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad BGR frame so both dimensions are multiples of 32."""
    return pad_to_multiple_of_bgr(frame_bgr, 32)


def bgr_to_residual_y_u8(frame_bgr: np.ndarray) -> np.ndarray:
    """Fallback residual proxy when cv_reader residual is unavailable.

    IMPORTANT:
      - `mv_res_score_map()` expects a residual-like signal centered at 128, and uses
        `abs(res_y - 128)` as energy.
      - Using raw luma Y directly will incorrectly give high energy to pure black/white
        regions (|Y-128| is large), causing many black patches to be selected.

    This proxy uses Sobel gradient magnitude on luma as a texture/edge energy estimate,
    then maps it into uint8 residual space with 128 as the zero point:
      res_proxy = 128 + clip(grad_norm * 127)

    Flat regions (black/white) have low gradient, so they no longer dominate.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return np.full((0, 0), 128, dtype=np.uint8)

    # Luma
    y = _bgr_to_luma_u8(frame_bgr).astype(np.float32)
    if y.size == 0:
        return np.full((0, 0), 128, dtype=np.uint8)

    # Sobel gradients (float32)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)  # >=0

    # Robust normalization to [0,1] using percentile to avoid outliers
    scale = float(np.percentile(mag, 95.0))
    if (not np.isfinite(scale)) or scale <= 1e-6:
        scale = 1.0
    mag_n = np.clip(mag / scale, 0.0, 1.0)

    # Map to residual-like uint8 with 128 as center
    res_proxy = (128.0 + mag_n * 127.0).astype(np.uint8)
    return res_proxy


def decode_frame_bgr_at(video_path: str, frame_id: int, backsearch_max: int = 32) -> Optional[np.ndarray]:
    """Decode a single frame at given frame_id.
    
    If the exact frame fails, try nearby frames up to backsearch_max.
    
    Returns:
        BGR frame or None if decoding fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Try exact frame first
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        cap.release()
        return frame
    
    # Try nearby frames
    for offset in range(1, backsearch_max + 1):
        for sign in [1, -1]:
            try_id = frame_id + sign * offset
            if try_id < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, try_id)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                cap.release()
                return frame
    
    cap.release()
    return None


def decode_frames_bgr_opencv(video_path: str, frame_ids: List[int], backsearch_max: int = 32) -> List[np.ndarray]:
    """Decode multiple frames using a single VideoCapture (faster).

    Important:
      - When a stream is partially corrupt, OpenCV/FFmpeg may fail at some frame ids.
        Doing an unbounded decrement loop can become O(video_length) per failed frame
        and destroy throughput. We cap the backsearch and otherwise reuse last_good.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out: List[np.ndarray] = []
    last_good: Optional[np.ndarray] = None
    bs = int(max(0, backsearch_max))

    for fid0 in frame_ids:
        fid = int(fid0)
        if total > 0:
            fid = max(0, min(fid, total - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()

        if (not ok) or frame is None:
            # bounded backward search
            if bs > 0:
                dec = fid
                tries = 0
                frame = None
                while dec > 0 and tries < bs:
                    dec -= 1
                    tries += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, dec)
                    ok2, f2 = cap.read()
                    if ok2 and f2 is not None:
                        frame = f2
                        break

            # if still none, reuse last_good immediately (fast)
            if frame is None:
                if last_good is None:
                    cap.release()
                    raise RuntimeError(f"Failed to decode any frame around fid={fid0} for {video_path}")
                frame = last_good
        else:
            last_good = frame

        out.append(frame)

    cap.release()
    return out


def decode_frames_bgr_ffmpeg_subprocess(video_path: str, frame_ids: List[int]) -> List[np.ndarray]:
    """Decode selected frames using system ffmpeg subprocess and rawvideo pipe."""
    frame_ids = [int(x) for x in frame_ids]
    if not frame_ids:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video metadata: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size for {video_path}: {width}x{height}")

    uniq_ids = sorted(set(max(0, min(int(fid), total - 1)) if total > 0 else max(0, int(fid))
                          for fid in frame_ids))
    if not uniq_ids:
        return []

    select_expr = "+".join(f"eq(n\\,{fid})" for fid in uniq_ids)
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(video_path),
        "-vf", f"select={select_expr}",
        "-vsync", "0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg decode failed: {err.strip()}")

    frame_size = int(width) * int(height) * 3
    if frame_size <= 0:
        raise RuntimeError(f"Invalid frame_size for {video_path}")
    raw = proc.stdout
    num_decoded = len(raw) // frame_size
    if num_decoded <= 0:
        raise RuntimeError(f"ffmpeg returned no frames for {video_path}")
    if len(raw) % frame_size != 0:
        raw = raw[: num_decoded * frame_size]

    decoded_map = {}
    for i in range(min(num_decoded, len(uniq_ids))):
        start = i * frame_size
        end = start + frame_size
        arr = np.frombuffer(raw[start:end], dtype=np.uint8).copy().reshape(height, width, 3)
        decoded_map[int(uniq_ids[i])] = arr

    out: List[np.ndarray] = []
    last_good: Optional[np.ndarray] = None
    for fid in frame_ids:
        key = max(0, min(int(fid), total - 1)) if total > 0 else max(0, int(fid))
        fr = decoded_map.get(key)
        if fr is None:
            if last_good is None:
                # fallback: pick nearest decoded frame
                nearest = None
                if decoded_map:
                    nearest = decoded_map[min(decoded_map.keys(), key=lambda x: abs(x - key))]
                if nearest is None:
                    raise RuntimeError(f"missing decoded frame for fid={fid} in {video_path}")
                fr = nearest
            else:
                fr = last_good
        out.append(np.ascontiguousarray(fr))
        last_good = fr
    return out


def decode_frames_bgr(
    video_path: str,
    frame_ids: List[int],
    backsearch_max: int = 32,
    backend: str = "ffmpeg_native",
) -> List[np.ndarray]:
    """Decode multiple frames with the ffmpeg subprocess backend.

    ``backend`` is kept for compatibility with older callers. ``auto`` and
    ``ffmpeg_native`` both use ffmpeg. ``opencv`` is no longer a supported
    production path because it is slower for the selector workloads.
    """
    backend = str(backend).lower().strip()
    if backend == "opencv":
        raise ValueError("opencv decode backend has been removed; use ffmpeg_native")
    return decode_frames_bgr_ffmpeg_subprocess(video_path, frame_ids)
