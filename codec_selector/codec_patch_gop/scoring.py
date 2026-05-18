#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scoring utilities for fusing motion and residual signals."""

from typing import Optional

import cv2
import numpy as np


def mv_res_score_map(
    mv: np.ndarray,
    res_y: np.ndarray,
    mv_energy: Optional[np.ndarray] = None,
    mv_unit_div: float = 4.0,
    mv_pct: float = 95.0,
    res_pct: float = 95.0,
    w_mv: float = 1.0,
    w_res: float = 1.0,
    mv_compensate: str = "none",
    mv_dir_mode: str = "l0",
    w_mv_l0: float = 1.0,
    w_mv_l1: float = 1.0,
) -> np.ndarray:
    """Produce fused score map in [0,1] float32 with shape (H,W)."""
    H, W = res_y.shape[:2]

    x = np.abs(res_y.astype(np.float32) - 128.0)
    a = float(np.percentile(x, res_pct))
    a = max(a, 1.0)
    res_norm = np.clip(x / a, 0.0, 1.0).astype(np.float32)

    mv_norm_u = np.zeros((H, W), dtype=np.float32)
    if mv_energy is not None:
        mag = np.asarray(mv_energy).astype(np.float32)
        if mag.ndim == 3:
            mag = np.squeeze(mag)
        if mag.ndim == 2 and mag.size > 0:
            mag = np.where(np.isnan(mag), 0.0, mag)
            valid_mag = mag[mag > 0] if np.any(mag > 0) else mag
            b = float(np.percentile(valid_mag, mv_pct))
            b = max(b, 1e-6)
            mv_norm = np.clip(mag / b, 0.0, 1.0)
            mv_norm_u = cv2.resize(mv_norm, (int(W), int(H)), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    elif mv.ndim == 3 and mv.shape[2] >= 2:
        vx0 = mv[:, :, 0].astype(np.float32) / float(mv_unit_div)
        vy0 = mv[:, :, 1].astype(np.float32) / float(mv_unit_div)

        has_l1 = mv.shape[2] >= 4
        if has_l1:
            vx1 = mv[:, :, 2].astype(np.float32) / float(mv_unit_div)
            vy1 = mv[:, :, 3].astype(np.float32) / float(mv_unit_div)
        else:
            vx1 = None
            vy1 = None

        if mv_compensate == "median":
            medx0 = float(np.median(vx0))
            medy0 = float(np.median(vy0))
            vx0 = vx0 - medx0
            vy0 = vy0 - medy0
            if has_l1 and vx1 is not None and vy1 is not None:
                medx1 = float(np.median(vx1))
                medy1 = float(np.median(vy1))
                vx1 = vx1 - medx1
                vy1 = vy1 - medy1

        mag0 = np.sqrt(vx0 * vx0 + vy0 * vy0).astype(np.float32)
        if has_l1 and vx1 is not None and vy1 is not None:
            mag1 = np.sqrt(vx1 * vx1 + vy1 * vy1).astype(np.float32)
            mode = str(mv_dir_mode).lower().strip()
            if mode == "l1":
                mag = mag1
            elif mode == "max":
                mag = np.maximum(mag0, mag1)
            elif mode == "sum":
                mag = mag0 + mag1
            elif mode == "biw":
                mag = float(w_mv_l0) * mag0 + float(w_mv_l1) * mag1
            else:
                mag = mag0
        else:
            mag = mag0

        b = float(np.percentile(mag, mv_pct))
        b = max(b, 1e-6)
        mv_norm = np.clip(mag / b, 0.0, 1.0)
        mv_norm_u = cv2.resize(mv_norm, (int(W), int(H)), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    denom = float(w_mv + w_res) if (w_mv + w_res) != 0 else 1.0
    fused = (float(w_mv) * mv_norm_u + float(w_res) * res_norm) / denom
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def score_map_to_patch_scores(fused_hw: np.ndarray, patch: int = 16) -> np.ndarray:
    """Convert (H,W) fused map to patch scores (hb,wb) by summing within each patch."""
    H, W = fused_hw.shape[:2]
    p = int(patch)
    assert H % p == 0 and W % p == 0
    hb, wb = H // p, W // p
    x = fused_hw.reshape(hb, p, wb, p).sum(axis=(1, 3))
    return x.astype(np.float32)


def patch_scores_to_block_scores(ps: np.ndarray, block_size: int = 2) -> np.ndarray:
    """Convert patch scores (hb,wb) into block scores (hb//b,wb//b)."""
    hb, wb = ps.shape[:2]
    b = int(max(1, int(block_size)))
    assert hb % b == 0 and wb % b == 0
    bs = ps.reshape(hb // b, b, wb // b, b).sum(axis=(1, 3))
    return bs.astype(np.float32)
