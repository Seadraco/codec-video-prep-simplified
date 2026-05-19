"""LLaVA-OneVision helpers for compressed-video preprocessing.

This module contains only the model-agnostic preprocessing and input assembly
pieces. Evaluation wrappers can import it instead of copying codec logic into
the model implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from compressed_video_preinfer import run_preinfer

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class LlavaCodecPreprocessConfig:
    cache_root: str
    target_canvas: int = 0
    group_size: int = 32
    images_per_group: int = 4
    patch: int = 14
    max_pixels: int = 150000
    min_group_frames: int = 8
    max_group_frames: int = 64
    bitcost_grid: str = "adaptive"
    grouping_mode: str = "readiness"
    frame_sampling_mode: str = "uniform_count"
    sample_fps: float = 4.0
    readiness_sum_threshold: float = 0.0
    readiness_sum_threshold_mode: str = "legacy"
    readiness_norm_sum_threshold: float = 2250000.0
    avoid_keyframes: bool = True
    parallel_decode_cv_reader: bool = False
    canvas_format: str = "jpg"

    @classmethod
    def from_legacy_kwargs(
        cls,
        cache_root: str,
        codec_target_canvas: int | str = 0,
        codec_group_size: int | str = 32,
        codec_images_per_group: int | str = 4,
        codec_patch: int | str = 14,
        codec_max_pixels: int | str = 150000,
        codec_min_group_frames: int | str = 8,
        codec_max_group_frames: int | str = 64,
        **kwargs: Any,
    ) -> "LlavaCodecPreprocessConfig":
        return cls(
            cache_root=str(cache_root),
            target_canvas=int(codec_target_canvas or 0),
            group_size=int(codec_group_size),
            images_per_group=int(codec_images_per_group),
            patch=int(codec_patch),
            max_pixels=int(codec_max_pixels),
            min_group_frames=int(codec_min_group_frames),
            max_group_frames=int(codec_max_group_frames),
            bitcost_grid=str(kwargs.get("codec_bitcost_grid", "adaptive")),
            grouping_mode=str(kwargs.get("codec_grouping_mode", "readiness")),
            frame_sampling_mode=str(kwargs.get("codec_frame_sampling_mode", "uniform_count")),
            sample_fps=float(kwargs.get("codec_sample_fps", 4.0)),
            readiness_sum_threshold=float(kwargs.get("codec_readiness_sum_threshold", 0.0)),
            readiness_sum_threshold_mode=str(kwargs.get("codec_readiness_sum_threshold_mode", "legacy")),
            readiness_norm_sum_threshold=float(kwargs.get("codec_readiness_norm_sum_threshold", 2250000.0)),
            avoid_keyframes=_as_bool(kwargs.get("codec_avoid_keyframes", True)),
            parallel_decode_cv_reader=_as_bool(kwargs.get("codec_parallel_decode_cv_reader", False)),
            canvas_format=str(kwargs.get("codec_canvas_format", "jpg")),
        )

    def validate(self) -> None:
        if int(self.target_canvas) <= 0:
            raise ValueError("target_canvas must be > 0")
        if int(self.images_per_group) <= 0:
            raise ValueError("images_per_group must be > 0")
        if int(self.group_size) <= 0:
            raise ValueError("group_size must be > 0")
        if int(self.target_canvas) % int(self.images_per_group) != 0:
            raise ValueError("target_canvas must be divisible by images_per_group")
        if int(self.group_size) % int(self.images_per_group) != 0:
            raise ValueError("group_size must be divisible by images_per_group")

    def num_sampled_frames(self, total_frames: int | None = None) -> int:
        target_groups = int(self.target_canvas) // int(self.images_per_group)
        target_frames = target_groups * int(self.group_size)
        if total_frames is None:
            return int(target_frames)
        return int(min(target_frames, max(1, int(total_frames))))


@dataclass
class LlavaCodecResult:
    images: list[Image.Image]
    src_positions: np.ndarray
    fps: float
    out_dir: str
    meta: dict[str, Any]


class LlavaOneVisionCodecPreprocessor:
    """Run compressed-video preprocessing with cache/lock handling."""

    def __init__(self, config: LlavaCodecPreprocessConfig):
        config.validate()
        self.config = config
        self.cache_root = Path(config.cache_root)

    def cache_dir_for(self, video: str, total_frames: int | None = None) -> Path:
        cfg = self.config
        raw = (
            f"{video}|tc={cfg.target_canvas}|gs={cfg.group_size}"
            f"|ipg={cfg.images_per_group}|patch={cfg.patch}"
            f"|mp={cfg.max_pixels}|grid={cfg.bitcost_grid}"
            f"|mode={cfg.grouping_mode}|fmt={cfg.canvas_format}"
            f"|n={cfg.num_sampled_frames(total_frames)}"
        )
        key = hashlib.md5(raw.encode()).hexdigest()
        return self.cache_root / f"{Path(video).stem}_{key}"

    def run(self, video: str, total_frames: int | None = None) -> LlavaCodecResult:
        out_dir = self.cache_dir_for(video, total_frames=total_frames)
        meta_path = out_dir / "meta.json"
        src_pos_path = out_dir / "src_patch_position.npy"
        if meta_path.exists() and src_pos_path.exists():
            return load_llava_codec_result(out_dir)

        self.cache_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_root / f".{out_dir.name}.lock"
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if meta_path.exists() and src_pos_path.exists():
                return load_llava_codec_result(out_dir)
            return self._run_locked(video, out_dir, total_frames=total_frames)
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)

    def _run_locked(
        self,
        video: str,
        out_dir: Path,
        total_frames: int | None = None,
    ) -> LlavaCodecResult:
        cfg = self.config
        tmp_dir = Path(tempfile.mkdtemp(dir=str(self.cache_root), prefix=f".tmp_{out_dir.name[:48]}_"))
        try:
            run_preinfer(
                video=str(video),
                out_dir=str(tmp_dir),
                num_sampled_frames=cfg.num_sampled_frames(total_frames),
                group_size=int(cfg.group_size),
                images_per_group=int(cfg.images_per_group),
                patch=int(cfg.patch),
                max_pixels=int(cfg.max_pixels),
                min_group_frames=int(cfg.min_group_frames),
                max_group_frames=int(cfg.max_group_frames),
                bitcost_grid=str(cfg.bitcost_grid),
                canvas_format=str(cfg.canvas_format),
                grouping_mode=str(cfg.grouping_mode),
                frame_sampling_mode=str(cfg.frame_sampling_mode),
                sample_fps=float(cfg.sample_fps),
                readiness_sum_threshold=float(cfg.readiness_sum_threshold),
                readiness_sum_threshold_mode=str(cfg.readiness_sum_threshold_mode),
                readiness_norm_sum_threshold=float(cfg.readiness_norm_sum_threshold),
                avoid_keyframes=bool(cfg.avoid_keyframes),
                parallel_decode_cv_reader=bool(cfg.parallel_decode_cv_reader),
            )
            if out_dir.exists():
                shutil.rmtree(out_dir)
            tmp_dir.rename(out_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return load_llava_codec_result(out_dir)


def load_llava_codec_result(out_dir: str | Path) -> LlavaCodecResult:
    out_path = Path(out_dir)
    with open(out_path / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    canvas_files = meta.get("canvas_files")
    if not canvas_files:
        for ext in ("npy", "jpg", "png"):
            hits = sorted(p.name for p in out_path.glob(f"canvas_*.{ext}"))
            if hits:
                canvas_files = hits
                break
        canvas_files = canvas_files or []
    images: list[Image.Image] = []
    for name in canvas_files:
        fp = out_path / name
        if str(name).endswith(".npy"):
            images.append(Image.fromarray(np.load(fp)))
        else:
            images.append(Image.open(fp).convert("RGB"))
    src_positions = np.load(out_path / "src_patch_position.npy")
    return LlavaCodecResult(
        images=images,
        src_positions=src_positions,
        fps=float(meta.get("fps") or 30.0),
        out_dir=str(out_path),
        meta=meta,
    )


def drop_padding_canvases(
    images: list[Image.Image],
    src_positions: np.ndarray,
) -> tuple[list[Image.Image], np.ndarray, int]:
    n_canvas = len(images)
    if n_canvas == 0:
        return images, src_positions, 0
    total_patches = int(src_positions.shape[0])
    if total_patches % n_canvas != 0:
        raise ValueError(
            f"src_positions length {total_patches} not divisible by canvas count {n_canvas}"
        )
    ppc = total_patches // n_canvas
    positions = src_positions.reshape(n_canvas, ppc, 3)
    canvas_t = positions[..., 0]
    keep_mask = (canvas_t >= 0).any(axis=1)
    bad = keep_mask & ~((canvas_t >= 0).all(axis=1))
    if bool(bad.any()):
        raise ValueError("encountered half-padding canvas; expected canvas-granular padding")
    dropped = int(n_canvas - int(keep_mask.sum()))
    if dropped == 0:
        return images, src_positions, 0
    kept_images = [img for img, keep in zip(images, keep_mask.tolist()) if keep]
    kept_positions = positions[keep_mask].reshape(-1, 3)
    return kept_images, kept_positions, dropped


def convert_positions_to_block_layout(
    positions: Any,
    t: int,
    h: int,
    w: int,
    spatial_merge_size: int = 2,
) -> Any:
    import torch

    sms = int(spatial_merge_size)
    if sms == 1:
        return positions
    total = int(t) * int(h) * int(w)
    indices = torch.arange(total, device=positions.device).view(t, h, w)
    h_m = int(h) // sms
    w_m = int(w) // sms
    indices = (
        indices.view(t, h_m, sms, w_m, sms)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(total)
    )
    return positions[indices]


def codec_positions_for_processor(
    src_positions: np.ndarray,
    image_grid_thw: Any,
    device: Any,
    spatial_merge_size: int = 2,
) -> Any:
    import torch

    positions = torch.from_numpy(src_positions).long().to(device)
    chunks: list[Any] = []
    offset = 0
    for row in image_grid_thw:
        t = int(row[0].item())
        h = int(row[1].item())
        w = int(row[2].item())
        n = t * h * w
        chunk = positions[offset : offset + n]
        chunks.append(convert_positions_to_block_layout(chunk, t, h, w, spatial_merge_size))
        offset += n
    if offset != int(positions.shape[0]):
        raise ValueError(
            "codec patch position length mismatch: "
            f"used={offset}, available={positions.shape[0]}"
        )
    return torch.cat(chunks, dim=0)


def format_timestamp(seconds: float, decimals: int) -> str:
    return f"<{seconds:.{decimals}f} seconds>"


def timestamp_runs(
    patch_positions: Any,
    fps: float,
    decimals: int,
    spatial_merge_size: int = 2,
) -> list[tuple[str, int]]:
    t_values = patch_positions[:, 0]
    unique_t, counts = t_values.unique_consecutive(return_counts=True)
    merge_factor = int(spatial_merge_size) ** 2
    runs: list[tuple[str, int]] = []
    for t_val, count in zip(unique_t.tolist(), counts.tolist()):
        if int(t_val) < 0:
            continue
        token_count = int(count) // merge_factor
        if token_count <= 0:
            continue
        runs.append((format_timestamp(float(t_val) / float(fps), decimals), token_count))
    return runs


def rewrite_text_with_codec_positions(
    text: str,
    patch_positions: Any,
    fps: float,
    decimals: int,
    spatial_merge_size: int = 2,
) -> str:
    parts: list[str] = []
    for timestamp, token_count in timestamp_runs(
        patch_positions,
        fps=fps,
        decimals=decimals,
        spatial_merge_size=spatial_merge_size,
    ):
        parts.append(timestamp)
        parts.append(VISION_START)
        parts.append(IMAGE_PAD * token_count)
        parts.append(VISION_END)
        parts.append("\n")
    vision_text = "".join(parts)

    first_vs = text.find(VISION_START)
    last_ve = text.rfind(VISION_END)
    if first_vs == -1 or last_ve == -1:
        return text
    tail_start = last_ve + len(VISION_END)
    if tail_start < len(text) and text[tail_start] == "\n":
        tail_start += 1
    return text[:first_vs] + vision_text + text[tail_start:]


def prepare_llava_onevision_inputs(
    processor: Any,
    text: str,
    codec_result: LlavaCodecResult | dict[str, Any],
    task: str | None = None,
    timestamp_decimals: int | None = None,
    max_pixels: int = 150000,
    spatial_merge_size: int = 2,
) -> dict[str, Any]:
    """Build processor/model inputs from a compressed-video preprocessing result."""
    if isinstance(codec_result, dict):
        images = list(codec_result["images"])
        src_positions = codec_result["src_positions"]
        fps = float(codec_result.get("fps") or 30.0)
    else:
        images = list(codec_result.images)
        src_positions = codec_result.src_positions
        fps = float(codec_result.fps)

    images, src_positions, _ = drop_padding_canvases(images, src_positions)
    if not images:
        raise RuntimeError("all codec canvases are padding; nothing to feed the model")

    proc_max_pixels = max(
        int(max_pixels),
        max((im.width * im.height for im in images), default=int(max_pixels)),
    )
    image_data = processor.image_processor(
        images=images,
        max_pixels=proc_max_pixels,
        return_tensors="pt",
    )
    image_grid_thw = image_data["image_grid_thw"]
    patch_positions = codec_positions_for_processor(
        src_positions,
        image_grid_thw,
        device=image_grid_thw.device,
        spatial_merge_size=spatial_merge_size,
    )
    decimals = 1 if timestamp_decimals is None else int(timestamp_decimals)
    rewritten_text = rewrite_text_with_codec_positions(
        text,
        patch_positions,
        fps=fps,
        decimals=decimals,
        spatial_merge_size=spatial_merge_size,
    )
    tokenized = processor.tokenizer(rewritten_text, return_tensors="pt")
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "pixel_values": image_data["pixel_values"],
        "image_grid_thw": image_grid_thw,
        "patch_positions": patch_positions,
    }
