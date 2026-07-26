"""Selector-independent cache for decoded candidate frames and BitCost maps."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]


CACHE_FORMAT_VERSION = 1
BITCOST_ARRAY_KEYS = ("sub_mb_bit_cost", "mb_bit_cost", "ctu_bit_cost")


def common_cache_key(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def common_cache_lock(cache_root: Path, key: str) -> Iterator[None]:
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / f".{key}.lock"
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        if fcntl is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def load_common_cache(
    cache_dir: Path,
    expected_key_payload: Dict[str, Any],
    expected_frame_ids: List[int],
) -> Optional[Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]]:
    meta_path = cache_dir / "common_meta.json"
    frames_path = cache_dir / "frames.npy"
    complete_path = cache_dir / "_COMPLETE"
    if not (meta_path.exists() and frames_path.exists() and complete_path.exists()):
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("format_version", -1)) != CACHE_FORMAT_VERSION:
            return None
        if meta.get("key_payload") != expected_key_payload:
            return None
        frame_ids = [int(value) for value in meta.get("frame_ids", [])]
        if frame_ids != [int(value) for value in expected_frame_ids]:
            return None

        frames_array = np.load(frames_path, mmap_mode="r", allow_pickle=False)
        if frames_array.ndim != 4 or int(frames_array.shape[0]) != len(frame_ids):
            return None
        frames = [frames_array[index] for index in range(len(frame_ids))]

        item_meta = meta.get("bitcost_items")
        if not isinstance(item_meta, list) or len(item_meta) != len(frame_ids):
            return None
        bitcost_items: List[Dict[str, Any]] = [
            dict(value) if isinstance(value, dict) else {}
            for value in item_meta
        ]
        for key in BITCOST_ARRAY_KEYS:
            array_path = cache_dir / f"{key}.npy"
            present_path = cache_dir / f"{key}_present.npy"
            if not (array_path.exists() and present_path.exists()):
                continue
            arrays = np.load(array_path, mmap_mode="r", allow_pickle=False)
            present = np.load(present_path, allow_pickle=False)
            if int(arrays.shape[0]) != len(frame_ids) or present.shape != (len(frame_ids),):
                return None
            for index, is_present in enumerate(present.tolist()):
                if bool(is_present):
                    bitcost_items[index][key] = arrays[index]
        return frames, bitcost_items, meta
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def save_common_cache(
    cache_dir: Path,
    key_payload: Dict[str, Any],
    frame_ids: List[int],
    frames_bgr: List[np.ndarray],
    bitcost_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(frames_bgr) != len(frame_ids) or len(bitcost_items) != len(frame_ids):
        raise ValueError("common cache inputs must have matching lengths")

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(
            dir=str(cache_dir.parent),
            prefix=f".tmp_{cache_dir.name[:32]}_",
        )
    )
    try:
        frames_array = np.stack(frames_bgr, axis=0).astype(np.uint8, copy=False)
        np.save(tmp_dir / "frames.npy", frames_array, allow_pickle=False)

        scalar_items: List[Dict[str, Any]] = []
        for frame_id, item in zip(frame_ids, bitcost_items):
            scalar: Dict[str, Any] = {"frame_idx": int(item.get("frame_idx", frame_id))}
            if "pict_type" in item:
                value = item["pict_type"]
                scalar["pict_type"] = value.item() if isinstance(value, np.generic) else value
            scalar_items.append(scalar)

        stored_keys: List[str] = []
        for key in BITCOST_ARRAY_KEYS:
            present = np.asarray([key in item for item in bitcost_items], dtype=bool)
            if not bool(present.any()):
                continue
            first = np.asarray(
                next(item[key] for item in bitcost_items if key in item)
            )
            arrays = np.zeros((len(bitcost_items), *first.shape), dtype=first.dtype)
            for index, item in enumerate(bitcost_items):
                if key not in item:
                    continue
                value = np.asarray(item[key])
                if value.shape != first.shape:
                    raise ValueError(f"inconsistent {key} shape in common cache")
                arrays[index] = value
            np.save(tmp_dir / f"{key}.npy", arrays, allow_pickle=False)
            np.save(tmp_dir / f"{key}_present.npy", present, allow_pickle=False)
            stored_keys.append(key)

        meta = {
            "format_version": CACHE_FORMAT_VERSION,
            "key_payload": key_payload,
            "frame_ids": [int(value) for value in frame_ids],
            "frame_shape": list(frames_array.shape),
            "bitcost_array_keys": stored_keys,
            "bitcost_items": scalar_items,
        }
        (tmp_dir / "common_meta.json").write_text(
            json.dumps(meta, ensure_ascii=True, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        (tmp_dir / "_COMPLETE").touch()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        tmp_dir.rename(cache_dir)
        return meta
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
