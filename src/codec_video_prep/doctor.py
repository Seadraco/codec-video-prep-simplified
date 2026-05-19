"""Installation diagnostics for codec-video-prep."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from .config import PreinferConfig


def check() -> list[tuple[str, bool, str]]:
    rows: list[tuple[str, bool, str]] = []
    try:
        importlib.import_module("codec_video_prep.cv_reader_fast")
        rows.append(("cv_reader_fast import", True, "OK"))
    except Exception as exc:
        rows.append(("cv_reader_fast import", False, str(exc)))

    libs_dir = Path(__file__).resolve().parent / "libs"
    libs = list(libs_dir.glob("libav*.so*")) + list(libs_dir.glob("libsw*.so*"))
    rows.append(("bundled FFmpeg libs", bool(libs), f"{len(libs)} found" if libs else "missing"))

    cfg = PreinferConfig(video="", out_dir="")
    rows.append(("thread_type", cfg.thread_type == "auto", cfg.thread_type))
    rows.append(("thread_count", cfg.thread_count == 16, str(cfg.thread_count)))
    target_only = os.environ.get("CVR_DISABLE_TARGET_ONLY", "default disabled under frame threading")
    rows.append(("frame-thread target-only policy", True, target_only))
    return rows


def main() -> None:
    failed = False
    for name, ok, detail in check():
        failed = failed or not ok
        print(f"{name}: {'OK' if ok else 'FAIL'} ({detail})")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
