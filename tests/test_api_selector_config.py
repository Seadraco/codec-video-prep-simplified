from __future__ import annotations

from codec_selector.core.config import PipelineResult
from codec_video_prep import api
from codec_video_prep.config import PreinferConfig


def test_environment_selects_research_mode(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return PipelineResult(
            out_dir=str(tmp_path),
            meta_path=str(tmp_path / "missing.json"),
            canvas_files=[],
            summary={"timing_sec": {}},
        )

    monkeypatch.setattr(api, "run_bitcost_readiness", fake_run)
    monkeypatch.setenv("CODEC_SELECTOR_MODE", "diverse_mixed_simple")
    monkeypatch.setenv("CODEC_DEDUP_DESCRIPTOR", "full")
    api.run_preinfer_config(PreinferConfig(video="input.mp4", out_dir=str(tmp_path)))

    config = captured["config"]
    assert config.selector_mode == "diverse_mixed_simple"
    assert config.dedup_descriptor == "full"


def test_explicit_values_work_without_environment(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return PipelineResult(
            out_dir=str(tmp_path),
            meta_path=str(tmp_path / "missing.json"),
            canvas_files=[],
            summary={"timing_sec": {}},
        )

    monkeypatch.setattr(api, "run_bitcost_readiness", fake_run)
    monkeypatch.delenv("CODEC_SELECTOR_MODE", raising=False)
    monkeypatch.delenv("CODEC_DEDUP_DESCRIPTOR", raising=False)
    api.run_preinfer_config(
        PreinferConfig(
            video="input.mp4",
            out_dir=str(tmp_path),
            selector_mode="diverse_mixed_simple",
            dedup_descriptor="pooled4",
        )
    )

    config = captured["config"]
    assert config.selector_mode == "diverse_mixed_simple"
    assert config.dedup_descriptor == "pooled4"
