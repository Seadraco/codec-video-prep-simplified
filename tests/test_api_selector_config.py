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
    monkeypatch.setenv("CODEC_DIVERSITY_FRACTION", "0.4")
    monkeypatch.setenv("CODEC_NOVELTY_WEIGHT", "1.0")
    monkeypatch.setenv("CODEC_DEDUP_ENABLED", "0")
    monkeypatch.setenv("CODEC_DEDUP_DESCRIPTOR", "full")
    monkeypatch.setenv("CODEC_DEDUP_THRESHOLD", "0.05")
    api.run_preinfer_config(PreinferConfig(video="input.mp4", out_dir=str(tmp_path)))

    config = captured["config"]
    assert config.selector_mode == "diverse_mixed_simple"
    assert config.diversity_fraction == 0.4
    assert config.novelty_weight == 1.0
    assert config.dedup_enabled is False
    assert config.dedup_descriptor == "full"
    assert config.dedup_threshold == 0.05


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
    monkeypatch.delenv("CODEC_DIVERSITY_FRACTION", raising=False)
    monkeypatch.delenv("CODEC_NOVELTY_WEIGHT", raising=False)
    monkeypatch.delenv("CODEC_DEDUP_ENABLED", raising=False)
    monkeypatch.delenv("CODEC_DEDUP_DESCRIPTOR", raising=False)
    monkeypatch.delenv("CODEC_DEDUP_THRESHOLD", raising=False)
    api.run_preinfer_config(
        PreinferConfig(
            video="input.mp4",
            out_dir=str(tmp_path),
            selector_mode="diverse_mixed_simple",
            diversity_fraction=0.1,
            novelty_weight=0.0,
            dedup_enabled=True,
            dedup_descriptor="pooled4",
            dedup_threshold=0.015,
        )
    )

    config = captured["config"]
    assert config.selector_mode == "diverse_mixed_simple"
    assert config.diversity_fraction == 0.1
    assert config.novelty_weight == 0.0
    assert config.dedup_enabled is True
    assert config.dedup_descriptor == "pooled4"
    assert config.dedup_threshold == 0.015
