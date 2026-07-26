from __future__ import annotations

from compressed_video_preinfer.cli import build_parser as build_legacy_parser
from codec_selector.core.config import PipelineResult
from codec_video_prep import api
from codec_video_prep.cli import build_parser
from codec_video_prep.config import PreinferConfig


def test_cli_exposes_version_without_required_job_arguments(capsys) -> None:
    try:
        build_parser().parse_args(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("--version did not exit through argparse")
    assert capsys.readouterr().out.strip().endswith("0.2.5.post4")


def test_legacy_cli_alias_exposes_same_version(capsys) -> None:
    try:
        build_legacy_parser().parse_args(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("legacy --version did not exit through argparse")
    assert capsys.readouterr().out.strip().endswith("0.2.5.post4")


def test_research_defaults_use_validated_adaptive_profile() -> None:
    config = PreinferConfig(video="input.mp4", out_dir="output")
    assert config.selector_mode == "topk_2x2_bitcost"
    assert config.diversity_fraction == 0.30
    assert config.novelty_weight == 0.5
    assert config.dedup_enabled is True
    assert config.dedup_descriptor == "pooled4"
    assert config.dedup_threshold_mode == "group_quantile"
    assert config.dedup_quantile == 0.15
    assert config.diversity_activation_mode == "sample_stride"
    assert config.diversity_min_sample_stride_seconds == 5.0


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
    monkeypatch.setenv("CODEC_DEDUP_THRESHOLD_MODE", "group_quantile")
    monkeypatch.setenv("CODEC_DEDUP_QUANTILE", "0.2")
    monkeypatch.setenv("CODEC_DIVERSITY_ACTIVATION_MODE", "sample_stride")
    monkeypatch.setenv("CODEC_DIVERSITY_MIN_SAMPLE_STRIDE_SECONDS", "3")
    monkeypatch.setenv("CODEC_COMMON_CACHE_DIR", "/tmp/common-codec-cache")
    api.run_preinfer_config(PreinferConfig(video="input.mp4", out_dir=str(tmp_path)))

    config = captured["config"]
    assert config.selector_mode == "diverse_mixed_simple"
    assert config.diversity_fraction == 0.4
    assert config.novelty_weight == 1.0
    assert config.dedup_enabled is False
    assert config.dedup_descriptor == "full"
    assert config.dedup_threshold == 0.05
    assert config.dedup_threshold_mode == "group_quantile"
    assert config.dedup_quantile == 0.2
    assert config.diversity_activation_mode == "sample_stride"
    assert config.diversity_min_sample_stride_seconds == 3.0
    assert config.common_cache_dir == "/tmp/common-codec-cache"


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
