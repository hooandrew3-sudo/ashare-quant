"""端到端流水线测试：demo 数据全链路可复现。"""

from __future__ import annotations

from pathlib import Path

from quant.config import Config
from quant.data.storage import Storage
from quant.pipeline import prepare_data, run_research


def test_yaml_config_nested_build(tmp_path):
    from quant.config import load_config

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "data:\n  source: baostock\n  sync:\n    universe: csi800\n"
        "backtest:\n  start: '2024-02-01'\n  end: '2026-07-31'\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.data.source == "baostock"
    assert cfg.data.sync.universe == "csi800"
    assert cfg.backtest.start == "2024-02-01"
    assert isinstance(cfg.data.root, Path)


def test_yaml_unknown_key_rejected(tmp_path):
    """未知配置键必须报错，禁止静默忽略（防拼写错误导致策略走错参数）。"""
    from quant.config import load_config

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "portfolio:\n  top_n: 20\n  max_weight: 0.05\n  turnover_capp: 0.2\n",
        encoding="utf-8",
    )
    try:
        load_config(cfg_path)
    except ValueError as exc:
        assert "turnover_capp" in str(exc)
    else:
        raise AssertionError("未知键应触发 ValueError")


def test_config_enum_validation():
    cfg = Config()
    cfg.data.source = "parquet"  # 绕过 synthetic 隔离校验，聚焦枚举校验
    cfg.model.label_mode = "binaryy"  # 拼写错误
    try:
        cfg.validate()
    except ValueError as exc:
        assert "label_mode" in str(exc)
    else:
        raise AssertionError("非法枚举应触发 ValueError")


def test_demo_pipeline(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path / "data"
    cfg.data.demo.n_stocks = 60
    cfg.data.demo.years = 4
    cfg.run.seed = 5
    cfg.model.n_splits = 3
    cfg.model.params["n_estimators"] = 60
    cfg.model.gbt_params["n_estimators"] = 60
    output = tmp_path / "artifacts"

    prepare_data(cfg)
    storage = Storage(cfg.data.root)
    bundle = storage.load_bundle()
    res = run_research(cfg, bundle, output_root=output)

    assert res["run_id"]
    assert res["metrics"].get("annualized_return") is not None
    assert "max_drawdown" in res["metrics"]
    assert (output / res["run_id"] / "equity.parquet").exists()
    assert (output / res["run_id"] / "factors.parquet").exists()
    assert (output / res["run_id"] / "ic_report.json").exists()
    assert res["report"].endswith(".html")

    # 可复现性：同配置同种子 → 同指标
    cfg2 = Config()
    cfg2.data.root = tmp_path / "data"
    cfg2.data.demo.n_stocks = 60
    cfg2.data.demo.years = 4
    cfg2.run.seed = 5
    cfg2.model.n_splits = 3
    cfg2.model.params["n_estimators"] = 60
    cfg2.model.gbt_params["n_estimators"] = 60
    bundle2 = storage.load_bundle()
    res2 = run_research(cfg2, bundle2, output_root=output)
    assert res2["metrics"]["annualized_return"] == res["metrics"]["annualized_return"]
