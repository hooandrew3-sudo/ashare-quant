"""模型层测试：标签、特征、Walk-Forward 无前视。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import Config
from quant.data.synthetic import generate_synthetic
from quant.factors.compute import compute_all_factors
from quant.model.label import build_label
from quant.model.train import prepare_xy, walk_forward


def _setup():
    cfg = Config()
    cfg.data.demo.n_stocks = 50
    cfg.data.demo.years = 4
    cfg.run.seed = 3
    cfg.model.n_splits = 3
    cfg.model.params["n_estimators"] = 50
    cfg.model.gbt_params["n_estimators"] = 50
    bundle = generate_synthetic(
        n_stocks=cfg.data.demo.n_stocks,
        years=cfg.data.demo.years,
        seed=cfg.run.seed,
    )
    return cfg, bundle


def test_label_no_lookahead():
    cfg, bundle = _setup()
    label = build_label(bundle.prices, bundle.benchmark, cfg)
    assert set(["date", "symbol", "excess", "value"]).issubset(label.columns)
    assert label["value"].isin([0, 1]).all()
    # 最后 horizon 天无标签
    last_label_date = label["date"].max()
    assert last_label_date < bundle.prices["date"].max()


def test_quantile_contrast_and_regression_labels():
    cfg, bundle = _setup()
    cfg.model.label_mode = "quantile_contrast"
    label_q = build_label(bundle.prices, bundle.benchmark, cfg)
    assert label_q["value"].isin([0, 1]).all()
    assert len(label_q) < len(bundle.prices) * 0.8  # 中间 40% 被剔除
    cfg.model.label_mode = "regression"
    label_r = build_label(bundle.prices, bundle.benchmark, cfg)
    assert label_r["value"].dtype.kind == "f"


def test_walk_forward_no_leakage():
    cfg, bundle = _setup()
    factor_long = compute_all_factors(bundle, cfg)
    label_long = build_label(bundle.prices, bundle.benchmark, cfg)
    xy = prepare_xy(factor_long, label_long, cfg)
    feature_cols = [c for c in xy.columns if c.startswith("f_")]
    res = walk_forward(xy, cfg, feature_cols)
    oos = res["oos"]
    assert not oos.empty
    # 训练日期必须严格早于测试日期
    for _, row in res["fold_metrics"].iterrows():
        train_end = pd.Timestamp(row["train_end"])
        test_min = pd.Timestamp(oos.loc[oos["fold"] == row["fold"], "date"].min())
        assert test_min > train_end
    assert oos["score"].between(0, 1).all()


def test_ensemble_scores():
    from quant.model.train import ensemble_scores

    dates = pd.bdate_range("2023-01-02", periods=20)
    df5 = pd.DataFrame(
        {
            "date": dates,
            "symbol": [f"S{i % 4}" for i in range(20)],
            "score": [i / 20 for i in range(20)],
            "fold": [1] * 20,
        }
    )
    df20 = pd.DataFrame(
        {
            "date": dates,
            "symbol": [f"S{i % 4}" for i in range(20)],
            "score": [0.9 - i / 20 for i in range(20)],
            "fold": [1] * 20,
        }
    )
    merged = ensemble_scores({5: df5, 20: df20})
    assert "score" in merged.columns
    assert merged["score"].between(0, 1).all()
    assert len(merged) == 20


def test_model_gate():
    from quant.model.gate import evaluate_model_gate

    cfg = Config()
    cfg.model.gate_min_composite_t = 5.0
    meta = {
        "fold_summary": {"n_folds": 10, "auc_mean": 0.5, "rank_ic_excess_mean": 0.01},
        "composite_t": 8.1,
    }
    assert evaluate_model_gate(cfg, meta)["passed"]
    # 复合因子 t 低于门槛 → 不通过
    assert not evaluate_model_gate(cfg, dict(meta, composite_t=3.0))["passed"]
    # 旧模型缺 composite_t → 不通过（要求重训）
    assert not evaluate_model_gate(cfg, dict(meta, composite_t=None))["passed"]
    # 禁用复合因子时跳过该检查
    cfg.factors.composite = False
    assert evaluate_model_gate(cfg, dict(meta, composite_t=None))["passed"]
