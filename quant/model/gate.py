"""模型上线门禁：基于注册模型元数据的 OOS 质量评估（可配置阈值）。

默认口径遵循 docs/FINAL_VERDICT.md 的重标定结论：
  - ML 层（OOS AUC / 单因子 IC）无增量，默认不设门槛（min=0，仅作信息展示）；
  - 信号质量以复合因子 t 值为主门槛（默认 ≥5）；
  - 必须有足够 OOS fold 数，避免在少量样本上评估。
"""

from __future__ import annotations

from typing import Any

from quant.config import Config


def evaluate_model_gate(cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    """评估模型是否满足上线门禁。

    meta 来自 ModelRegistry.load_run()["meta"]（fold_summary + composite_t）。
    """
    m = cfg.model
    fold = meta.get("fold_summary") or {}
    checks: dict[str, dict[str, Any]] = {}

    n_folds = fold.get("n_folds")
    checks["n_folds"] = {
        "value": n_folds,
        "min": m.gate_min_folds,
        "ok": (n_folds or 0) >= m.gate_min_folds,
    }
    auc = fold.get("auc_mean")
    checks["auc"] = {
        "value": auc,
        "min": m.gate_min_auc,
        # AUC 为信息项：默认 min=0，缺失不阻断
        "ok": auc is None or auc >= m.gate_min_auc,
    }
    ric = fold.get("rank_ic_excess_mean")
    checks["rank_ic"] = {
        "value": ric,
        "min": m.gate_min_rank_ic,
        "ok": ric is None or ric >= m.gate_min_rank_ic,
    }
    ct = meta.get("composite_t")
    if cfg.factors.composite:
        # 复合因子启用时，t 值为硬门槛；缺失（旧模型/无元数据）视为不通过
        checks["composite_t"] = {
            "value": ct,
            "min": m.gate_min_composite_t,
            "ok": ct is not None and ct >= m.gate_min_composite_t,
        }
    else:
        checks["composite_t"] = {
            "value": None,
            "min": m.gate_min_composite_t,
            "ok": True,
            "skipped": True,
        }

    failed = [k for k, c in checks.items() if not c["ok"]]
    return {
        "enabled": m.gate_enabled,
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
    }
