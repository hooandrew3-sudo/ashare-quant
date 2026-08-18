"""模型注册表：产物落盘 + 元数据（数据指纹、参数、区间、指标）。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from quant.config import Config
from quant.utils import ensure_dir


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = ensure_dir(Path(root) / "models")

    def save_run(
        self,
        cfg: Config,
        data_fingerprint: str,
        result: dict[str, Any],
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + cfg.fingerprint()
        out = ensure_dir(self.root / run_id)
        models = result.get("models") or {}
        if models:
            for horizon, model in models.items():
                if model is not None:
                    joblib.dump(model, out / f"model_h{horizon}.joblib")
        model = result.get("final_model")
        if model is not None:
            joblib.dump(model, out / "model.joblib")
        calibrator = result.get("calibrator")
        if calibrator is not None:
            joblib.dump(calibrator, out / "calibrator.joblib")
        oos = result.get("oos")
        if oos is not None and not oos.empty:
            oos.to_parquet(out / "oos_scores.parquet", index=False)
        fm = result.get("fold_metrics")
        if fm is not None and not fm.empty:
            fm.to_csv(out / "fold_metrics.csv", index=False)
            fm.to_json(out / "fold_metrics.json", orient="records", indent=2)
        meta = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "config_fingerprint": cfg.fingerprint(),
            "data_fingerprint": data_fingerprint,
            "feature_cols": result.get("feature_cols", []),
            "horizons": sorted(models.keys()) if models else [cfg.model.horizon],
            "model_params": {
                "name": cfg.model.name,
                "params": cfg.model.params,
                "gbt_params": cfg.model.gbt_params,
            },
            "calibrator": getattr(calibrator, "method", None) if calibrator is not None else None,
            "fold_summary": {
                k: (round(float(v), 5) if isinstance(v, (int, float)) else v)
                for k, v in _summarize_folds(result.get("fold_metrics")).items()
            },
        }
        with open(out / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return run_id

    def load_run(self, run_id: str) -> dict[str, Any]:
        out = self.root / run_id
        horizons = (out / "meta.json").exists() and json.loads(
            (out / "meta.json").read_text(encoding="utf-8")
        ).get("horizons", [])
        models = {
            h: joblib.load(out / f"model_h{h}.joblib")
            for h in (horizons or [])
            if (out / f"model_h{h}.joblib").exists()
        }
        return {
            "model": joblib.load(out / "model.joblib") if (out / "model.joblib").exists() else None,
            "models": models,
            "calibrator": joblib.load(out / "calibrator.joblib")
            if (out / "calibrator.joblib").exists()
            else None,
            "oos": pd.read_parquet(out / "oos_scores.parquet")
            if (out / "oos_scores.parquet").exists()
            else pd.DataFrame(),
            "meta": json.loads((out / "meta.json").read_text(encoding="utf-8")),
        }


def _summarize_folds(fm) -> dict[str, float | int | None]:
    if fm is None or fm.empty:
        return {}
    out: dict[str, float | int | None] = {"n_folds": int(len(fm))}
    for col in ("auc", "rank_ic_excess"):
        vals = pd.to_numeric(fm[col], errors="coerce").dropna()
        out[f"{col}_mean"] = round(float(vals.mean()), 5) if len(vals) else None
        out[f"{col}_std"] = round(float(vals.std()), 5) if len(vals) > 1 else None
    return out
