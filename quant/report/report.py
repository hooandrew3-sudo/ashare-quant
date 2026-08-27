"""自包含 HTML 报告：摘要、净值曲线、IC、压力测试、交易明细。"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant.utils import ensure_dir


def _table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "<p>无数据</p>"
    out = df.head(max_rows).copy()
    return out.to_html(index=False, escape=True, border=0, classes="tbl")


def _metric_table(metrics: dict) -> str:
    rows = []
    labels = {
        "annualized_return": "年化收益",
        "annualized_vol": "年化波动",
        "sharpe": "Sharpe",
        "max_drawdown": "最大回撤",
        "calmar": "Calmar",
        "information_ratio": "信息比率",
        "monthly_win_rate": "月度胜率",
        "total_return": "累计收益",
        "benchmark_return": "基准收益",
        "excess_return": "超额收益",
        "alpha": "Alpha",
        "beta": "Beta",
        "turnover_annual": "年化换手",
        "cost_drag_total": "成本拖累(元)",
        "sell_hit_rate": "卖出命中率",
    }
    for k, label in labels.items():
        if k in metrics and metrics[k] is not None:
            v = metrics[k]
            if isinstance(v, float):
                if k in ("annualized_return", "max_drawdown", "excess_return",
                         "benchmark_return", "total_return", "monthly_win_rate",
                         "sell_hit_rate"):
                    v = f"{v:.2%}"
                elif k in ("sharpe", "calmar", "information_ratio", "beta"):
                    v = f"{v:.2f}"
            rows.append(f"<tr><td>{label}</td><td>{v}</td></tr>")
    oos = metrics.get("oos_model") or {}
    for k, label in (
        ("oos_folds", "OOS 折数"),
        ("oos_auc_mean", "OOS AUC"),
        ("oos_rank_ic_mean", "OOS Rank IC"),
        ("oos_rank_ic_ir", "OOS ICIR"),
    ):
        if oos.get(k) is not None:
            v = oos[k]
            if isinstance(v, float):
                v = f"{v:.4f}"
            rows.append(f"<tr><td>{label}</td><td>{v}</td></tr>")
    return "<table class='tbl'>" + "".join(rows) + "</table>"


def _stress_table(stress: dict) -> str:
    if not stress:
        return "<p>无压力测试</p>"
    rows = []
    for name, s in stress.items():
        dd = f"{s['max_drawdown']:.2%}" if s.get("max_drawdown") is not None else "-"
        ret = f"{s['return']:.2%}" if s.get("return") is not None else "-"
        status = "PASS" if s.get("ok") else ("NA" if s.get("reason") else "-")
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{s.get('coverage_days')}</td>"
                    f"<td>{dd}</td><td>{ret}</td><td>{status}</td></tr>")
    return "<table class='tbl'><tr><th>场景</th><th>覆盖天数</th><th>最大回撤</th><th>区间收益</th><th>状态</th></tr>" \
        + "".join(rows) + "</table>"


def _equity_img(equity: pd.DataFrame) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""
    if equity.empty:
        return ""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity["date"], equity["portfolio_value"], label="Strategy")
    ax.plot(equity["date"], equity["benchmark_value"], label="Benchmark", alpha=0.7)
    ax.set_title("Equity Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"<img src='data:image/png;base64,{b64}' style='width:100%'/>"


def build_html_report(
    title: str,
    output_path: str | Path,
    metrics: dict[str, Any],
    equity: pd.DataFrame,
    monthly: pd.DataFrame,
    stress: dict[str, Any],
    ic_frame: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> Path:
    out = Path(output_path)
    ensure_dir(out.parent)
    css = """
    <style>
      body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; margin: 24px; color: #222; }
      h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
      .tbl { border-collapse: collapse; font-size: 13px; }
      .tbl th, .tbl td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }
      .tbl tr:nth-child(even) { background: #f7f7f7; }
      pre { background: #f5f5f5; padding: 10px; font-size: 12px; overflow: auto; }
    </style>"""
    ic_html = _table(ic_frame) if ic_frame is not None else "<p>无 IC 数据</p>"
    trades_html = _table(trades) if trades is not None else "<p>无交易</p>"
    monthly_html = _table(monthly) if monthly is not None else "<p>无月度数据</p>"
    cfg_html = (
        f"<pre>{html.escape(json.dumps(config_snapshot, ensure_ascii=False, indent=2, default=str))}</pre>"
        if config_snapshot
        else ""
    )
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>{html.escape(title)}</title>{css}</head><body>
<h1>{html.escape(title)}</h1>
<h2>绩效摘要</h2>{_metric_table(metrics)}
<h2>净值曲线</h2>{_equity_img(equity)}
<h2>压力测试</h2>{_stress_table(stress)}
<h2>月度收益</h2>{monthly_html}
<h2>因子 IC</h2>{ic_html}
<h2>交易明细（前 50 条）</h2>{trades_html}
<h2>配置快照</h2>{cfg_html}
</body></html>"""
    out.write_text(doc, encoding="utf-8")
    return out
