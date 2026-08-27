"""列出所有 model 的 strategy_fingerprint，定位生产可用的 real 模型。"""
import json
from pathlib import Path

# 我们关心的几个近期模型
candidates = [
    "20260819_201252_f75f83d6d7c1",  # 8/19 训练 (失败原因)
    "20260821_073939_03fb3161569b",  # 8/21 上午 (real.yaml 修复)
    "20260821_081542_45d789b950df",  # 8/21 上午 (ab_default 实验)
    "20260821_175428_083ed44c76d3",  # 8/21 下午 (ab_full_default)
    "20260821_175428_6467e03ad13b",  # 8/21 下午 (ab_full_real)
    "20260821_184318_480975b879ed",  # 8/21 晚上 (ab_hybrid 当前阻塞)
]

prod_real_fp = "797ba5f8960f"   # configs/real.yaml
prod_default_fp = "6651b9cbc504"  # configs/default.yaml

print(f"{'model_id':<35} {'strategy_fp':<15} {'data_fp':<22} {'matches real?':<15} {'matches default?'}")
print("=" * 110)
for mid in candidates:
    meta_p = Path(f"artifacts/models/{mid}/meta.json")
    if not meta_p.exists():
        print(f"{mid:<35} MISSING")
        continue
    meta = json.load(open(meta_p))
    sfp = meta.get("strategy_fingerprint", "?")
    dfp = meta.get("data_fingerprint", "?")
    matches_r = "YES" if sfp == prod_real_fp else ""
    matches_d = "YES" if sfp == prod_default_fp else ""
    print(f"{mid:<35} {sfp:<15} {str(dfp):<22} {matches_r:<15} {matches_d}")
