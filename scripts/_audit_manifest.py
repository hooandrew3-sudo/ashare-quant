import json
from pathlib import Path

m = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
ds = m.get("datasets", {})
for k, v in ds.items():
    syms = list(v.keys())
    if syms and syms != ["*"]:
        sample = v[syms[0]]
        print(f"{k}: {len(syms)} symbols, sample start={sample.get('start')}, end={sample.get('end')}")
    else:
        print(f"{k}: {v}")
print("meta:", m.get("meta"))
print("updated_at:", m.get("updated_at"))
