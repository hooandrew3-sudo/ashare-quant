"""按年份分片拉取月末公告情绪：python scripts/fetch_sentiment_year.py <year> <root>"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.data.sentiment_cninfo import fetch_sentiment_dates
from quant.data.storage import Storage


def main(year: str, root: str) -> None:
    days = json.loads(
        Path("data/cache/sentiment_days_by_year.json").read_text(encoding="utf-8")
    )[year]
    senti = fetch_sentiment_dates(days, verbose=True)
    st = Storage(root)
    st.save("sentiment", senti, partition_by_symbol=False)
    print(f"year {year} done: {len(senti)} rows", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
