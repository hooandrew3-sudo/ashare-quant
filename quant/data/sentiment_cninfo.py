"""公告情绪因子接入（巨潮资讯 CNINFO，合法公开来源）。

策略：只在月末调仓日（含末日前 7 天窗口可选）拉取全市场公告标题，
用金融情感词典打分（预增/回购/中标/增持 = 正向；减持/亏损/立案/处罚 = 负向），
按 (symbol, date) 汇总。事件时点 = 公告日，杜绝前视。
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import pandas as pd

from quant.utils import setup_logging


POSITIVE_WORDS = [
    "预增", "回购", "中标", "增持", "签约", "扭亏", "分红", "送转",
    "高送转", "业绩增长", "重大合同", "收购", "合并", "定增", "获批",
    "突破", "创新高", "盈利",
]
NEGATIVE_WORDS = [
    "预减", "亏损", "立案", "处罚", "违规", "诉讼", "仲裁", "减持",
    "质押", "冻结", "终止", "退市", "风险警示", "ST", "警示函",
    "问询", "关注函", "整改", "赔偿", "逾期",
]


def score_title(title: str) -> float:
    t = title or ""
    s = 0.0
    for w in POSITIVE_WORDS:
        if w in t:
            s += 1.0
    for w in NEGATIVE_WORDS:
        if w in t:
            s -= 1.0
    return s


def _query_day(se_date: str, page: int = 1, page_size: int = 30) -> dict:
    url = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
    data = {
        "pageNum": str(page), "pageSize": str(page_size), "column": "szse",
        "tabName": "fulltext", "plate": "", "stock": "", "searchkey": "",
        "secid": "", "category": "", "trade": "", "seDate": f"{se_date}~{se_date}",
        "sortName": "", "sortType": "", "isHLtitle": "true",
    }
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "http://www.cninfo.com.cn/",
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def fetch_day(se_date: str, log=None, pause: float = 0.15) -> pd.DataFrame:
    """拉取某日全市场公告并打分，返回 (symbol, date, sentiment)。"""
    log = log or setup_logging(False)
    page, rows = 1, []
    while True:
        try:
            r = _query_day(se_date, page=page)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s 第 %d 页失败: %s", se_date, page, exc)
            break
        anns = r.get("announcements") or []
        if not anns:
            break
        for a in anns:
            sec_code = a.get("secCode", "")
            if len(str(sec_code)) != 6:
                continue
            if sec_code[0] in ("6", "9"):
                symbol = f"{sec_code}.SH"
            elif sec_code[0] in ("0", "3"):
                symbol = f"{sec_code}.SZ"
            else:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp(se_date),
                    "sentiment": score_title(a.get("announcementTitle", "")),
                }
            )
        total_pages = int(r.get("totalpages") or 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(pause)
    return pd.DataFrame(rows)


def fetch_sentiment_dates(dates: list[str], verbose: bool = True) -> pd.DataFrame:
    """批量拉取多个交易日的公告情绪，返回 long 表。"""
    log = setup_logging(verbose)
    parts: list[pd.DataFrame] = []
    for i, d in enumerate(dates, 1):
        t0 = time.time()
        df = fetch_day(d, log=log)
        if not df.empty:
            parts.append(df)
        log.info("[%d/%d] %s: %d 条公告评分 (%.1fs)", i, len(dates), d, len(df), time.time() - t0)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["symbol", "date", "sentiment"]
    )
    if not out.empty:
        out = out.groupby(["symbol", "date"], as_index=False)["sentiment"].sum()
    return out
