"""飞书通知连通性自测：python scripts/test_notify.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.monitor.alerts import FeishuNotifier, _load_dotenv


def main() -> None:
    _load_dotenv()
    notifier = FeishuNotifier()
    notifier.send(
        "飞书推送编码测试",
        "这是一条中文测试消息：每日信号 + 纸面调仓通知已就绪。"
        "UTF-8 编码校验 2026-08-16",
    )
    print("SEND OK")


if __name__ == "__main__":
    main()
