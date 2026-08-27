"""告警通知：飞书 / Telegram / 邮件 / 日志，支持多通道组合。"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import time
import urllib.parse
import urllib.request
from email.mime.text import MIMEText
from pathlib import Path
from typing import Protocol


class Notifier(Protocol):
    def send(self, title: str, body: str) -> None: ...


class LogNotifier:
    """默认通道：仅写日志，便于本地调试。"""

    def __init__(self):
        self.log = logging.getLogger("ashare.alert")

    def send(self, title: str, body: str) -> None:
        self.log.info("ALERT %s\n%s", title, body)


class TelegramNotifier:
    """Telegram Bot 推送：环境变量 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID。"""

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        if not self.token or not self.chat_id:
            raise ValueError("Telegram 通知需要 TELEGRAM_BOT_TOKEN 与 TELEGRAM_CHAT_ID")
        self.log = logging.getLogger("ashare.alert")

    def send(self, title: str, body: str) -> None:
        text = f"*{title}*\n{body[:3800]}"
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = urllib.parse.urlencode(
            {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
        ).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            resp.read()
        self.log.info("Telegram 已发送: %s", title)


class EmailNotifier:
    """SMTP 邮件推送：环境变量 SMTP_HOST/SMTP_USER/SMTP_PASSWORD/SMTP_TO。"""

    def __init__(
        self,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        to: str | None = None,
    ):
        self.host = host or os.getenv("SMTP_HOST", "")
        self.user = user or os.getenv("SMTP_USER", "")
        self.password = password or os.getenv("SMTP_PASSWORD", "")
        self.to = to or os.getenv("SMTP_TO", "")
        if not (self.host and self.user and self.to):
            raise ValueError("邮件通知需要 SMTP_HOST/SMTP_USER/SMTP_PASSWORD/SMTP_TO")
        self.log = logging.getLogger("ashare.alert")

    def send(self, title: str, body: str) -> None:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = title
        msg["From"] = self.user
        msg["To"] = self.to
        with smtplib.SMTP(self.host, 587, timeout=15) as server:
            server.starttls()
            if self.password:
                server.login(self.user, self.password)
            server.sendmail(self.user, [self.to], msg.as_string())
        self.log.info("邮件已发送: %s", title)


class FeishuNotifier:
    """飞书自定义应用推送：环境变量 FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_CHAT_ID。"""

    TOKEN_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    SEND_URL = "https://open.feishu.cn/open-apis/im/v1/messages"

    def __init__(
        self,
        app_id: str | None = None,
        app_secret: str | None = None,
        chat_id: str | None = None,
    ):
        self.app_id = app_id or os.getenv("FEISHU_APP_ID", "")
        self.app_secret = app_secret or os.getenv("FEISHU_APP_SECRET", "")
        self.chat_id = chat_id or os.getenv("FEISHU_CHAT_ID", "")
        if not (self.app_id and self.app_secret and self.chat_id):
            raise ValueError("飞书通知需要 FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_CHAT_ID")
        self._token: str | None = None
        self._token_expires = 0.0
        self.log = logging.getLogger("ashare.alert")

    def _tenant_access_token(self) -> str:
        """获取并缓存 tenant_access_token（有效期 2 小时，提前 5 分钟刷新）。"""
        if self._token and time.time() < self._token_expires:
            return self._token
        payload = json.dumps(
            {"app_id": self.app_id, "app_secret": self.app_secret}
        ).encode("utf-8")
        req = urllib.request.Request(
            self.TOKEN_URL,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("code") != 0:
            raise RuntimeError(f"飞书获取 token 失败: {data.get('msg')}")
        self._token = str(data["tenant_access_token"])
        expire = int(data.get("expire", 7200))
        self._token_expires = time.time() + max(60, expire - 300)
        return self._token

    def send(self, title: str, body: str) -> None:
        text = f"{title}\n{body[:3800]}"
        content = json.dumps({"text": text}, ensure_ascii=False)
        payload = json.dumps(
            {
                "receive_id": self.chat_id,
                "msg_type": "text",
                "content": content,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.SEND_URL}?receive_id_type=chat_id",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._tenant_access_token()}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("code") != 0:
            raise RuntimeError(f"飞书发送失败: {data.get('msg')}")
        self.log.info("飞书已发送: %s", title)


class MultiNotifier:
    """组合多个通道，任一失败不影响其他通道。"""

    def __init__(self, notifiers: list[Notifier]):
        self.notifiers = notifiers
        self.log = logging.getLogger("ashare.alert")

    def send(self, title: str, body: str) -> None:
        for n in self.notifiers:
            try:
                n.send(title, body)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("通知通道失败 %s: %s", type(n).__name__, exc)


def _load_dotenv(path: str | Path = ".env") -> None:
    """极简 .env 加载：不覆盖已有环境变量，零依赖。"""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_notifier() -> Notifier:
    """按环境变量自动装配通知通道；无配置时退回日志。"""
    _load_dotenv()
    channels: list[Notifier] = [LogNotifier()]
    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        channels.append(TelegramNotifier())
    if os.getenv("SMTP_HOST"):
        try:
            channels.append(EmailNotifier())
        except ValueError:
            pass
    if os.getenv("FEISHU_APP_ID") and os.getenv("FEISHU_APP_SECRET") and os.getenv("FEISHU_CHAT_ID"):
        try:
            channels.append(FeishuNotifier())
        except ValueError:
            pass
    return MultiNotifier(channels) if len(channels) > 1 else channels[0]
