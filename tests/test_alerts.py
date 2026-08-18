"""飞书通知测试：tenant_access_token 获取/缓存、UTF-8 消息体编码。"""

from __future__ import annotations

import json
from unittest import mock

from quant.monitor.alerts import FeishuNotifier


class _FakeResp:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _fake_urlopen(calls: list):
    def _urlopen(req, timeout=15):  # noqa: ANN001
        calls.append(req)
        if "tenant_access_token" in req.full_url:
            return _FakeResp(
                json.dumps(
                    {"code": 0, "tenant_access_token": "tok_123", "expire": 7200}
                ).encode("utf-8")
            )
        return _FakeResp(json.dumps({"code": 0, "msg": "success"}).encode("utf-8"))

    return _urlopen


def test_feishu_send_utf8_payload() -> None:
    calls: list = []
    n = FeishuNotifier(app_id="cli_test", app_secret="secret", chat_id="oc_test")
    with mock.patch(
        "quant.monitor.alerts.urllib.request.urlopen", side_effect=_fake_urlopen(calls)
    ):
        n.send("每日信号", "候选 1041 只\nTop10:\n600008.SH: 0.8855")

    assert len(calls) == 2
    send_req = calls[1]
    assert send_req.get_header("Authorization") == "Bearer tok_123"
    assert "receive_id_type=chat_id" in send_req.full_url
    body = json.loads(send_req.data.decode("utf-8"))
    assert body["receive_id"] == "oc_test"
    assert body["msg_type"] == "text"
    inner = json.loads(body["content"])
    assert inner["text"] == "每日信号\n候选 1041 只\nTop10:\n600008.SH: 0.8855"
    assert "每日信号".encode("utf-8") in send_req.data  # 中文以 UTF-8 明文传输


def test_feishu_token_cached() -> None:
    calls: list = []
    n = FeishuNotifier(app_id="cli_test", app_secret="secret", chat_id="oc_test")
    with mock.patch(
        "quant.monitor.alerts.urllib.request.urlopen", side_effect=_fake_urlopen(calls)
    ):
        n.send("a", "b")
        n.send("c", "d")
    token_calls = [c for c in calls if "tenant_access_token" in c.full_url]
    assert len(token_calls) == 1  # token 复用，不重复获取
    assert len(calls) == 3
