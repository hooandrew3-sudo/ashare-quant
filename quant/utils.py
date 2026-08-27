"""通用工具：日志、文件哈希、可复现随机、进程锁。"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import sys
import time
from pathlib import Path


def setup_logging(verbose: bool = True) -> logging.Logger:
    """配置根日志：控制台 + 可选文件（由调用方 add_file_handler 开启）。"""
    logger = logging.getLogger("ashare")
    if logger.handlers:
        return logger
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def add_file_handler(logger: logging.Logger, path: Path) -> None:
    """将日志同时写入文件（生产模式）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)


def sha256_file(path: Path) -> str:
    """计算文件 sha256，用于数据/产物指纹。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_df(df) -> str:
    """计算 DataFrame 内容指纹（列名+值，稳定排序）。"""
    if df is None or len(df) == 0:
        return hashlib.sha256(b"empty").hexdigest()
    cols = sorted(df.columns.astype(str).tolist())
    body = df[cols].to_csv(index=False).encode("utf-8", "ignore")
    return hashlib.sha256(body).hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextlib.contextmanager
def process_lock(root: Path | str, timeout_sec: float = 5.0):
    """跨进程互斥锁（Windows msvcrt.locking）。

    防止计划任务补跑与手工运行并发：双写 paper state、并发增量同步。
    锁按 root 目录隔离。进程死亡时 Windows 自动释放强制锁（崩溃后可
    直接重入，无需陈旧检测）。获取失败快速抛错而非排队——量化任务
    宁可显式失败也不能双跑。
    """
    import msvcrt

    lock_path = Path(root) / ".pipeline.lock"
    Path(root).mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+b")  # noqa: SIM115
    deadline = time.time() + timeout_sec
    while True:
        try:
            # 先加锁后写入：Windows 强制锁下，对他人持有的锁定区域执行
            # seek/truncate/write 都会被拒（PermissionError）
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            break
        except OSError:
            if time.time() >= deadline:
                f.close()
                raise RuntimeError(
                    "另一个流水线进程正在运行（锁被占用），本次退出以避免并发双跑。"
                    f"锁文件: {lock_path}"
                )
            time.sleep(0.2)
    # 持锁后记录 PID（诊断用；失败不影响互斥语义）
    try:
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()).encode())
        f.flush()
    except OSError:
        pass
    try:
        yield
    finally:
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        f.close()
        with contextlib.suppress(OSError):
            os.remove(lock_path)
