"""通用工具：日志、文件哈希、可复现随机。"""

from __future__ import annotations

import hashlib
import logging
import sys
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
