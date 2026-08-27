"""概率校准：Platt Scaling / Isotonic Regression，确保 score 与真实胜率一致。"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from quant.utils import setup_logging

LOGGER = logging.getLogger("ashare.calibration")


class ProbabilityCalibrator:
    """Wrap sklearn calibrator，提供 fit/transform 且兼容 NaN/边界。"""

    def __init__(self, method: Literal["isotonic", "sigmoid"] = "isotonic"):
        self.method = method
        self.model = None
        self._fitted = False

    def fit(self, scores: np.ndarray, y: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(y, dtype=int).ravel()
        mask = np.isfinite(scores)
        scores = scores[mask]
        y = y[mask]
        if len(scores) < 100 or len(np.unique(y)) < 2:
            LOGGER.warning("校准样本不足（n=%d），跳过拟合", len(scores))
            return
        x = scores.reshape(-1, 1)
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
        else:
            self.model = LogisticRegression(max_iter=1000, C=1.0)
        try:
            self.model.fit(x, y)
            self._fitted = True
            LOGGER.info("概率校准完成: method=%s, n=%d", self.method, len(scores))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("校准拟合失败: %s，跳过", exc)
            self._fitted = False

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted or self.model is None:
            return scores
        x = np.asarray(scores, dtype=float).reshape(-1, 1)
        if self.method == "isotonic":
            out = self.model.predict(x)
        else:
            out = self.model.predict_proba(x)[:, 1]
        return np.asarray(out, dtype=float)

    def fit_transform(self, scores: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(scores, y)
        return self.transform(scores)