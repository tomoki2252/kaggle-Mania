"""Eloベースの予測モデル (学習不要)

diff_elo を受け取り、標準 Elo 勝率公式で予測確率を返す。
P(TeamA wins) = 1 / (1 + 10^(-diff_elo / 400))

オプションでElo差スケーリングを適用可能。
"""

import numpy as np
import pandas as pd

from framework.models.base import ModelWrapper


class EloPredictor(ModelWrapper):
    """Elo勝率公式による予測 (学習不要)"""

    def __init__(self, elo_scale: float = 1.0):
        self._elo_scale = elo_scale

    @property
    def name(self) -> str:
        return f"elo_predictor_scale{self._elo_scale}"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """学習不要 (no-op)"""
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        diff = X["diff_elo"].values * self._elo_scale
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def get_params(self) -> dict:
        return {"elo_scale": self._elo_scale}
