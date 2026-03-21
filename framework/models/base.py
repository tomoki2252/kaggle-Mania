"""モデルの抽象基底クラス"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ModelWrapper(ABC):
    """モデルラッパーの基底クラス

    全てのモデルは fit / predict_proba を実装する。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """モデル名"""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """学習"""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """TeamA が勝つ確率を返す (shape: (n_samples,))"""

    @property
    def handles_nan(self) -> bool:
        """モデルが NaN を扱えるかどうか"""
        return False

    def get_params(self) -> dict:
        """モデルパラメータを返す (保存用)"""
        return {}
