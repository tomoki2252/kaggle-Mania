"""任意の ModelWrapper にキャリブレーションを追加する汎用ラッパー

使い方:
    from framework.models.calibrated import CalibratedModelWrapper
    from framework.models.ridge_model import ScaledLogisticModel

    # 任意の ModelWrapper をラップ
    model = CalibratedModelWrapper(
        base_factory=lambda: ScaledLogisticModel(C=0.01),
        method="isotonic",
        cv=5,
    )

    # model_factory として渡す場合
    model_factory = lambda: CalibratedModelWrapper(
        base_factory=lambda: ScaledLogisticModel(C=0.01),
        method="isotonic",
    )
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

from framework.models.base import ModelWrapper


class _ModelWrapperAsEstimator(ClassifierMixin, BaseEstimator):
    """ModelWrapper を sklearn classifier として使えるようにするアダプタ"""

    def __init__(self, base_factory=None):
        self.base_factory = base_factory
        self._model = None

    def fit(self, X, y):
        self._model = self.base_factory()
        self._model.fit(pd.DataFrame(X), y)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        p = self._model.predict_proba(pd.DataFrame(X))
        return np.column_stack([1 - p, p])


class CalibratedModelWrapper(ModelWrapper):
    """任意の ModelWrapper にキャリブレーションを追加する汎用ラッパー

    内部で CalibratedClassifierCV を使用し、Isotonic regression または
    Platt scaling を適用する。

    Args:
        base_factory: ベースモデルを生成するファクトリ関数
        method: "isotonic" または "sigmoid" (Platt scaling)
        cv: キャリブレーション用の内部CV分割数
    """

    def __init__(
        self,
        base_factory: Callable[[], ModelWrapper],
        method: str = "isotonic",
        cv: int = 5,
    ):
        self._base_factory = base_factory
        self._method = method
        self._cv = cv
        self._calibrated_model: CalibratedClassifierCV | None = None
        self._base_name: str | None = None

    @property
    def name(self) -> str:
        if self._base_name is None:
            self._base_name = self._base_factory().name
        return f"{self._base_name}+{self._method}_cv{self._cv}"

    @property
    def handles_nan(self) -> bool:
        return self._base_factory().handles_nan

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        base_estimator = _ModelWrapperAsEstimator(base_factory=self._base_factory)
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self._calibrated_model = CalibratedClassifierCV(
            estimator=base_estimator, cv=self._cv, method=self._method
        )
        self._calibrated_model.fit(X_arr, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self._calibrated_model.predict_proba(X_arr)[:, 1]
