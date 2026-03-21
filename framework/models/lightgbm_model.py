"""LightGBM モデル"""

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from framework.models.base import ModelWrapper

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.02,
    "num_leaves": 8,
    "max_depth": 5,
    "min_child_samples": 30,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "verbose": -1,
    "n_jobs": -1,
}


class LightGBMModel(ModelWrapper):
    """LightGBM バイナリ分類モデル

    アンサンブル (複数シードで学習して平均) に対応。
    """

    def __init__(
        self,
        params: dict | None = None,
        num_boost_round: int = 2000,
        n_ensemble: int = 1,
        seed: int = 42,
    ):
        if lgb is None:
            raise ImportError("lightgbm が必要です: uv add lightgbm")
        self._params = {**DEFAULT_PARAMS, **(params or {})}
        self._num_boost_round = num_boost_round
        self._n_ensemble = n_ensemble
        self._seed = seed
        self._models: list = []
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        return f"lightgbm_n{self._n_ensemble}"

    @property
    def handles_nan(self) -> bool:
        return True

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._feature_names = list(X.columns)
        self._models = []

        for i in range(self._n_ensemble):
            seed = self._seed + i * 17
            params = {
                **self._params,
                "seed": seed,
                "subsample_seed": seed,
                "colsample_bytree": 0.6 + (i % 3) * 0.1,
                "subsample": 0.65 + (i % 3) * 0.05,
            }
            dtrain = lgb.Dataset(
                X.values.astype(np.float32), y,
                feature_name=self._feature_names,
            )
            model = lgb.train(
                params, dtrain,
                num_boost_round=self._num_boost_round,
            )
            self._models.append(model)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(X))
        x_vals = X.values.astype(np.float32)
        for model in self._models:
            preds += model.predict(x_vals)
        return preds / len(self._models)

    def get_params(self) -> dict:
        result = {
            "params": self._params,
            "num_boost_round": self._num_boost_round,
            "n_ensemble": self._n_ensemble,
        }
        if self._models:
            imp = self._models[0].feature_importance(importance_type="gain")
            result["feature_importance"] = dict(
                zip(self._feature_names, imp.tolist())
            )
        return result
