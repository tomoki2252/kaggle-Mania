"""ロジスティック回帰モデル"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from framework.models.base import ModelWrapper


class LogisticModel(ModelWrapper):
    """scikit-learn LogisticRegression のラッパー"""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self.model = None

    @property
    def name(self) -> str:
        return f"logistic_C{self.C}"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.model = LogisticRegression(
            C=self.C, solver="lbfgs", max_iter=self.max_iter
        )
        self.model.fit(X.values, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X.values)[:, 1]

    def get_params(self) -> dict:
        params = {"C": self.C, "max_iter": self.max_iter}
        if self.model is not None:
            params["coefficients"] = dict(
                zip(
                    self.feature_names_ if hasattr(self, "feature_names_") else [],
                    self.model.coef_[0].tolist(),
                )
            )
            params["intercept"] = self.model.intercept_[0]
        return params
