"""Ridge Classifier (確率出力付き) と Scaled Logistic Regression

NaN は SimpleImputer(median) で補完する。
これにより Massey (男子のみ) 等が女子データで NaN でも動作する。
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler

from framework.models.base import ModelWrapper


class ScaledLogisticModel(ModelWrapper):
    """SimpleImputer + StandardScaler + LogisticRegression"""

    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self.C = C
        self.max_iter = max_iter
        self._imputer = None
        self._scaler = None
        self._model = None

    @property
    def name(self) -> str:
        return f"scaled_logistic_C{self.C}"

    @property
    def handles_nan(self) -> bool:
        return True

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self._imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_imp = self._imputer.fit_transform(X_arr)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_imp)
        self._model = LogisticRegression(
            C=self.C, solver="lbfgs", max_iter=self.max_iter, random_state=42
        )
        self._model.fit(X_scaled, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_imp = self._imputer.transform(X_arr)
        X_scaled = self._scaler.transform(X_imp)
        return self._model.predict_proba(X_scaled)[:, 1]

    def get_coefficients(self, feature_names: list[str]) -> pd.DataFrame:
        """係数レポート (scaled 空間での係数)"""
        if self._model is None:
            return pd.DataFrame()
        coefs = self._model.coef_[0]
        df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        })
        return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


class CalibratedLogisticModel(ModelWrapper):
    """SimpleImputer + StandardScaler + LogisticRegression + Calibration

    LogisticRegression の出力確率に対して、CalibratedClassifierCV で
    Isotonic regression または Platt scaling を適用する。
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 2000,
        calibration_method: str = "isotonic",
        cv_calibration: int = 5,
    ):
        self.C = C
        self.max_iter = max_iter
        self.calibration_method = calibration_method
        self._cv_calibration = cv_calibration
        self._imputer = None
        self._scaler = None
        self._model = None

    @property
    def name(self) -> str:
        return f"calibrated_logistic_C{self.C}_{self.calibration_method}"

    @property
    def handles_nan(self) -> bool:
        return True

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self._imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_imp = self._imputer.fit_transform(X_arr)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_imp)
        base = LogisticRegression(
            C=self.C, solver="lbfgs", max_iter=self.max_iter, random_state=42
        )
        self._model = CalibratedClassifierCV(
            estimator=base, cv=self._cv_calibration, method=self.calibration_method
        )
        self._model.fit(X_scaled, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_imp = self._imputer.transform(X_arr)
        X_scaled = self._scaler.transform(X_imp)
        return self._model.predict_proba(X_scaled)[:, 1]


class RidgeModel(ModelWrapper):
    """SimpleImputer + StandardScaler + RidgeClassifier + Platt Calibration"""

    def __init__(self, alpha: float = 1.0, cv_calibration: int = 3):
        self.alpha = alpha
        self._cv_calibration = cv_calibration
        self._imputer = None
        self._scaler = None
        self._model = None

    @property
    def name(self) -> str:
        return f"ridge_alpha{self.alpha}"

    @property
    def handles_nan(self) -> bool:
        return True

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self._imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_imp = self._imputer.fit_transform(X_arr)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_imp)
        base = RidgeClassifier(alpha=self.alpha, random_state=42)
        self._model = CalibratedClassifierCV(
            estimator=base, cv=self._cv_calibration, method="sigmoid"
        )
        self._model.fit(X_scaled, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_imp = self._imputer.transform(X_arr)
        X_scaled = self._scaler.transform(X_imp)
        return self._model.predict_proba(X_scaled)[:, 1]
