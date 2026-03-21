"""評価メトリクス"""

import numpy as np
import pandas as pd


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier score (= MSE for binary classification)"""
    return float(np.mean((y_pred - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Log loss (binary cross entropy)"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy (threshold=0.5)"""
    return float(np.mean((y_pred > 0.5) == y_true))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seasons: np.ndarray | None = None,
    genders: np.ndarray | None = None,
) -> dict:
    """包括的な評価レポートを生成

    Args:
        y_true: 正解ラベル
        y_pred: 予測確率
        seasons: シーズン配列 (年別の内訳用)
        genders: 性別配列 ("M"/"W") (男女別の内訳用)

    Returns:
        評価結果の辞書
    """
    result = {
        "overall": {
            "brier_score": brier_score(y_true, y_pred),
            "log_loss": log_loss(y_true, y_pred),
            "accuracy": accuracy(y_true, y_pred),
            "n_samples": len(y_true),
        }
    }

    # 年別の内訳
    if seasons is not None:
        result["by_season"] = {}
        for season in sorted(set(seasons)):
            mask = seasons == season
            if mask.sum() == 0:
                continue
            result["by_season"][int(season)] = {
                "brier_score": brier_score(y_true[mask], y_pred[mask]),
                "log_loss": log_loss(y_true[mask], y_pred[mask]),
                "accuracy": accuracy(y_true[mask], y_pred[mask]),
                "n_samples": int(mask.sum()),
            }

    # 男女別の内訳
    if genders is not None:
        result["by_gender"] = {}
        for gender in sorted(set(genders)):
            mask = genders == gender
            if mask.sum() == 0:
                continue
            label = "Men" if gender == "M" else "Women"
            result["by_gender"][label] = {
                "brier_score": brier_score(y_true[mask], y_pred[mask]),
                "log_loss": log_loss(y_true[mask], y_pred[mask]),
                "accuracy": accuracy(y_true[mask], y_pred[mask]),
                "n_samples": int(mask.sum()),
            }

    return result


def format_evaluation_report(eval_result: dict) -> str:
    """評価結果を見やすいテキストに整形"""
    lines = []
    ov = eval_result["overall"]
    lines.append(f"Overall: Brier={ov['brier_score']:.4f}, "
                 f"LogLoss={ov['log_loss']:.4f}, "
                 f"Acc={ov['accuracy']:.4f}, "
                 f"N={ov['n_samples']}")

    if "by_gender" in eval_result:
        lines.append("")
        for gender, metrics in eval_result["by_gender"].items():
            lines.append(f"  {gender}: Brier={metrics['brier_score']:.4f}, "
                         f"Acc={metrics['accuracy']:.4f}, N={metrics['n_samples']}")

    if "by_season" in eval_result:
        lines.append("")
        for season, metrics in eval_result["by_season"].items():
            lines.append(f"  {season}: Brier={metrics['brier_score']:.4f}, "
                         f"Acc={metrics['accuracy']:.4f}, N={metrics['n_samples']}")

    return "\n".join(lines)
