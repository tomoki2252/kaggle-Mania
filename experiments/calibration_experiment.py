"""キャリブレーション効果の検証

ベストモデル (Massey全特徴量 + Logistic C=0.01, combined) に対して
Isotonic regression / Platt scaling によるキャリブレーションを追加し、
キャリブレーション無しと比較する。

使い方:
    uv run python experiments/calibration_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.config import ExperimentConfig
from framework.features.elo import (
    EloFeature,
    ELO_PARAMS_OPTIMIZED_M,
    ELO_PARAMS_OPTIMIZED_W,
)
from framework.features.massey import MasseyFeature
from framework.features.recent_form import RecentFormFeature
from framework.features.season_stats import SeasonStatsFeature
from framework.features.seed_diff import SeedDiffFeature
from framework.features.team_rating import TeamRatingFeature
from framework.models.ridge_model import CalibratedLogisticModel, ScaledLogisticModel
from framework.runner import run_experiment


def make_features():
    return [
        SeedDiffFeature(),
        EloFeature(
            params_by_gender={
                "M": ELO_PARAMS_OPTIMIZED_M,
                "W": ELO_PARAMS_OPTIMIZED_W,
            },
            compute_sos=True,
            compute_conf_strength=True,
        ),
        SeasonStatsFeature(use_detailed=True),
        MasseyFeature(),
        RecentFormFeature(last_n=10),
        TeamRatingFeature(method="iterative"),
    ]


def _get_brier(summary: dict) -> float:
    res = summary.get("results", {})
    if "combined" in res:
        overall = res["combined"].get("overall", {})
        return overall.get("brier_score", 1.0)
    return 1.0


def _get_gender_brier(summary: dict, label: str) -> str:
    res = summary.get("results", {})
    if "combined" in res:
        by_gender = res["combined"].get("by_gender", {})
        if label in by_gender:
            return f"{by_gender[label].get('brier_score', '-'):.4f}"
    return "-"


def main():
    print("=" * 60)
    print("  キャリブレーション効果の検証")
    print("=" * 60)

    all_results = {}

    experiments = [
        # ベースライン (キャリブレーション無し)
        ("massey_logistic_C0.01_no_cal", lambda: ScaledLogisticModel(C=0.01)),
        # Isotonic regression (cv=5)
        ("massey_logistic_C0.01_isotonic_cv5", lambda: CalibratedLogisticModel(C=0.01, calibration_method="isotonic", cv_calibration=5)),
        # Isotonic regression (cv=3)
        ("massey_logistic_C0.01_isotonic_cv3", lambda: CalibratedLogisticModel(C=0.01, calibration_method="isotonic", cv_calibration=3)),
        # Platt scaling (cv=5)
        ("massey_logistic_C0.01_sigmoid_cv5", lambda: CalibratedLogisticModel(C=0.01, calibration_method="sigmoid", cv_calibration=5)),
        # Platt scaling (cv=3)
        ("massey_logistic_C0.01_sigmoid_cv3", lambda: CalibratedLogisticModel(C=0.01, calibration_method="sigmoid", cv_calibration=3)),
        # C値も変えて試す
        ("massey_logistic_C0.02_isotonic_cv5", lambda: CalibratedLogisticModel(C=0.02, calibration_method="isotonic", cv_calibration=5)),
        ("massey_logistic_C0.05_isotonic_cv5", lambda: CalibratedLogisticModel(C=0.05, calibration_method="isotonic", cv_calibration=5)),
        ("massey_logistic_C0.1_isotonic_cv5", lambda: CalibratedLogisticModel(C=0.1, calibration_method="isotonic", cv_calibration=5)),
    ]

    for name, factory in experiments:
        config = ExperimentConfig(name=name, gender_mode="combined")
        summary = run_experiment(
            config=config,
            feature_generators=make_features(),
            model_factory=factory,
        )
        all_results[name] = summary
        brier = _get_brier(summary)
        m = _get_gender_brier(summary, "Men")
        w = _get_gender_brier(summary, "Women")
        print(f"  >> {name}: Overall={brier:.4f} [M={m}, W={w}]\n")

    # === 最終サマリー ===
    print("\n" + "=" * 60)
    print("  最終結果サマリー")
    print("=" * 60)
    print(f"\n{'Name':<50} {'Overall':>8} {'Men':>8} {'Women':>8}")
    print("-" * 80)
    for name, summary in sorted(all_results.items(), key=lambda x: _get_brier(x[1])):
        brier = _get_brier(summary)
        m = _get_gender_brier(summary, "Men")
        w = _get_gender_brier(summary, "Women")
        print(f"  {name:<48} {brier:>8.4f} {m:>8} {w:>8}")

    print("\n完了")


if __name__ == "__main__":
    main()
