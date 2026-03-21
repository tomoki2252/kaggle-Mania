"""コーチ特徴量 v2: モード比較実験

minimal / tournament / full の3モードを比較し、
既存特徴量との冗長性が低い特徴量セットを特定する。

使い方:
    uv run python experiments/coach_experiment_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.config import ExperimentConfig
from framework.features.coach import CoachFeature
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
from framework.models.calibrated import CalibratedModelWrapper
from framework.models.ridge_model import ScaledLogisticModel
from framework.runner import run_experiment


def make_base_features():
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
        return res["combined"].get("overall", {}).get("brier_score", 1.0)
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
    print("  コーチ特徴量 v2: モード比較")
    print("=" * 60)

    all_results = {}

    experiments = [
        # ベースライン
        ("base_C0.01_iso", make_base_features(), 0.01),
        ("base_C0.05_iso", make_base_features(), 0.05),
        # minimal: tenure, is_new, mid_season_change のみ
        ("coach_minimal_C0.01_iso", make_base_features() + [CoachFeature(mode="minimal")], 0.01),
        ("coach_minimal_C0.05_iso", make_base_features() + [CoachFeature(mode="minimal")], 0.05),
        ("coach_minimal_C0.1_iso", make_base_features() + [CoachFeature(mode="minimal")], 0.1),
        # tournament: minimal + tourney_apps, tourney_win_rate, overperform
        ("coach_tourney_C0.01_iso", make_base_features() + [CoachFeature(mode="tournament")], 0.01),
        ("coach_tourney_C0.05_iso", make_base_features() + [CoachFeature(mode="tournament")], 0.05),
        ("coach_tourney_C0.1_iso", make_base_features() + [CoachFeature(mode="tournament")], 0.1),
    ]

    for name, features, C in experiments:
        config = ExperimentConfig(name=name, gender_mode="combined")
        summary = run_experiment(
            config=config,
            feature_generators=features,
            model_factory=lambda C=C: CalibratedModelWrapper(
                base_factory=lambda C=C: ScaledLogisticModel(C=C),
                method="isotonic", cv=5,
            ),
        )
        all_results[name] = summary
        brier = _get_brier(summary)
        m = _get_gender_brier(summary, "Men")
        w = _get_gender_brier(summary, "Women")
        print(f"  >> {name}: Brier={brier:.4f} [M={m}, W={w}]\n")

    # === サマリー ===
    print("\n" + "=" * 60)
    print("  最終結果サマリー")
    print("=" * 60)
    print(f"\n{'Name':<40} {'Overall':>8} {'Men':>8} {'Women':>8}")
    print("-" * 70)
    for name, summary in sorted(all_results.items(), key=lambda x: _get_brier(x[1])):
        brier = _get_brier(summary)
        m = _get_gender_brier(summary, "Men")
        w = _get_gender_brier(summary, "Women")
        print(f"  {name:<38} {brier:>8.4f} {m:>8} {w:>8}")

    print("\n完了")


if __name__ == "__main__":
    main()
