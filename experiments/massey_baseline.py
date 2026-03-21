"""Massey Ordinals + Season Aggregates 統合 Baseline

全特徴量 (Elo + SeasonStats + Massey + Seed + TeamRating + RecentForm) を使い、
ScaledLogistic と Ridge を比較する。
ハイパーパラメータ最適化も実行する。

使い方:
    uv run python experiments/massey_baseline.py
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
from framework.models.ridge_model import RidgeModel, ScaledLogisticModel
from framework.runner import run_experiment


def make_features():
    """全特徴量ジェネレータ"""
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


def _get_brier(mode_result: dict) -> float:
    """mode の結果辞書から brier_score を取得"""
    # nested structure: results[mode]["overall"]["brier_score"]
    if "overall" in mode_result:
        return mode_result["overall"].get("brier_score", 1.0)
    return mode_result.get("brier_score", 1.0)


def _get_n(mode_result: dict) -> int:
    if "overall" in mode_result:
        return mode_result["overall"].get("n_samples", 0)
    return mode_result.get("n_samples", 0)


def extract_overall_brier(summary: dict) -> float:
    """summary から男女加重平均 Brier を計算"""
    res = summary.get("results", {})
    # combined mode
    if "combined" in res:
        return _get_brier(res["combined"])
    # separate mode: M + W の加重平均
    total_bs, total_n = 0.0, 0
    for mode in ["M", "W"]:
        if mode in res:
            bs = _get_brier(res[mode])
            n = _get_n(res[mode])
            total_bs += bs * n
            total_n += n
    return total_bs / max(total_n, 1)


def extract_gender_brier(summary: dict, gender: str) -> str:
    """特定 gender の Brier を文字列で返す"""
    res = summary.get("results", {})
    if gender in res:
        return f"{_get_brier(res[gender]):.4f}"
    if "combined" in res:
        # combined mode では by_gender を参照
        by_gender = res["combined"].get("by_gender", {})
        label = "Men" if gender == "M" else "Women"
        if label in by_gender:
            return f"{by_gender[label].get('brier_score', '-'):.4f}"
    return "-"


def print_summary(name: str, summary: dict):
    """結果を整形表示"""
    overall = extract_overall_brier(summary)
    m = extract_gender_brier(summary, "M")
    w = extract_gender_brier(summary, "W")
    print(f"  {name}: Overall={overall:.4f}  [M={m}, W={w}]")


def main():
    print("=" * 60)
    print("  Massey + 全特徴量 統合 Baseline 実験")
    print("=" * 60)

    all_results = {}

    # === Phase 1: Logistic C 探索 (separate) ===
    print("\n--- Phase 1: Logistic Regression C 探索 ---")
    for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
        name = f"massey_logistic_C{C}"
        config = ExperimentConfig(name=name, gender_mode="separate")
        summary = run_experiment(
            config=config,
            feature_generators=make_features(),
            model_factory=lambda C=C: ScaledLogisticModel(C=C),
        )
        all_results[name] = summary
        print_summary(name, summary)

    # === Phase 2: Ridge alpha 探索 (separate) ===
    print("\n--- Phase 2: Ridge Classifier alpha 探索 ---")
    for alpha in [0.1, 1.0, 10.0, 50.0, 100.0]:
        name = f"massey_ridge_{alpha}"
        config = ExperimentConfig(name=name, gender_mode="separate")
        summary = run_experiment(
            config=config,
            feature_generators=make_features(),
            model_factory=lambda alpha=alpha: RidgeModel(alpha=alpha),
        )
        all_results[name] = summary
        print_summary(name, summary)

    # === Phase 3: Best params で combined vs separate ===
    print("\n--- Phase 3: Best model - combined vs separate ---")

    # best logistic
    logistic_keys = [k for k in all_results if "logistic" in k]
    best_logistic_key = min(logistic_keys, key=lambda k: extract_overall_brier(all_results[k]))
    best_logistic_brier = extract_overall_brier(all_results[best_logistic_key])
    best_C = float(best_logistic_key.split("C")[1])
    print(f"  Best Logistic: C={best_C}, Brier={best_logistic_brier:.4f}")

    # best ridge
    ridge_keys = [k for k in all_results if "ridge" in k]
    best_ridge_key = min(ridge_keys, key=lambda k: extract_overall_brier(all_results[k]))
    best_ridge_brier = extract_overall_brier(all_results[best_ridge_key])
    best_alpha = float(best_ridge_key.split("ridge_")[1])
    print(f"  Best Ridge: alpha={best_alpha}, Brier={best_ridge_brier:.4f}")

    # Gender comparison with best model
    for model_label, factory, brier in [
        (f"logistic_C{best_C}", lambda: ScaledLogisticModel(C=best_C), best_logistic_brier),
        (f"ridge_{best_alpha}", lambda: RidgeModel(alpha=best_alpha), best_ridge_brier),
    ]:
        for mode in ["combined", "separate"]:
            name = f"massey_best_{model_label}_{mode}"
            config = ExperimentConfig(name=name, gender_mode=mode)
            summary = run_experiment(
                config=config,
                feature_generators=make_features(),
                model_factory=factory,
            )
            all_results[name] = summary
            print_summary(name, summary)

    # === 最終サマリー ===
    print("\n" + "=" * 60)
    print("  最終結果サマリー")
    print("=" * 60)
    print(f"\n{'Name':<45} {'Overall':>8} {'Men':>8} {'Women':>8}")
    print("-" * 75)
    for name, summary in sorted(all_results.items(), key=lambda x: extract_overall_brier(x[1])):
        overall = extract_overall_brier(summary)
        m_str = extract_gender_brier(summary, "M")
        w_str = extract_gender_brier(summary, "W")
        print(f"  {name:<43} {overall:>8.4f} {m_str:>8} {w_str:>8}")

    # === Coefficient report ===
    print("\n" + "=" * 60)
    print("  Coefficient Report (Best Logistic)")
    print("=" * 60)

    from framework.data_loader import load_all_data, build_tourney_matchups
    from framework.runner import _generate_all_features, _build_matchup_features

    data = load_all_data(ExperimentConfig(name="tmp").data_dir)
    features_gens = make_features()

    for gender in ["M", "W"]:
        team_feat, feat_cols = _generate_all_features(features_gens, data, gender, 2026)
        train_matchups = build_tourney_matchups(data[f"{gender}_tourney"], gender)
        train_matchups = train_matchups[
            (train_matchups["Season"] >= 2003) & (train_matchups["Season"] < 2026)
        ]
        if len(train_matchups) == 0:
            continue

        tr_features, diff_cols = _build_matchup_features(train_matchups, team_feat, feat_cols)

        X_train = tr_features[diff_cols]
        y_train = train_matchups["label"].values

        model = ScaledLogisticModel(C=best_C)
        model.fit(X_train, y_train)
        coef_df = model.get_coefficients(diff_cols)

        print(f"\n  --- {gender} Top 20 (C={best_C}) ---")
        print(f"  {'Rank':<5} {'Feature':<40} {'Coef':>10}")
        print("  " + "-" * 58)
        for i, row in coef_df.head(20).iterrows():
            print(f"  {i+1:<5} {row['feature']:<40} {row['coefficient']:>10.4f}")

        out_dir = Path(__file__).resolve().parent.parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        coef_df.to_csv(out_dir / f"massey_coef_{gender}.csv", index=False)

    print("\n完了")


if __name__ == "__main__":
    main()
