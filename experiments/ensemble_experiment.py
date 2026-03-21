"""アンサンブル実験

仮説: 異なるモデルの予測を組み合わせることで、
各モデルの弱点を補完し精度が改善する。

検証方法:
- CV内で各モデルの予測を生成し、ブレンドした予測でBrier scoreを評価
- ブレンド重みは事前に固定（OOF上で最適化しない）

使い方:
    uv run python experiments/ensemble_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from framework.config import ExperimentConfig
from framework.cv import season_cv_split
from framework.data_loader import build_tourney_matchups, load_all_data
from framework.evaluation import brier_score
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
from framework.runner import (
    _build_matchup_features,
    _call_model_factory,
    _generate_all_features,
)


def make_full_features():
    """現在のベスト特徴量セット"""
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
        CoachFeature(mode="minimal"),
    ]


def make_elo_features():
    """Eloのみの特徴量"""
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
    ]


def run_ensemble_cv(
    data, all_matchups, config,
    models_config: list[dict],
    blend_weights: list[float],
):
    """複数モデルの予測をCV内でブレンドして評価

    Args:
        models_config: [{"features": [...], "factory": callable}, ...]
        blend_weights: 各モデルの重み (合計1.0)
    """
    splits = season_cv_split(
        all_matchups, config.val_seasons, config.min_train_season
    )

    oof_records = []

    for train_matchups, val_matchups, val_season in splits:
        # 各モデルの予測を収集
        model_preds = []

        for mc in models_config:
            features = mc["features"]
            factory = mc["factory"]

            all_team_features = []
            all_feature_cols = None
            for g in ["M", "W"]:
                team_feat, feat_cols = _generate_all_features(
                    features, data, g, val_season
                )
                all_team_features.append(team_feat)
                all_feature_cols = feat_cols

            team_features = pd.concat(all_team_features, ignore_index=True)

            tr_features, diff_cols = _build_matchup_features(
                train_matchups, team_features, all_feature_cols
            )
            va_features, _ = _build_matchup_features(
                val_matchups, team_features, all_feature_cols
            )

            model = factory()

            if model.handles_nan:
                X_train = tr_features[diff_cols].reset_index(drop=True)
                y_train = train_matchups["label"].values
                X_val = va_features[diff_cols].reset_index(drop=True)
            else:
                X_train = tr_features[diff_cols].dropna()
                y_train = train_matchups["label"].values[X_train.index]
                X_train = X_train.reset_index(drop=True)
                valid_idx = va_features[diff_cols].dropna().index
                X_val = va_features[diff_cols].iloc[valid_idx].reset_index(drop=True)

            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)
            preds = np.clip(preds, config.clip_min, config.clip_max)
            model_preds.append(preds)

        # ブレンド
        y_val = val_matchups["label"].values
        va_info = val_matchups.reset_index(drop=True)
        blended = sum(w * p for w, p in zip(blend_weights, model_preds))
        blended = np.clip(blended, config.clip_min, config.clip_max)

        for i in range(len(X_val)):
            oof_records.append({
                "Season": int(va_info.iloc[i]["Season"]),
                "Gender": va_info.iloc[i]["Gender"],
                "label": int(y_val[i]),
                "pred": float(blended[i]),
                "val_season": val_season,
            })

        bs = brier_score(y_val, blended)
        print(f"    val={val_season}: Brier={bs:.4f}")

    oof_df = pd.DataFrame(oof_records)
    return oof_df


def evaluate_oof(oof_df: pd.DataFrame, name: str):
    """OOF結果を評価・表示"""
    overall = ((oof_df.pred - oof_df.label) ** 2).mean()
    men = oof_df[oof_df.Gender == "M"]
    women = oof_df[oof_df.Gender == "W"]
    bs_m = ((men.pred - men.label) ** 2).mean()
    bs_w = ((women.pred - women.label) ** 2).mean()
    print(f"  {name:<50} Brier={overall:.4f} [M={bs_m:.4f}, W={bs_w:.4f}]")
    return overall


def main():
    print("=" * 60)
    print("  アンサンブル実験")
    print("=" * 60)

    config = ExperimentConfig(name="ensemble_test", gender_mode="combined")
    data = load_all_data(config.data_dir)

    matchups_list = []
    for gender in ["M", "W"]:
        m = build_tourney_matchups(data[f"{gender}_tourney"], gender)
        matchups_list.append(m)
    all_matchups = pd.concat(matchups_list, ignore_index=True)

    # モデル定義
    full_model = {
        "features": make_full_features(),
        "factory": lambda: CalibratedModelWrapper(
            base_factory=lambda: ScaledLogisticModel(C=0.05),
            method="isotonic", cv=5,
        ),
    }

    elo_model = {
        "features": make_elo_features(),
        "factory": lambda: CalibratedModelWrapper(
            base_factory=lambda: ScaledLogisticModel(C=0.1),
            method="isotonic", cv=5,
        ),
    }

    all_results = {}

    # === 1. ベースライン (単体) ===
    print("\n--- ベースライン: フルモデル単体 ---")
    oof = run_ensemble_cv(data, all_matchups, config,
                          [full_model], [1.0])
    all_results["full_only"] = evaluate_oof(oof, "full_only")

    print("\n--- ベースライン: Elo単体 ---")
    oof = run_ensemble_cv(data, all_matchups, config,
                          [elo_model], [1.0])
    all_results["elo_only"] = evaluate_oof(oof, "elo_only")

    # === 2. 固定重みブレンド ===
    # 理論的根拠: 2モデルの相関0.964、両方に固有情報がある
    # 重みは事前に固定 (OOFで最適化しない)
    for elo_w in [0.1, 0.2, 0.3, 0.4]:
        name = f"blend_elo{elo_w:.1f}"
        print(f"\n--- ブレンド: full={1-elo_w:.1f}, elo={elo_w:.1f} ---")
        oof = run_ensemble_cv(data, all_matchups, config,
                              [full_model, elo_model],
                              [1 - elo_w, elo_w])
        all_results[name] = evaluate_oof(oof, name)

    # === 3. 男女で異なるブレンド重み ===
    # 仮説: 女子はEloの精度が高い(0.1393) → Eloの重みを大きく
    #       男子はフルモデルが優位(0.1910 vs 0.1946) → フルモデルの重みを大きく
    print("\n--- 男女別ブレンド ---")
    for elo_w_m, elo_w_w in [
        (0.0, 0.4), (0.0, 0.5), (0.0, 0.6),
        (0.1, 0.4), (0.1, 0.5), (0.1, 0.6),
        (0.2, 0.4), (0.2, 0.5), (0.2, 0.6),
    ]:
        name = f"blend_M{1-elo_w_m:.1f}_W{1-elo_w_w:.1f}"
        print(f"\n  Men: full={1-elo_w_m:.1f}/elo={elo_w_m:.1f}, "
              f"Women: full={1-elo_w_w:.1f}/elo={elo_w_w:.1f}")

        # CV内で男女別にブレンド
        splits = season_cv_split(
            all_matchups, config.val_seasons, config.min_train_season
        )
        oof_records = []
        for train_matchups, val_matchups, val_season in splits:
            model_preds_all = []
            for mc in [full_model, elo_model]:
                features = mc["features"]
                all_team_features = []
                all_feature_cols = None
                for g in ["M", "W"]:
                    team_feat, feat_cols = _generate_all_features(
                        features, data, g, val_season
                    )
                    all_team_features.append(team_feat)
                    all_feature_cols = feat_cols
                team_features = pd.concat(all_team_features, ignore_index=True)
                tr_features, diff_cols = _build_matchup_features(
                    train_matchups, team_features, all_feature_cols
                )
                va_features, _ = _build_matchup_features(
                    val_matchups, team_features, all_feature_cols
                )
                model = mc["factory"]()
                if model.handles_nan:
                    X_train = tr_features[diff_cols].reset_index(drop=True)
                    y_train = train_matchups["label"].values
                    X_val = va_features[diff_cols].reset_index(drop=True)
                else:
                    X_train = tr_features[diff_cols].dropna()
                    y_train = train_matchups["label"].values[X_train.index]
                    X_train = X_train.reset_index(drop=True)
                    valid_idx = va_features[diff_cols].dropna().index
                    X_val = va_features[diff_cols].iloc[valid_idx].reset_index(drop=True)
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_val)
                preds = np.clip(preds, config.clip_min, config.clip_max)
                model_preds_all.append(preds)

            y_val = val_matchups["label"].values
            va_info = val_matchups.reset_index(drop=True)
            genders = va_info["Gender"].values

            for i in range(len(y_val)):
                if genders[i] == "M":
                    w0, w1 = 1 - elo_w_m, elo_w_m
                else:
                    w0, w1 = 1 - elo_w_w, elo_w_w
                blended = w0 * model_preds_all[0][i] + w1 * model_preds_all[1][i]
                blended = np.clip(blended, config.clip_min, config.clip_max)
                oof_records.append({
                    "Season": int(va_info.iloc[i]["Season"]),
                    "Gender": genders[i],
                    "label": int(y_val[i]),
                    "pred": float(blended),
                    "val_season": val_season,
                })

        oof_df = pd.DataFrame(oof_records)
        all_results[name] = evaluate_oof(oof_df, name)

    # === サマリー ===
    print("\n" + "=" * 60)
    print("  結果サマリー")
    print("=" * 60)
    for name, brier in sorted(all_results.items(), key=lambda x: x[1]):
        marker = " <<<" if brier == min(all_results.values()) else ""
        print(f"  {name:<50} {brier:.4f}{marker}")

    print("\n完了")


if __name__ == "__main__":
    main()
