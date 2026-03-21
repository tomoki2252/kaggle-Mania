"""最終アンサンブル提出スクリプト

3モデルアンサンブル: L2 (Ridge Logistic) + L1 (Lasso Logistic) + Elo-only
男女別の固定重みでブレンド:
  Men:   L2=1.0, L1=0.0, Elo=0.0  (フルモデルのみ)
  Women: L2=0.2, L1=0.5, Elo=0.3  (L1が主力、Elo補完)

仮説と根拠:
- 男子: 豊富な特徴量(Massey, Coach等)がある → L2正則化で全特徴量活用が最適
- 女子: 特徴量が少なく過学習リスクが高い → L1の自動特徴量選択が有効
- Eloモデル: 少数特徴量で安定した予測 → 特に女子で補完効果あり

CV結果: Overall=0.1641 [M=0.1910, W=0.1369] (ベースライン0.1657から0.0016改善)

使い方:
    uv run python experiments/final_ensemble_submission.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from framework.config import ExperimentConfig
from framework.cv import season_cv_split
from framework.data_loader import (
    build_submission_matchups,
    build_tourney_matchups,
    load_all_data,
)
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
from framework.runner import _build_matchup_features, _generate_all_features

# ============================================================
# 設定
# ============================================================

# ブレンド重み (L2, L1, Elo)
WEIGHTS_M = [1.0, 0.0, 0.0]  # Men: L2のみ
WEIGHTS_W = [0.2, 0.5, 0.3]  # Women: L1主力 + Elo補完


def make_full_features():
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


# ============================================================
# モデル定義
# ============================================================

class L1CalibratedModel:
    """L1正則化 + Isotonic キャリブレーション"""

    def __init__(self, C=0.01):
        self._C = C
        self._imp = None
        self._scaler = None
        self._model = None

    @property
    def handles_nan(self):
        return True

    def fit(self, X, y):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self._imp = SimpleImputer(strategy="median")
        X_imp = self._imp.fit_transform(X_arr)
        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X_imp)
        base = LogisticRegression(
            C=self._C, solver="saga", l1_ratio=1.0, max_iter=5000,
        )
        self._model = CalibratedClassifierCV(
            estimator=base, cv=5, method="isotonic",
        )
        self._model.fit(X_sc, y)

    def predict_proba(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_imp = self._imp.transform(X_arr)
        X_sc = self._scaler.transform(X_imp)
        return self._model.predict_proba(X_sc)[:, 1]


def make_l2_model():
    return CalibratedModelWrapper(
        base_factory=lambda: ScaledLogisticModel(C=0.05),
        method="isotonic",
        cv=5,
    )


def make_l1_model():
    return L1CalibratedModel(C=0.01)


def make_elo_model():
    return CalibratedModelWrapper(
        base_factory=lambda: ScaledLogisticModel(C=0.1),
        method="isotonic",
        cv=5,
    )


# ============================================================
# CV評価
# ============================================================

def run_ensemble_cv(data, all_matchups, config):
    """3モデルアンサンブルのCV評価"""
    splits = season_cv_split(
        all_matchups, config.val_seasons, config.min_train_season
    )

    models_config = [
        {"features": make_full_features(), "factory": make_l2_model, "name": "L2"},
        {"features": make_full_features(), "factory": make_l1_model, "name": "L1"},
        {"features": make_elo_features(), "factory": make_elo_model, "name": "Elo"},
    ]

    oof_records = []

    for train_matchups, val_matchups, val_season in splits:
        model_preds = []

        for mc in models_config:
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
            X_train = tr_features[diff_cols].reset_index(drop=True)
            y_train = train_matchups["label"].values
            X_val = va_features[diff_cols].reset_index(drop=True)

            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)
            preds = np.clip(preds, config.clip_min, config.clip_max)
            model_preds.append(preds)

        # 男女別ブレンド
        y_val = val_matchups["label"].values
        va_info = val_matchups.reset_index(drop=True)
        genders = va_info["Gender"].values

        for i in range(len(y_val)):
            weights = WEIGHTS_M if genders[i] == "M" else WEIGHTS_W
            blended = sum(w * p[i] for w, p in zip(weights, model_preds))
            blended = np.clip(blended, config.clip_min, config.clip_max)
            oof_records.append({
                "Season": int(va_info.iloc[i]["Season"]),
                "Gender": genders[i],
                "label": int(y_val[i]),
                "pred": float(blended),
                "val_season": val_season,
            })

        bs = brier_score(y_val,
                         np.array([r["pred"] for r in oof_records[-len(y_val):]]))
        print(f"    val={val_season}: Brier={bs:.4f}")

    oof_df = pd.DataFrame(oof_records)
    return oof_df


# ============================================================
# Submission生成
# ============================================================

def generate_ensemble_submission(data, all_matchups, config):
    """3モデルの予測をブレンドしてsubmission生成"""
    sub_matchups = build_submission_matchups(
        config.data_dir, config.submission_stage
    )
    max_season = config.submission_season

    models_config = [
        {"features": make_full_features(), "factory": make_l2_model, "name": "L2"},
        {"features": make_full_features(), "factory": make_l1_model, "name": "L1"},
        {"features": make_elo_features(), "factory": make_elo_model, "name": "Elo"},
    ]

    all_sub_preds = []

    for mc in models_config:
        features = mc["features"]
        print(f"  Training {mc['name']} model for submission...")

        gender_preds = []
        for gender in ["M", "W"]:
            g_sub = sub_matchups[sub_matchups["Gender"] == gender]
            if len(g_sub) == 0:
                continue

            team_feat, feat_cols = _generate_all_features(
                features, data, gender, max_season
            )

            g_train = all_matchups[
                (all_matchups["Gender"] == gender)
                & (all_matchups["Season"] < max_season)
            ]
            tr_feat, diff_cols = _build_matchup_features(
                g_train, team_feat, feat_cols
            )
            sub_feat, _ = _build_matchup_features(g_sub, team_feat, feat_cols)

            model = mc["factory"]()
            X_train = tr_feat[diff_cols].reset_index(drop=True)
            y_train = g_train["label"].values
            model.fit(X_train, y_train)

            X_sub = sub_feat[diff_cols]
            if hasattr(model, "handles_nan") and model.handles_nan:
                preds = model.predict_proba(X_sub)
            else:
                has_nan = X_sub.isna().any(axis=1)
                preds = np.full(len(X_sub), 0.5)
                if (~has_nan).sum() > 0:
                    preds[~has_nan] = model.predict_proba(X_sub[~has_nan])

            preds = np.clip(preds, config.clip_min, config.clip_max)

            g_sub_copy = g_sub.copy()
            g_sub_copy["pred"] = preds
            gender_preds.append(g_sub_copy)

        model_df = pd.concat(gender_preds, ignore_index=True)
        all_sub_preds.append(model_df)

    # ブレンド
    print("  Blending predictions...")
    base_df = all_sub_preds[0][["Season", "TeamA", "TeamB", "Gender"]].copy()
    blended_preds = np.zeros(len(base_df))

    for idx in range(len(base_df)):
        gender = base_df.iloc[idx]["Gender"]
        weights = WEIGHTS_M if gender == "M" else WEIGHTS_W
        pred = sum(
            w * all_sub_preds[m].iloc[idx]["pred"]
            for m, w in enumerate(weights)
        )
        blended_preds[idx] = np.clip(pred, config.clip_min, config.clip_max)

    base_df["Pred"] = blended_preds
    base_df["ID"] = (
        base_df["Season"].astype(str) + "_"
        + base_df["TeamA"].astype(str) + "_"
        + base_df["TeamB"].astype(str)
    )

    # サンプル提出ファイルの ID 順に並べる
    filename = (
        "SampleSubmissionStage1.csv"
        if config.submission_stage == "stage1"
        else "SampleSubmissionStage2.csv"
    )
    sample = pd.read_csv(config.data_dir / filename)
    submission = sample[["ID"]].merge(base_df[["ID", "Pred"]], on="ID", how="left")
    submission["Pred"] = submission["Pred"].fillna(0.5)

    return submission


# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 60)
    print("  最終アンサンブル提出")
    print("=" * 60)
    print(f"  重み (L2, L1, Elo):")
    print(f"    Men:   {WEIGHTS_M}")
    print(f"    Women: {WEIGHTS_W}")

    config = ExperimentConfig(name="final_ensemble", gender_mode="combined")
    data = load_all_data(config.data_dir)

    matchups_list = []
    for gender in ["M", "W"]:
        m = build_tourney_matchups(data[f"{gender}_tourney"], gender)
        matchups_list.append(m)
    all_matchups = pd.concat(matchups_list, ignore_index=True)

    # === CV評価 ===
    print("\n--- CV評価 ---")
    oof_df = run_ensemble_cv(data, all_matchups, config)

    overall = ((oof_df.pred - oof_df.label) ** 2).mean()
    men = oof_df[oof_df.Gender == "M"]
    women = oof_df[oof_df.Gender == "W"]
    bs_m = ((men.pred - men.label) ** 2).mean()
    bs_w = ((women.pred - women.label) ** 2).mean()
    print(f"\n  Overall Brier: {overall:.4f} [M={bs_m:.4f}, W={bs_w:.4f}]")

    # OOF保存
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    # === Submission生成 ===
    print("\n--- Submission生成 ---")
    submission = generate_ensemble_submission(data, all_matchups, config)
    submission.to_csv(output_dir / "submission.csv", index=False)

    print(f"\n  Submission: {len(submission)} 行")
    print(f"  Pred stats: mean={submission['Pred'].mean():.4f}, "
          f"std={submission['Pred'].std():.4f}")

    # 結果保存
    results = {
        "name": "final_ensemble",
        "cv_brier": {
            "overall": round(overall, 4),
            "men": round(bs_m, 4),
            "women": round(bs_w, 4),
        },
        "weights": {
            "men": WEIGHTS_M,
            "women": WEIGHTS_W,
        },
        "models": ["L2 (ScaledLogistic C=0.05 + Isotonic)",
                    "L1 (Logistic C=0.01 + Isotonic)",
                    "Elo (ScaledLogistic C=0.1 + Isotonic)"],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  結果保存: {output_dir}/")
    print("完了")


if __name__ == "__main__":
    main()
