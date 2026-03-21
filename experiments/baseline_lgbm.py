"""LightGBM ベースライン実験

Elo + シーズン統計 + シード + Massey + 直近フォーム + SOS + カンファレンス強度
元の baseline/train.py に相当。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.config import ExperimentConfig
from framework.features.elo import EloFeature, EloParams
from framework.features.season_stats import SeasonStatsFeature
from framework.features.seed_diff import SeedDiffFeature
from framework.features.massey import MasseyFeature
from framework.features.recent_form import RecentFormFeature
from framework.models.lightgbm_model import LightGBMModel
from framework.runner import run_experiment


def add_matchup_features(df, diff_cols):
    """対戦ペアにカスタム特徴量を追加"""
    new_cols = list(diff_cols)

    # Elo勝率
    if "diff_elo" in df.columns:
        df["elo_win_prob"] = 1.0 / (1.0 + 10.0 ** (-df["diff_elo"] / 400.0))
        new_cols.append("elo_win_prob")

    # シード交互作用
    if "diff_seed_num" in df.columns:
        # A_seed_num, B_seed_num は既に削除済みなので diff から復元
        # seed_product = A * B, seed_sum = A + B は diff だけからは計算できない
        # → Gender フラグだけ追加
        pass

    return df, new_cols


def main():
    config = ExperimentConfig(
        name="baseline_lgbm",
        val_seasons=[2021, 2022, 2023, 2024, 2025],
        min_train_season=2003,
        gender_mode="both",
    )

    # Elo (SOS + カンファレンス強度も含む)
    elo_params = EloParams(
        K=32.0, carryover=0.75, home_advantage=100.0,
        use_mov=False, use_conf_regression=False,
    )

    feature_generators = [
        EloFeature(params=elo_params, compute_sos=True, compute_conf_strength=True),
        SeasonStatsFeature(use_detailed=True),
        SeedDiffFeature(),
        MasseyFeature(),
        RecentFormFeature(last_n=10),
    ]

    results = run_experiment(
        config=config,
        feature_generators=feature_generators,
        model_factory=lambda: LightGBMModel(
            num_boost_round=2000,
            n_ensemble=5,
            seed=42,
        ),
        verbose=True,
        matchup_feature_fn=add_matchup_features,
    )


if __name__ == "__main__":
    main()
