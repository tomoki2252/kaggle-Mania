"""Eloレーティング 最適化版実験

Optuna + season-based CV で最適化済みのパラメータ (男女別) を使用するEloモデル。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.config import ExperimentConfig
from framework.features.elo import (
    EloFeature,
    ELO_PARAMS_OPTIMIZED_M,
    ELO_PARAMS_OPTIMIZED_W,
    ELO_SCALE_OPTIMIZED_M,
    ELO_SCALE_OPTIMIZED_W,
)
from framework.models.elo_predictor import EloPredictor
from framework.runner import run_experiment


def main():
    config = ExperimentConfig(
        name="elo_optimized",
        val_seasons=[2021, 2022, 2023, 2024, 2025],
        min_train_season=2003,
        gender_mode="separate",
    )

    feature_generators = [
        EloFeature(
            params_by_gender={
                "M": ELO_PARAMS_OPTIMIZED_M,
                "W": ELO_PARAMS_OPTIMIZED_W,
            },
        ),
    ]

    # 男女別に最適化された elo_scale を適用
    scale_map = {"M": ELO_SCALE_OPTIMIZED_M, "W": ELO_SCALE_OPTIMIZED_W}

    results = run_experiment(
        config=config,
        feature_generators=feature_generators,
        model_factory=lambda gender="M": EloPredictor(elo_scale=scale_map.get(gender, 1.0)),
        verbose=True,
    )


if __name__ == "__main__":
    main()
