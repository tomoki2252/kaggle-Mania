"""Eloレーティング ベースライン実験

固定パラメータ (K=32, carryover=0.75, home=100, MOVなし) でのEloモデル。
元の elo_model/train.py に相当。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.config import ExperimentConfig
from framework.features.elo import EloFeature, EloParams
from framework.models.elo_predictor import EloPredictor
from framework.runner import run_experiment


def main():
    config = ExperimentConfig(
        name="elo_baseline",
        val_seasons=[2021, 2022, 2023, 2024, 2025],
        min_train_season=2003,
        gender_mode="both",
    )

    # ベースラインパラメータ (元の elo_model/train.py と同じ)
    feature_generators = [
        EloFeature(params=EloParams(
            K=32.0, carryover=0.75, home_advantage=100.0,
            use_mov=False, use_conf_regression=False,
        )),
    ]

    results = run_experiment(
        config=config,
        feature_generators=feature_generators,
        model_factory=lambda: EloPredictor(elo_scale=1.0),
        verbose=True,
    )


if __name__ == "__main__":
    main()
