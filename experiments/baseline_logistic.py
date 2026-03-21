"""ベースライン実験: シード差 + ロジスティック回帰

最小限の特徴量とモデルで検証基盤の動作を確認する。
"""

import sys
from pathlib import Path

# プロジェクトルートを PYTHONPATH に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.config import ExperimentConfig
from framework.features.seed_diff import SeedDiffFeature
from framework.models.logistic import LogisticModel
from framework.runner import run_experiment


def main():
    config = ExperimentConfig(
        name="baseline_logistic",
        val_seasons=[2021, 2022, 2023, 2024, 2025],
        min_train_season=2003,
        gender_mode="both",  # combined と separate の両方を比較
        submission_stage="stage2",
    )

    feature_generators = [SeedDiffFeature()]

    results = run_experiment(
        config=config,
        feature_generators=feature_generators,
        model_factory=lambda: LogisticModel(C=1.0),
        verbose=True,
    )


if __name__ == "__main__":
    main()
