"""実験設定を管理するデータクラス"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    """実験の設定を保持するクラス"""

    # 実験名 (outputs/<name>/ に結果が保存される)
    name: str

    # 検証設定
    val_seasons: list[int] = field(default_factory=lambda: [2021, 2022, 2023, 2024, 2025])
    # 学習に使う最初のシーズン (それ以前のデータは除外)
    min_train_season: int = 2003

    # 男女モデル設定: "combined", "separate", "both"
    gender_mode: str = "both"

    # submission生成用シーズン
    submission_season: int = 2026

    # submission用サンプルファイル ("stage1" or "stage2")
    submission_stage: str = "stage2"

    # 予測値のクリッピング
    clip_min: float = 0.02
    clip_max: float = 0.98

    # パス
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    output_base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs")

    @property
    def output_dir(self) -> Path:
        return self.output_base_dir / self.name

    def train_seasons(self, val_season: int) -> list[int]:
        """指定されたバリデーションシーズンより前の学習用シーズンを返す"""
        return list(range(self.min_train_season, val_season))
