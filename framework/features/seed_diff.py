"""シード差特徴量 - ベースライン用の最小限の特徴量"""

import pandas as pd

from framework.features.base import FeatureGenerator


class SeedDiffFeature(FeatureGenerator):
    """NCAAトーナメントのシード番号を特徴量として提供する

    リーク防止: シード情報はトーナメント開始前に公開される情報のため、
    その season のシードを使うのはリークにならない。
    ただし max_season より後の season のデータは使わない。
    """

    @property
    def name(self) -> str:
        return "seed"

    @property
    def feature_columns(self) -> list[str]:
        return ["seed_num"]

    def generate(
        self,
        data: dict[str, pd.DataFrame],
        gender: str,
        max_season: int,
    ) -> pd.DataFrame:
        seeds = data[f"{gender}_seeds"].copy()
        seeds = seeds[seeds["Season"] <= max_season]
        seeds["seed_num"] = seeds["Seed"].apply(lambda s: int(s[1:3]))
        return seeds[["Season", "TeamID", "seed_num"]]
