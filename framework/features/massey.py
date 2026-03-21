"""Massey Ordinals 特徴量 (男子のみ)

複数のランキングシステムの最終日の順位を集約して特徴量とする。
"""

import numpy as np
import pandas as pd

from framework.features.base import FeatureGenerator

TOP_MASSEY_SYSTEMS = ["POM", "SAG", "MOR", "WLK", "DOL", "COL", "RTH", "WOL", "CNG", "MB"]
INDIVIDUAL_SYSTEMS = ["POM", "SAG", "MOR"]


class MasseyFeature(FeatureGenerator):
    """Massey Ordinals 特徴量 (男子のみ)

    女子にはデータが存在しないため、女子の場合は空のDataFrameを返す。
    """

    @property
    def name(self) -> str:
        return "massey"

    @property
    def feature_columns(self) -> list[str]:
        cols = [
            "massey_top_mean", "massey_top_median",
            "massey_top_min", "massey_top_max", "massey_top_std",
            "massey_all_mean", "massey_all_median",
        ]
        for sys_name in INDIVIDUAL_SYSTEMS:
            cols.append(f"massey_{sys_name.lower()}")
        return cols

    def generate(
        self,
        data: dict[str, pd.DataFrame],
        gender: str,
        max_season: int,
    ) -> pd.DataFrame:
        # 女子にはデータなし
        if gender != "M" or "M_massey" not in data:
            return pd.DataFrame(
                columns=["Season", "TeamID"] + self.feature_columns
            )

        massey = data["M_massey"]
        massey = massey[massey["Season"] <= max_season]

        # 各システムの最終日のランキングを取得
        last_day = massey.groupby(["Season", "SystemName"])["RankingDayNum"].max().reset_index()
        last_day.columns = ["Season", "SystemName", "MaxDay"]
        massey_last = massey.merge(
            last_day,
            left_on=["Season", "SystemName", "RankingDayNum"],
            right_on=["Season", "SystemName", "MaxDay"],
        )

        # トップシステムの集約
        top = massey_last[massey_last["SystemName"].isin(TOP_MASSEY_SYSTEMS)]
        top_agg = top.groupby(["Season", "TeamID"])["OrdinalRank"].agg(
            massey_top_mean="mean",
            massey_top_median="median",
            massey_top_min="min",
            massey_top_max="max",
            massey_top_std="std",
        ).reset_index()

        all_agg = massey_last.groupby(["Season", "TeamID"])["OrdinalRank"].agg(
            massey_all_mean="mean",
            massey_all_median="median",
        ).reset_index()

        result = top_agg.merge(all_agg, on=["Season", "TeamID"], how="outer")

        # 個別システム
        for sys_name in INDIVIDUAL_SYSTEMS:
            sys_data = massey_last[massey_last["SystemName"] == sys_name][
                ["Season", "TeamID", "OrdinalRank"]
            ]
            sys_data = sys_data.rename(columns={"OrdinalRank": f"massey_{sys_name.lower()}"})
            result = result.merge(sys_data, on=["Season", "TeamID"], how="left")

        return result
