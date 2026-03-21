"""直近フォーム特徴量

レギュラーシーズン最後の N 試合の成績を特徴量とする。
"""

import pandas as pd

from framework.features.base import FeatureGenerator


class RecentFormFeature(FeatureGenerator):
    """直近 N 試合のフォーム"""

    def __init__(self, last_n: int = 10):
        self._last_n = last_n

    @property
    def name(self) -> str:
        return "recent_form"

    @property
    def feature_columns(self) -> list[str]:
        return ["recent_win_pct", "recent_avg_margin"]

    def generate(
        self,
        data: dict[str, pd.DataFrame],
        gender: str,
        max_season: int,
    ) -> pd.DataFrame:
        rs = data[f"{gender}_rs_compact"]
        rs = rs[rs["Season"] <= max_season]

        w = rs[["Season", "DayNum", "WTeamID", "WScore", "LScore"]].rename(
            columns={"WTeamID": "TeamID", "WScore": "Score", "LScore": "OppScore"}
        )
        w["Win"] = 1
        l = rs[["Season", "DayNum", "LTeamID", "LScore", "WScore"]].rename(
            columns={"LTeamID": "TeamID", "LScore": "Score", "WScore": "OppScore"}
        )
        l["Win"] = 0
        games = pd.concat([w, l]).sort_values(["Season", "TeamID", "DayNum"])

        recent = games.groupby(["Season", "TeamID"]).tail(self._last_n)
        agg = recent.groupby(["Season", "TeamID"]).agg(
            recent_wins=("Win", "sum"),
            recent_games=("Win", "count"),
            recent_score=("Score", "sum"),
            recent_opp=("OppScore", "sum"),
        ).reset_index()
        agg["recent_win_pct"] = agg["recent_wins"] / agg["recent_games"]
        agg["recent_avg_margin"] = (agg["recent_score"] - agg["recent_opp"]) / agg["recent_games"]
        return agg[["Season", "TeamID", "recent_win_pct", "recent_avg_margin"]]
