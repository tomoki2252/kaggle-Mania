"""コーチ特徴量 (男子のみ)

MTeamCoaches.csv を活用して、コーチの経験・安定性・実績に関する特徴量を生成する。
女子にはコーチデータが存在しないため NaN を返す（Imputer で処理される）。

リーク防止:
- max_season 以降のデータは使わない
- 各 season のトーナメント結果は使わない（レギュラーシーズン結果のみ）
- コーチのトーナメント出場回数は max_season-1 以前のトーナメント結果のみ使用
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from framework.features.base import FeatureGenerator


class CoachFeature(FeatureGenerator):
    """コーチに関する特徴量を生成する

    Args:
        mode: 特徴量セットの選択
            "full" - 全8特徴量
            "minimal" - 既存特徴量と低相関の3特徴量のみ
            "tournament" - トーナメント特化の特徴量セット
    """

    def __init__(self, mode: str = "full"):
        self._mode = mode

    @property
    def name(self) -> str:
        return f"coach_{self._mode}"

    @property
    def feature_columns(self) -> list[str]:
        if self._mode == "minimal":
            return [
                "coach_tenure",
                "coach_is_new",
                "coach_mid_season_change",
            ]
        elif self._mode == "tournament":
            return [
                "coach_tenure",
                "coach_is_new",
                "coach_mid_season_change",
                "coach_tourney_apps",
                "coach_tourney_win_rate",
                "coach_tourney_overperform",
            ]
        else:  # full
            return [
                "coach_tenure",
                "coach_season_exp",
                "coach_tourney_apps",
                "coach_tourney_wins",
                "coach_tourney_win_rate",
                "coach_is_new",
                "coach_mid_season_change",
                "coach_career_win_rate",
            ]

    def generate(
        self,
        data: dict[str, pd.DataFrame],
        gender: str,
        max_season: int,
    ) -> pd.DataFrame:
        if gender == "W" or f"{gender}_coaches" not in data:
            return self._empty_features(data, gender, max_season)

        coaches = data[f"{gender}_coaches"].copy()
        coaches = coaches[coaches["Season"] <= max_season]

        tourney = data[f"{gender}_tourney"].copy()
        tourney = tourney[tourney["Season"] < max_season]

        rs = data[f"{gender}_rs_compact"].copy()
        rs = rs[rs["Season"] <= max_season]

        # 各シーズンのトーナメントコーチ (最終コーチ = LastDayNum が最大)
        season_coach = (
            coaches.sort_values("LastDayNum", ascending=False)
            .groupby(["Season", "TeamID"])
            .first()
            .reset_index()[["Season", "TeamID", "CoachName", "FirstDayNum"]]
        )

        # mid-season change
        coach_count = coaches.groupby(["Season", "TeamID"]).size().reset_index(name="n_coaches")
        season_coach = season_coach.merge(coach_count, on=["Season", "TeamID"], how="left")
        season_coach["coach_mid_season_change"] = (season_coach["n_coaches"] > 1).astype(int)

        # --- tenure: 同一チームでの連続在任年数 ---
        sc_sorted = season_coach.sort_values(["TeamID", "CoachName", "Season"])
        sc_sorted["prev_season"] = sc_sorted.groupby(["TeamID", "CoachName"])["Season"].shift(1)
        sc_sorted["is_consecutive"] = (sc_sorted["Season"] - sc_sorted["prev_season"] == 1).astype(int)

        tenures = []
        for _, group in sc_sorted.groupby(["TeamID", "CoachName"]):
            t = 0
            for _, row in group.iterrows():
                if row["is_consecutive"] == 1:
                    t += 1
                else:
                    t = 0
                tenures.append({"Season": row["Season"], "TeamID": row["TeamID"], "coach_tenure": t})
        tenure_df = pd.DataFrame(tenures)
        season_coach = season_coach.merge(tenure_df, on=["Season", "TeamID"], how="left")
        season_coach["coach_is_new"] = (season_coach["coach_tenure"] == 0).astype(int)

        if self._mode == "minimal":
            return season_coach[["Season", "TeamID"] + self.feature_columns].copy()

        # --- coach_season_exp: 累積シーズン経験 ---
        sc_sorted2 = season_coach.sort_values(["CoachName", "Season"])
        sc_sorted2["coach_season_exp"] = sc_sorted2.groupby("CoachName").cumcount()
        season_coach = season_coach.merge(
            sc_sorted2[["Season", "TeamID", "coach_season_exp"]],
            on=["Season", "TeamID"], how="left",
        )

        # --- トーナメント出場・勝利 (season < max_season) ---
        tourney_w = tourney[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"})
        tourney_l = tourney[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"})
        tourney_teams = pd.concat([tourney_w, tourney_l]).drop_duplicates()
        tourney_teams["in_tourney"] = 1

        sc_with_tourney = season_coach.merge(
            tourney_teams, on=["Season", "TeamID"], how="left"
        )
        sc_with_tourney["in_tourney"] = sc_with_tourney["in_tourney"].fillna(0).astype(int)

        tourney_win_counts = tourney.groupby(["Season", "WTeamID"]).size().reset_index(name="t_wins")
        tourney_win_counts.rename(columns={"WTeamID": "TeamID"}, inplace=True)
        sc_with_tourney = sc_with_tourney.merge(
            tourney_win_counts, on=["Season", "TeamID"], how="left"
        )
        sc_with_tourney["t_wins"] = sc_with_tourney["t_wins"].fillna(0).astype(int)

        # 累積トーナメント出場・勝利 (自分より前のシーズンのみ)
        sc_with_tourney = sc_with_tourney.sort_values(["CoachName", "Season"])
        sc_with_tourney["coach_tourney_apps"] = (
            sc_with_tourney.groupby("CoachName")["in_tourney"].cumsum()
            - sc_with_tourney["in_tourney"]
        )
        sc_with_tourney["coach_tourney_wins"] = (
            sc_with_tourney.groupby("CoachName")["t_wins"].cumsum()
            - sc_with_tourney["t_wins"]
        )
        sc_with_tourney["coach_tourney_win_rate"] = np.where(
            sc_with_tourney["coach_tourney_apps"] > 0,
            sc_with_tourney["coach_tourney_wins"] / sc_with_tourney["coach_tourney_apps"],
            0.0,
        )

        if self._mode == "tournament":
            # --- tourney overperform: トーナメント勝利数 - シード期待勝利数 ---
            seeds = data[f"{gender}_seeds"].copy()
            seeds = seeds[seeds["Season"] < max_season]
            seeds["seed_num"] = seeds["Seed"].apply(lambda s: int(s[1:3]))
            # シードからの期待勝利数 (近似: 1st→3.3, 2nd→2.4, ..., 16th→0.1)
            seeds["expected_wins"] = np.clip(3.5 - seeds["seed_num"] * 0.22, 0, 6)

            sc_with_seeds = sc_with_tourney.merge(
                seeds[["Season", "TeamID", "expected_wins"]],
                on=["Season", "TeamID"], how="left",
            )
            sc_with_seeds["expected_wins"] = sc_with_seeds["expected_wins"].fillna(0)
            sc_with_seeds["overperform"] = sc_with_seeds["t_wins"] - sc_with_seeds["expected_wins"]

            sc_with_seeds = sc_with_seeds.sort_values(["CoachName", "Season"])
            sc_with_seeds["cum_overperform"] = (
                sc_with_seeds.groupby("CoachName")["overperform"].cumsum()
                - sc_with_seeds["overperform"]
            )
            sc_with_seeds["cum_t_apps"] = (
                sc_with_seeds.groupby("CoachName")["in_tourney"].cumsum()
                - sc_with_seeds["in_tourney"]
            )
            sc_with_seeds["coach_tourney_overperform"] = np.where(
                sc_with_seeds["cum_t_apps"] > 0,
                sc_with_seeds["cum_overperform"] / sc_with_seeds["cum_t_apps"],
                0.0,
            )

            season_coach = season_coach.merge(
                sc_with_seeds[["Season", "TeamID", "coach_tourney_apps",
                               "coach_tourney_win_rate", "coach_tourney_overperform"]],
                on=["Season", "TeamID"], how="left",
            )
            return season_coach[["Season", "TeamID"] + self.feature_columns].copy()

        # --- career_win_rate (レギュラーシーズン) ---
        wins = rs.groupby(["Season", "WTeamID"]).size().reset_index(name="w")
        wins.rename(columns={"WTeamID": "TeamID"}, inplace=True)
        losses = rs.groupby(["Season", "LTeamID"]).size().reset_index(name="l")
        losses.rename(columns={"LTeamID": "TeamID"}, inplace=True)

        team_record = wins.merge(losses, on=["Season", "TeamID"], how="outer").fillna(0)
        team_record["w"] = team_record["w"].astype(int)
        team_record["l"] = team_record["l"].astype(int)

        sc_with_record = sc_with_tourney.merge(
            team_record, on=["Season", "TeamID"], how="left"
        )
        sc_with_record["w"] = sc_with_record["w"].fillna(0)
        sc_with_record["l"] = sc_with_record["l"].fillna(0)

        sc_with_record = sc_with_record.sort_values(["CoachName", "Season"])
        sc_with_record["cum_w"] = (
            sc_with_record.groupby("CoachName")["w"].cumsum() - sc_with_record["w"]
        )
        sc_with_record["cum_g"] = (
            sc_with_record.groupby("CoachName")[["w", "l"]].transform("cumsum").sum(axis=1)
            - sc_with_record["w"] - sc_with_record["l"]
        )
        sc_with_record["coach_career_win_rate"] = np.where(
            sc_with_record["cum_g"] > 0,
            sc_with_record["cum_w"] / sc_with_record["cum_g"],
            0.5,
        )

        season_coach = season_coach.merge(
            sc_with_record[["Season", "TeamID", "coach_tourney_apps", "coach_tourney_wins",
                            "coach_tourney_win_rate", "coach_career_win_rate"]],
            on=["Season", "TeamID"], how="left",
        )

        return season_coach[["Season", "TeamID"] + self.feature_columns].copy()

    def _empty_features(
        self, data: dict[str, pd.DataFrame], gender: str, max_season: int
    ) -> pd.DataFrame:
        """女子用: 全チーム・全シーズンに NaN を返す"""
        teams = data[f"{gender}_teams"]["TeamID"].unique()
        seasons = range(1998, max_season + 1)
        rows = [
            {"Season": s, "TeamID": t, **{c: np.nan for c in self.feature_columns}}
            for s in seasons
            for t in teams
        ]
        return pd.DataFrame(rows)
