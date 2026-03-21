"""シーズン統計特徴量

DetailedResults からの詳細統計と CompactResults からの基本統計を提供する。
DetailedResults がないシーズン/チームは CompactResults でフォールバック。
"""

import numpy as np
import pandas as pd

from framework.features.base import FeatureGenerator


# DetailedResults で計算する特徴量
_DETAIL_COLS = [
    "win_pct", "ppg", "opp_ppg", "margin",
    "fg_pct", "fg3_pct", "ft_pct",
    "off_eff", "def_eff", "net_eff",
    "or_pct", "dr_pct",
    "ast_pg", "to_pg", "stl_pg", "blk_pg",
    "opp_fg_pct", "opp_fg3_pct",
    "pace", "pf_pg", "score_std", "margin_std",
    "ast_to_ratio", "efg_pct", "ts_pct",
]

# CompactResults で計算する特徴量 (サブセット)
_COMPACT_COLS = ["win_pct", "ppg", "opp_ppg", "margin", "score_std", "margin_std"]


def _compute_from_detailed(rs_detail: pd.DataFrame) -> pd.DataFrame:
    """DetailedResults からシーズン統計を計算"""
    w_cols = {
        "Season": "Season", "WTeamID": "TeamID",
        "WScore": "Score", "LScore": "OppScore",
        "WFGM": "FGM", "WFGA": "FGA", "WFGM3": "FGM3", "WFGA3": "FGA3",
        "WFTM": "FTM", "WFTA": "FTA", "WOR": "OR", "WDR": "DR",
        "WAst": "Ast", "WTO": "TO", "WStl": "Stl", "WBlk": "Blk", "WPF": "PF",
        "LFGM": "OppFGM", "LFGA": "OppFGA", "LFGM3": "OppFGM3", "LFGA3": "OppFGA3",
        "LFTM": "OppFTM", "LFTA": "OppFTA", "LOR": "OppOR", "LDR": "OppDR",
    }
    l_cols = {
        "Season": "Season", "LTeamID": "TeamID",
        "LScore": "Score", "WScore": "OppScore",
        "LFGM": "FGM", "LFGA": "FGA", "LFGM3": "FGM3", "LFGA3": "FGA3",
        "LFTM": "FTM", "LFTA": "FTA", "LOR": "OR", "LDR": "DR",
        "LAst": "Ast", "LTO": "TO", "LStl": "Stl", "LBlk": "Blk", "LPF": "PF",
        "WFGM": "OppFGM", "WFGA": "OppFGA", "WFGM3": "OppFGM3", "WFGA3": "OppFGA3",
        "WFTM": "OppFTM", "WFTA": "OppFTA", "WOR": "OppOR", "WDR": "OppDR",
    }

    w_df = rs_detail[list(w_cols.keys())].rename(columns=w_cols)
    w_df["Win"] = 1
    l_df = rs_detail[list(l_cols.keys())].rename(columns=l_cols)
    l_df["Win"] = 0
    games = pd.concat([w_df, l_df], ignore_index=True)

    games["Poss"] = games["FGA"] - games["OR"] + games["TO"] + 0.475 * games["FTA"]
    games["Margin"] = games["Score"] - games["OppScore"]

    agg = games.groupby(["Season", "TeamID"]).agg(
        games_played=("Win", "count"), wins=("Win", "sum"),
        total_score=("Score", "sum"), total_opp_score=("OppScore", "sum"),
        total_margin=("Margin", "sum"),
        total_fgm=("FGM", "sum"), total_fga=("FGA", "sum"),
        total_fgm3=("FGM3", "sum"), total_fga3=("FGA3", "sum"),
        total_ftm=("FTM", "sum"), total_fta=("FTA", "sum"),
        total_or=("OR", "sum"), total_dr=("DR", "sum"),
        total_ast=("Ast", "sum"), total_to=("TO", "sum"),
        total_stl=("Stl", "sum"), total_blk=("Blk", "sum"),
        total_pf=("PF", "sum"), total_poss=("Poss", "sum"),
        total_opp_fgm=("OppFGM", "sum"), total_opp_fga=("OppFGA", "sum"),
        total_opp_fgm3=("OppFGM3", "sum"), total_opp_fga3=("OppFGA3", "sum"),
        total_opp_or=("OppOR", "sum"), total_opp_dr=("OppDR", "sum"),
        score_std=("Score", "std"), margin_std=("Margin", "std"),
    ).reset_index()

    g = agg["games_played"]
    poss = agg["total_poss"].replace(0, np.nan)

    return pd.DataFrame({
        "Season": agg["Season"], "TeamID": agg["TeamID"],
        "win_pct": agg["wins"] / g,
        "ppg": agg["total_score"] / g,
        "opp_ppg": agg["total_opp_score"] / g,
        "margin": agg["total_margin"] / g,
        "fg_pct": agg["total_fgm"] / agg["total_fga"].replace(0, np.nan),
        "fg3_pct": agg["total_fgm3"] / agg["total_fga3"].replace(0, np.nan),
        "ft_pct": agg["total_ftm"] / agg["total_fta"].replace(0, np.nan),
        "off_eff": agg["total_score"] / poss * 100,
        "def_eff": agg["total_opp_score"] / poss * 100,
        "net_eff": agg["total_margin"] / poss * 100,
        "or_pct": agg["total_or"] / (agg["total_or"] + agg["total_opp_dr"]).replace(0, np.nan),
        "dr_pct": agg["total_dr"] / (agg["total_dr"] + agg["total_opp_or"]).replace(0, np.nan),
        "ast_pg": agg["total_ast"] / g,
        "to_pg": agg["total_to"] / g,
        "stl_pg": agg["total_stl"] / g,
        "blk_pg": agg["total_blk"] / g,
        "opp_fg_pct": agg["total_opp_fgm"] / agg["total_opp_fga"].replace(0, np.nan),
        "opp_fg3_pct": agg["total_opp_fgm3"] / agg["total_opp_fga3"].replace(0, np.nan),
        "pace": agg["total_poss"] / g,
        "pf_pg": agg["total_pf"] / g,
        "score_std": agg["score_std"],
        "margin_std": agg["margin_std"],
        "ast_to_ratio": agg["total_ast"] / agg["total_to"].replace(0, np.nan),
        "efg_pct": (agg["total_fgm"] + 0.5 * agg["total_fgm3"]) / agg["total_fga"].replace(0, np.nan),
        "ts_pct": agg["total_score"] / (2 * (agg["total_fga"] + 0.44 * agg["total_fta"])).replace(0, np.nan),
    })


def _compute_from_compact(rs_compact: pd.DataFrame) -> pd.DataFrame:
    """CompactResults からシーズン基本統計を計算"""
    w = rs_compact[["Season", "WTeamID", "WScore", "LScore"]].rename(
        columns={"WTeamID": "TeamID", "WScore": "Score", "LScore": "OppScore"})
    w["Win"] = 1
    l = rs_compact[["Season", "LTeamID", "LScore", "WScore"]].rename(
        columns={"LTeamID": "TeamID", "LScore": "Score", "WScore": "OppScore"})
    l["Win"] = 0
    games = pd.concat([w, l], ignore_index=True)
    games["Margin"] = games["Score"] - games["OppScore"]

    agg = games.groupby(["Season", "TeamID"]).agg(
        games_played=("Win", "count"), wins=("Win", "sum"),
        total_score=("Score", "sum"), total_opp_score=("OppScore", "sum"),
        score_std=("Score", "std"), margin_std=("Margin", "std"),
    ).reset_index()

    g = agg["games_played"]
    return pd.DataFrame({
        "Season": agg["Season"], "TeamID": agg["TeamID"],
        "win_pct": agg["wins"] / g,
        "ppg": agg["total_score"] / g,
        "opp_ppg": agg["total_opp_score"] / g,
        "margin": (agg["total_score"] - agg["total_opp_score"]) / g,
        "score_std": agg["score_std"],
        "margin_std": agg["margin_std"],
    })


class SeasonStatsFeature(FeatureGenerator):
    """シーズン統計特徴量

    DetailedResults がある場合は詳細統計、なければ CompactResults からの基本統計を使用。
    """

    def __init__(self, use_detailed: bool = True):
        self._use_detailed = use_detailed

    @property
    def name(self) -> str:
        return "season_stats"

    @property
    def feature_columns(self) -> list[str]:
        return _DETAIL_COLS if self._use_detailed else _COMPACT_COLS

    def generate(
        self,
        data: dict[str, pd.DataFrame],
        gender: str,
        max_season: int,
    ) -> pd.DataFrame:
        rs_compact = data[f"{gender}_rs_compact"]
        rs_compact = rs_compact[rs_compact["Season"] <= max_season]

        if self._use_detailed and f"{gender}_rs_detail" in data:
            rs_detail = data[f"{gender}_rs_detail"]
            rs_detail = rs_detail[rs_detail["Season"] <= max_season]

            detail_stats = _compute_from_detailed(rs_detail)
            compact_stats = _compute_from_compact(rs_compact)

            # DetailedResults にないチーム/シーズンは CompactResults でフォールバック
            detail_keys = set(zip(detail_stats["Season"], detail_stats["TeamID"]))
            compact_only = compact_stats[
                compact_stats.apply(
                    lambda r: (r["Season"], r["TeamID"]) not in detail_keys, axis=1
                )
            ]
            return pd.concat([detail_stats, compact_only], ignore_index=True)
        else:
            return _compute_from_compact(rs_compact)
