"""Eloレーティング特徴量

全試合を時系列処理してチームのEloレーティングを計算する。
RS + セカンダリ + カンファレンストーナメント + NCAAトーナメントの全試合を使用。
各シーズンのNCAAトーナメント直前のスナップショットを特徴量として返す。

オプションで SOS (Strength of Schedule) とカンファレンス強度も計算する。
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from framework.features.base import FeatureGenerator

INITIAL_ELO = 1500.0


@dataclass
class EloParams:
    """Eloモデルのハイパーパラメータ"""

    K: float = 32.0
    carryover: float = 0.75
    home_advantage: float = 100.0
    use_mov: bool = False
    mov_power: float = 0.7
    use_conf_regression: bool = False
    k_boost_factor: float = 1.0
    k_boost_ramp: float = 40.0
    tourney_k_mult: float = 1.0
    tourney_elo_scale: float = 1.0


# 最適化済みパラメータ (Optuna 300 trials, season-based CV で探索)
ELO_PARAMS_OPTIMIZED_M = EloParams(
    K=25.1, carryover=0.631, home_advantage=105.8,
    use_mov=False, mov_power=1.479,
    use_conf_regression=True,
    k_boost_factor=1.082, k_boost_ramp=34.3,
    tourney_k_mult=0.501,
)
# EloPredictor elo_scale=1.185

ELO_PARAMS_OPTIMIZED_W = EloParams(
    K=10.1, carryover=0.588, home_advantage=179.2,
    use_mov=True, mov_power=1.165,
    use_conf_regression=True,
    k_boost_factor=1.324, k_boost_ramp=24.2,
    tourney_k_mult=0.500,
)
# EloPredictor elo_scale=1.387

# 男女別の elo_scale (EloPredictor に渡す)
ELO_SCALE_OPTIMIZED_M = 1.185
ELO_SCALE_OPTIMIZED_W = 1.387


def _load_all_games(data: dict[str, pd.DataFrame], gender: str) -> pd.DataFrame:
    """全試合を時系列順にソートして返す"""
    rs = data[f"{gender}_rs_compact"].copy()
    rs["GameType"] = "Regular"

    tourney = data[f"{gender}_tourney"].copy()
    tourney["GameType"] = "NCAA"

    frames = [rs, tourney]

    # セカンダリトーナメント (存在する場合)
    sec_key = f"{gender}_secondary"
    if sec_key in data:
        sec = data[sec_key].copy()
        if "SecondaryTourney" in sec.columns:
            sec = sec.drop(columns=["SecondaryTourney"])
        sec["GameType"] = "Secondary"
        frames.append(sec)

    # カンファレンストーナメント (存在する場合)
    conf_tourney_key = f"{gender}_conf_tourney"
    if conf_tourney_key in data:
        conf = data[conf_tourney_key].copy()
        conf["WScore"] = 0
        conf["LScore"] = 0
        conf["WLoc"] = "N"
        conf["NumOT"] = 0
        conf["GameType"] = "ConfTourney"
        frames.append(conf)

    all_games = pd.concat(frames, ignore_index=True)
    type_order = {"Regular": 0, "ConfTourney": 1, "Secondary": 2, "NCAA": 3}
    all_games["_order"] = all_games["GameType"].map(type_order)
    all_games = all_games.sort_values(
        ["Season", "DayNum", "_order", "WTeamID"]
    ).reset_index(drop=True)
    return all_games


def _compute_elo_snapshots(
    games: pd.DataFrame,
    params: EloParams,
    conferences: dict | None = None,
) -> dict[int, dict[int, float]]:
    """全試合を処理し、各シーズンのNCAAトーナメント直前のEloスナップショットを返す"""
    elo: dict[int, float] = {}
    snapshots: dict[int, dict[int, float]] = {}
    prev_season = -1

    for _, row in games.iterrows():
        season = int(row["Season"])
        day_num = row["DayNum"]
        w_team = int(row["WTeamID"])
        l_team = int(row["LTeamID"])
        w_score = row["WScore"]
        l_score = row["LScore"]
        w_loc = row.get("WLoc", "N")
        game_type = row.get("GameType", "Regular")

        # 新シーズン開始: 平均への回帰
        if season != prev_season:
            if prev_season > 0:
                if params.use_conf_regression and conferences is not None:
                    conf_teams: dict[str, list[float]] = defaultdict(list)
                    for tid, rating in elo.items():
                        conf = conferences.get((season, tid))
                        if conf is not None:
                            conf_teams[conf].append(rating)
                    conf_avg = {
                        c: sum(r) / len(r) for c, r in conf_teams.items()
                    }
                    for tid in elo:
                        conf = conferences.get((season, tid))
                        target = conf_avg.get(conf, INITIAL_ELO) if conf else INITIAL_ELO
                        elo[tid] = params.carryover * elo[tid] + (1 - params.carryover) * target
                else:
                    for tid in elo:
                        elo[tid] = params.carryover * elo[tid] + (1 - params.carryover) * INITIAL_ELO
            prev_season = season
            snapshot_taken = False

        # NCAAトーナメント直前にスナップショット
        if not snapshot_taken and game_type == "NCAA":
            snapshots[season] = dict(elo)
            snapshot_taken = True

        # 初出チーム
        elo.setdefault(w_team, INITIAL_ELO)
        elo.setdefault(l_team, INITIAL_ELO)

        w_elo = elo[w_team]
        l_elo = elo[l_team]

        # 期待勝率 (ホーム補正込み)
        ha = params.home_advantage
        if w_loc == "H":
            w_exp = 1.0 / (1.0 + 10.0 ** ((l_elo - w_elo - ha) / 400.0))
        elif w_loc == "A":
            w_exp = 1.0 / (1.0 + 10.0 ** ((l_elo + ha - w_elo) / 400.0))
        else:
            w_exp = 1.0 / (1.0 + 10.0 ** ((l_elo - w_elo) / 400.0))

        # 実効K値
        k_eff = params.K
        if params.k_boost_factor > 1.0 and day_num < params.k_boost_ramp:
            ratio = day_num / params.k_boost_ramp
            k_eff *= params.k_boost_factor - (params.k_boost_factor - 1.0) * ratio
        if game_type == "NCAA" and params.tourney_k_mult != 1.0:
            k_eff *= params.tourney_k_mult

        # Elo更新
        if params.use_mov and w_score > 0 and l_score > 0:
            mov = w_score - l_score
            elo_diff = w_elo - l_elo
            mov_mult = ((mov + 2.5) ** params.mov_power) / (6.0 + 0.006 * abs(elo_diff))
            update = k_eff * mov_mult * (1.0 - w_exp)
        else:
            update = k_eff * (1.0 - w_exp)

        elo[w_team] = w_elo + update
        elo[l_team] = l_elo - update

    # 最終シーズンのスナップショット
    if prev_season not in snapshots:
        snapshots[prev_season] = dict(elo)

    return snapshots


class EloFeature(FeatureGenerator):
    """Eloレーティング特徴量

    オプションで SOS と カンファレンス強度も計算する。

    リーク防止: 各シーズンのNCAAトーナメント直前のスナップショットを使用。
    そのシーズンのトーナメント結果はスナップショットに含まれない。
    ただし過去シーズンのトーナメント結果はEloのキャリーオーバーに反映される。
    """

    def __init__(
        self,
        params: EloParams | None = None,
        params_by_gender: dict[str, EloParams] | None = None,
        compute_sos: bool = False,
        compute_conf_strength: bool = False,
    ):
        self._params = params or EloParams()
        self._params_by_gender = params_by_gender
        self._compute_sos = compute_sos
        self._compute_conf_strength = compute_conf_strength
        self._snapshots_cache: dict[str, dict[int, dict[int, float]]] = {}

    @property
    def name(self) -> str:
        return "elo"

    @property
    def feature_columns(self) -> list[str]:
        cols = ["elo"]
        if self._compute_sos:
            cols.extend(["sos", "sos_std"])
        if self._compute_conf_strength:
            cols.extend(["conf_elo_mean", "conf_elo_median"])
        return cols

    def _get_params(self, gender: str) -> EloParams:
        if self._params_by_gender and gender in self._params_by_gender:
            return self._params_by_gender[gender]
        return self._params

    def generate(
        self,
        data: dict[str, pd.DataFrame],
        gender: str,
        max_season: int,
    ) -> pd.DataFrame:
        params = self._get_params(gender)

        # キャッシュキー
        cache_key = f"{gender}_{id(params)}"
        if cache_key not in self._snapshots_cache:
            # カンファレンス情報
            conferences = None
            if params.use_conf_regression:
                conf_df = data.get(f"{gender}_conf")
                if conf_df is not None:
                    conferences = {
                        (int(r["Season"]), int(r["TeamID"])): r["ConfAbbrev"]
                        for _, r in conf_df.iterrows()
                    }

            games = _load_all_games(data, gender)
            self._snapshots_cache[cache_key] = _compute_elo_snapshots(
                games, params, conferences
            )

        snapshots = self._snapshots_cache[cache_key]

        # max_season 以前のスナップショットからチーム特徴量を構築
        records = []
        for season, snap in snapshots.items():
            if season > max_season:
                continue
            for team_id, elo_val in snap.items():
                records.append({
                    "Season": season,
                    "TeamID": team_id,
                    "elo": elo_val,
                })

        if not records:
            return pd.DataFrame(columns=["Season", "TeamID"] + self.feature_columns)

        result = pd.DataFrame(records)

        # SOS (Strength of Schedule)
        if self._compute_sos:
            result = self._add_sos(result, data, gender, max_season)

        # Conference Strength
        if self._compute_conf_strength:
            result = self._add_conf_strength(result, data, gender, max_season)

        return result

    def _add_sos(
        self, elo_df: pd.DataFrame, data: dict, gender: str, max_season: int
    ) -> pd.DataFrame:
        rs = data[f"{gender}_rs_compact"]
        rs = rs[rs["Season"] <= max_season]

        w = rs[["Season", "WTeamID", "LTeamID"]].rename(
            columns={"WTeamID": "TeamID", "LTeamID": "OppID"}
        )
        l = rs[["Season", "LTeamID", "WTeamID"]].rename(
            columns={"LTeamID": "TeamID", "WTeamID": "OppID"}
        )
        matchups = pd.concat([w, l])
        matchups = matchups.merge(
            elo_df[["Season", "TeamID", "elo"]].rename(
                columns={"TeamID": "OppID", "elo": "opp_elo"}
            ),
            on=["Season", "OppID"],
            how="left",
        )
        sos = matchups.groupby(["Season", "TeamID"]).agg(
            sos=("opp_elo", "mean"),
            sos_std=("opp_elo", "std"),
        ).reset_index()
        return elo_df.merge(sos, on=["Season", "TeamID"], how="left")

    def _add_conf_strength(
        self, elo_df: pd.DataFrame, data: dict, gender: str, max_season: int
    ) -> pd.DataFrame:
        conf_df = data.get(f"{gender}_conf")
        if conf_df is None:
            elo_df["conf_elo_mean"] = np.nan
            elo_df["conf_elo_median"] = np.nan
            return elo_df

        conf_df = conf_df[conf_df["Season"] <= max_season]
        merged = conf_df.merge(elo_df[["Season", "TeamID", "elo"]], on=["Season", "TeamID"], how="left")
        conf_avg = merged.groupby(["Season", "ConfAbbrev"])["elo"].agg(
            conf_elo_mean="mean",
            conf_elo_median="median",
        ).reset_index()
        team_conf = conf_df.merge(conf_avg, on=["Season", "ConfAbbrev"], how="left")
        return elo_df.merge(
            team_conf[["Season", "TeamID", "conf_elo_mean", "conf_elo_median"]],
            on=["Season", "TeamID"],
            how="left",
        )
