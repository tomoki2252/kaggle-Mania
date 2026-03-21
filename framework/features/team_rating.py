"""チーム Rating 特徴量

Regular Season Detailed Results から team-season ごとに latent team ratings を推定する。

2つの推定方式:
1. Ridge回帰: 各試合のスコアを team_offense + opp_defense + location で線形モデル化
2. 反復調整法 (Adjusted Efficiency): KenPom 的な iterative opponent-adjustment

生成される rating:
- offense_rating: 攻撃力 (100ポゼッション当たりの得点)
- defense_rating: 守備力 (100ポゼッション当たりの失点, 低い方が良い)
- pace_rating: ペース (1試合当たりのポゼッション数)
- net_rating: offense - defense
- consistency: スコアの安定性 (margin の標準偏差の逆数)
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from framework.features.base import FeatureGenerator


def _build_game_records(rs_detail: pd.DataFrame) -> pd.DataFrame:
    """DetailedResults を試合単位のレコードに変換する。

    各試合を2行（各チーム視点）に展開し、ポゼッション・効率を計算する。
    """
    w_cols = {
        "Season": "Season", "DayNum": "DayNum",
        "WTeamID": "TeamID", "LTeamID": "OppID",
        "WScore": "Score", "LScore": "OppScore",
        "WLoc": "Loc",
        "WFGA": "FGA", "WOR": "OR", "WTO": "TO",
        "WFTA": "FTA",
        "LFGA": "OppFGA", "LOR": "OppOR", "LTO": "OppTO",
        "LFTA": "OppFTA",
    }
    l_cols = {
        "Season": "Season", "DayNum": "DayNum",
        "LTeamID": "TeamID", "WTeamID": "OppID",
        "LScore": "Score", "WScore": "OppScore",
        "WLoc": "WLoc_orig",
        "LFGA": "FGA", "LOR": "OR", "LTO": "TO",
        "LFTA": "FTA",
        "WFGA": "OppFGA", "WOR": "OppOR", "WTO": "OppTO",
        "WFTA": "OppFTA",
    }

    w_df = rs_detail[list(w_cols.keys())].rename(columns=w_cols)
    loc_map = {"H": 1, "A": -1, "N": 0}
    w_df["home"] = w_df["Loc"].map(loc_map).fillna(0)

    l_df = rs_detail[list(l_cols.keys())].rename(columns=l_cols)
    loc_map_inv = {"H": -1, "A": 1, "N": 0}
    l_df["home"] = l_df["WLoc_orig"].map(loc_map_inv).fillna(0)
    l_df = l_df.drop(columns=["WLoc_orig"])
    w_df = w_df.drop(columns=["Loc"])

    games = pd.concat([w_df, l_df], ignore_index=True)

    # ポゼッション推定 (Dean Oliver の近似式)
    games["Poss"] = games["FGA"] - games["OR"] + games["TO"] + 0.475 * games["FTA"]
    games["OppPoss"] = games["OppFGA"] - games["OppOR"] + games["OppTO"] + 0.475 * games["OppFTA"]
    games["AvgPoss"] = ((games["Poss"] + games["OppPoss"]) / 2).clip(lower=1)

    # 100ポゼッション当たりの効率
    games["OffEff"] = games["Score"] / games["AvgPoss"] * 100
    games["DefEff"] = games["OppScore"] / games["AvgPoss"] * 100

    return games


# =============================================================================
# 方式1: Ridge回帰ベース (ベクトル化)
# =============================================================================

def _estimate_ratings_ridge(
    games: pd.DataFrame, alpha: float = 10.0
) -> pd.DataFrame:
    """Ridge回帰で team ratings を推定する (ベクトル化版)。"""
    ratings_list = []

    for season in sorted(games["Season"].unique()):
        sg = games[games["Season"] == season].reset_index(drop=True)
        teams = sorted(sg["TeamID"].unique())
        if len(teams) < 3:
            continue

        team_to_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)
        n_games = len(sg)

        # ベクトル化: team/opp のインデックスを一括変換
        t_indices = sg["TeamID"].map(team_to_idx).values
        o_indices = sg["OppID"].map(team_to_idx).values
        home_vals = sg["home"].values

        # --- Offense/Defense Design matrix (sparse) ---
        rows = np.arange(n_games)
        # team offense indicator
        data_t = np.ones(n_games)
        # opponent defense indicator
        data_o = np.ones(n_games)
        # home indicator
        data_h = home_vals

        row_idx = np.concatenate([rows, rows, rows])
        col_idx = np.concatenate([t_indices, n_teams + o_indices, np.full(n_games, 2 * n_teams)])
        data_all = np.concatenate([data_t, data_o, data_h])
        X = sparse.csr_matrix((data_all, (row_idx, col_idx)), shape=(n_games, 2 * n_teams + 1))

        y_off = sg["OffEff"].values
        y_def = sg["DefEff"].values

        # Offense model
        ridge_off = Ridge(alpha=alpha, fit_intercept=True)
        ridge_off.fit(X, y_off)
        off_coefs = ridge_off.coef_[:n_teams]
        home_off = ridge_off.coef_[-1]
        intercept_off = ridge_off.intercept_

        # Defense model
        ridge_def = Ridge(alpha=alpha, fit_intercept=True)
        ridge_def.fit(X, y_def)
        def_coefs = ridge_def.coef_[:n_teams]

        intercept_def = ridge_def.intercept_

        # --- Pace Design matrix (sparse) ---
        row_idx_p = np.concatenate([rows, rows, rows])
        col_idx_p = np.concatenate([t_indices, o_indices, np.full(n_games, n_teams)])
        data_p = np.concatenate([np.ones(n_games), np.ones(n_games), home_vals])
        X_pace = sparse.csr_matrix((data_p, (row_idx_p, col_idx_p)), shape=(n_games, n_teams + 1))

        y_pace = sg["AvgPoss"].values
        ridge_pace = Ridge(alpha=alpha, fit_intercept=True)
        ridge_pace.fit(X_pace, y_pace)
        pace_coefs = ridge_pace.coef_[:n_teams]

        # 結果を正規化
        off_ratings = off_coefs - off_coefs.mean() + intercept_off
        def_ratings = def_coefs - def_coefs.mean() + intercept_def
        pace_ratings = pace_coefs - pace_coefs.mean() + ridge_pace.intercept_

        # Consistency
        sg["Margin"] = sg["Score"] - sg["OppScore"]
        margin_std = sg.groupby("TeamID")["Margin"].std().reindex(teams).fillna(15.0)
        consistency = 1.0 / margin_std.clip(lower=1.0)

        # 結果をまとめて DataFrame に (ループ回避)
        season_df = pd.DataFrame({
            "Season": season,
            "TeamID": teams,
            "ridge_off": off_ratings,
            "ridge_def": def_ratings,
            "ridge_net": off_ratings - def_ratings,
            "ridge_pace": pace_ratings,
            "ridge_consistency": consistency.values,
        })
        ratings_list.append(season_df)

    if not ratings_list:
        return pd.DataFrame(columns=["Season", "TeamID"] + _RIDGE_COLS)
    return pd.concat(ratings_list, ignore_index=True)


# =============================================================================
# 方式2: 反復調整法 (ベクトル化)
# =============================================================================

def _estimate_ratings_iterative(
    games: pd.DataFrame, n_iter: int = 20
) -> pd.DataFrame:
    """反復調整法で opponent-adjusted efficiency を計算する (ベクトル化版)。"""
    ratings_list = []

    for season in sorted(games["Season"].unique()):
        sg = games[games["Season"] == season].reset_index(drop=True)
        teams = sorted(sg["TeamID"].unique())
        if len(teams) < 3:
            continue

        team_to_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        # ベクトル化のため配列に変換
        t_idx = sg["TeamID"].map(team_to_idx).values
        o_idx = sg["OppID"].map(team_to_idx).values
        off_eff = sg["OffEff"].values
        def_eff = sg["DefEff"].values
        avg_poss = sg["AvgPoss"].values
        home_vals = sg["home"].values

        league_off = off_eff.mean()
        league_def = def_eff.mean()
        league_pace = avg_poss.mean()

        # Home advantage
        home_mask = home_vals == 1
        away_mask = home_vals == -1
        if home_mask.sum() > 10 and away_mask.sum() > 10:
            home_adv = (off_eff[home_mask].mean() - off_eff[away_mask].mean()) / 2
        else:
            home_adv = 1.5

        # 初期値: team ごとの平均 (bincount で高速計算)
        _counts = np.bincount(t_idx, minlength=n_teams).astype(float)
        _counts_safe = np.maximum(_counts, 1)
        adj_off = np.bincount(t_idx, weights=off_eff, minlength=n_teams) / _counts_safe
        adj_def = np.bincount(t_idx, weights=def_eff, minlength=n_teams) / _counts_safe
        adj_pace = np.bincount(t_idx, weights=avg_poss, minlength=n_teams) / _counts_safe
        # 試合のないチームは league average
        no_games = _counts == 0
        adj_off[no_games] = league_off
        adj_def[no_games] = league_def
        adj_pace[no_games] = league_pace

        # location adjustment (試合ごとに固定)
        loc_adj = home_adv * home_vals

        # team ごとの試合数 (bincount で高速集計用)
        games_per_team = np.bincount(t_idx, minlength=n_teams).astype(float)
        games_per_team = np.maximum(games_per_team, 1)  # ゼロ除算防止

        # 反復調整 (全ベクトル化 + bincount)
        for _ in range(n_iter):
            # opponent の現在の rating を lookup
            opp_d = adj_def[o_idx]
            opp_o = adj_off[o_idx]
            opp_p = adj_pace[o_idx]

            # adjustment factors
            off_factor = league_def / np.maximum(opp_d, 1.0)
            def_factor = league_off / np.maximum(opp_o, 1.0)
            pace_factor = league_pace / np.maximum(opp_p, 1.0)

            # adjusted values per game
            adj_off_game = (off_eff - loc_adj) * off_factor
            adj_def_game = (def_eff + loc_adj) * def_factor
            adj_pace_game = avg_poss * pace_factor

            # team ごとの合計 -> 平均 (bincount で O(N) 集計)
            adj_off = np.bincount(t_idx, weights=adj_off_game, minlength=n_teams) / games_per_team
            adj_def = np.bincount(t_idx, weights=adj_def_game, minlength=n_teams) / games_per_team
            adj_pace = np.bincount(t_idx, weights=adj_pace_game, minlength=n_teams) / games_per_team

        # Consistency
        sg["Margin"] = sg["Score"] - sg["OppScore"]
        margin_std = sg.groupby("TeamID")["Margin"].std().reindex(teams).fillna(15.0)
        consistency = 1.0 / margin_std.clip(lower=1.0)

        season_df = pd.DataFrame({
            "Season": season,
            "TeamID": teams,
            "iter_off": adj_off,
            "iter_def": adj_def,
            "iter_net": adj_off - adj_def,
            "iter_pace": adj_pace,
            "iter_consistency": consistency.values,
        })
        ratings_list.append(season_df)

    if not ratings_list:
        return pd.DataFrame(columns=["Season", "TeamID"] + _ITER_COLS)
    return pd.concat(ratings_list, ignore_index=True)


# =============================================================================
# FeatureGenerator クラス
# =============================================================================

_RIDGE_COLS = [
    "ridge_off", "ridge_def", "ridge_net", "ridge_pace", "ridge_consistency",
]

_ITER_COLS = [
    "iter_off", "iter_def", "iter_net", "iter_pace", "iter_consistency",
]

_ALL_COLS = _RIDGE_COLS + _ITER_COLS


class TeamRatingFeature(FeatureGenerator):
    """チーム Rating 特徴量

    方式を選択可能: "ridge", "iterative", "both"
    """

    def __init__(self, method: str = "both", ridge_alpha: float = 10.0, n_iter: int = 20):
        self._method = method
        self._ridge_alpha = ridge_alpha
        self._n_iter = n_iter

    @property
    def name(self) -> str:
        return f"team_rating_{self._method}"

    @property
    def feature_columns(self) -> list[str]:
        if self._method == "ridge":
            return _RIDGE_COLS
        elif self._method == "iterative":
            return _ITER_COLS
        else:
            return _ALL_COLS

    def generate(
        self,
        data: dict[str, pd.DataFrame],
        gender: str,
        max_season: int,
    ) -> pd.DataFrame:
        detail_key = f"{gender}_rs_detail"
        if detail_key not in data:
            raise ValueError(f"DetailedResults が見つかりません: {detail_key}")

        rs_detail = data[detail_key]
        rs_detail = rs_detail[rs_detail["Season"] <= max_season]

        games = _build_game_records(rs_detail)

        dfs = []
        if self._method in ("ridge", "both"):
            ridge_df = _estimate_ratings_ridge(games, alpha=self._ridge_alpha)
            dfs.append(ridge_df)

        if self._method in ("iterative", "both"):
            iter_df = _estimate_ratings_iterative(games, n_iter=self._n_iter)
            dfs.append(iter_df)

        if len(dfs) == 1:
            return dfs[0]

        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on=["Season", "TeamID"], how="outer")

        return result
