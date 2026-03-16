"""
March Machine Learning Mania 2026 - Baseline Model v3
======================================================
Vectorized feature engineering + LightGBM ensemble
Features: Elo (RS+Tourney), Season Stats, Massey (individual + agg),
          Seeds, SOS, Conference Strength, Recent Form, Elo Win Prob
"""

import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

N_ENSEMBLE = 5  # Number of models in ensemble


# ============================================================
# 1. Data Loading
# ============================================================

def load_data() -> dict:
    dfs = {}
    for prefix, g in [("M", "M"), ("W", "W")]:
        dfs[f"{g}_rs_compact"] = pd.read_csv(DATA_DIR / f"{prefix}RegularSeasonCompactResults.csv")
        dfs[f"{g}_rs_detail"] = pd.read_csv(DATA_DIR / f"{prefix}RegularSeasonDetailedResults.csv")
        dfs[f"{g}_tourney"] = pd.read_csv(DATA_DIR / f"{prefix}NCAATourneyCompactResults.csv")
        dfs[f"{g}_seeds"] = pd.read_csv(DATA_DIR / f"{prefix}NCAATourneySeeds.csv")
        dfs[f"{g}_conf"] = pd.read_csv(DATA_DIR / f"{prefix}TeamConferences.csv")
    dfs["M_massey"] = pd.read_csv(DATA_DIR / "MMasseyOrdinals.csv")
    return dfs


# ============================================================
# 2. Elo Rating (with tournament games)
# ============================================================

def compute_elo_all(rs_compact: pd.DataFrame,
                    tourney_compact: pd.DataFrame | None = None,
                    K: float = 32, initial: float = 1500,
                    home_adv: float = 100, reset_factor: float = 0.75) -> pd.DataFrame:
    """Elo with margin-of-victory, including tournament games for carry-over."""
    # Combine RS + tourney, sorted by season/day
    all_games = rs_compact.copy()
    if tourney_compact is not None:
        all_games = pd.concat([all_games, tourney_compact], ignore_index=True)
    all_games = all_games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    seasons = sorted(all_games["Season"].unique())
    elo = {}
    records = []

    for season in seasons:
        if elo:
            mean_elo = np.mean(list(elo.values()))
            elo = {t: e * reset_factor + mean_elo * (1 - reset_factor) for t, e in elo.items()}

        sg = all_games[all_games["Season"] == season]

        # Record pre-tournament Elo (after RS games, before tourney)
        rs_games = sg[sg["DayNum"] <= 132]
        tourney_games = sg[sg["DayNum"] > 132]

        # Process RS games
        for _, row in rs_games.iterrows():
            _update_elo(elo, row, K, initial, home_adv)

        # Record end-of-RS Elo for this season's features
        for t, e in elo.items():
            records.append({"Season": season, "TeamID": t, "elo": e})

        # Process tourney games (updates carry to next season but not used as features)
        for _, row in tourney_games.iterrows():
            _update_elo(elo, row, K, initial, 0)  # No home advantage in tourney

    return pd.DataFrame(records)


def _update_elo(elo: dict, row, K: float, initial: float, home_adv: float):
    w, l = int(row["WTeamID"]), int(row["LTeamID"])
    elo.setdefault(w, initial)
    elo.setdefault(l, initial)

    ha = home_adv if row["WLoc"] == "H" else (-home_adv if row["WLoc"] == "A" else 0)
    elo_diff = elo[w] + ha - elo[l]
    expected_w = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

    mov = abs(row["WScore"] - row["LScore"])
    mov_mult = np.log(max(mov, 1) + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))

    update = K * mov_mult * (1 - expected_w)
    elo[w] += update
    elo[l] -= update


# ============================================================
# 3. Season Stats (vectorized)
# ============================================================

def compute_season_stats_vec(rs_detail: pd.DataFrame) -> pd.DataFrame:
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
        # Derived ratios
        "ast_to_ratio": agg["total_ast"] / agg["total_to"].replace(0, np.nan),
        "efg_pct": (agg["total_fgm"] + 0.5 * agg["total_fgm3"]) / agg["total_fga"].replace(0, np.nan),
        "ts_pct": agg["total_score"] / (2 * (agg["total_fga"] + 0.44 * agg["total_fta"])).replace(0, np.nan),
    })


def compute_season_stats_compact(rs_compact: pd.DataFrame) -> pd.DataFrame:
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


def compute_recent_form(rs_compact: pd.DataFrame, last_n: int = 10) -> pd.DataFrame:
    w = rs_compact[["Season", "DayNum", "WTeamID", "WScore", "LScore"]].rename(
        columns={"WTeamID": "TeamID", "WScore": "Score", "LScore": "OppScore"})
    w["Win"] = 1
    l = rs_compact[["Season", "DayNum", "LTeamID", "LScore", "WScore"]].rename(
        columns={"LTeamID": "TeamID", "LScore": "Score", "WScore": "OppScore"})
    l["Win"] = 0
    games = pd.concat([w, l]).sort_values(["Season", "TeamID", "DayNum"])

    recent = games.groupby(["Season", "TeamID"]).tail(last_n)
    agg = recent.groupby(["Season", "TeamID"]).agg(
        recent_wins=("Win", "sum"), recent_games=("Win", "count"),
        recent_margin=("Score", "sum"), recent_opp=("OppScore", "sum"),
    ).reset_index()
    agg["recent_win_pct"] = agg["recent_wins"] / agg["recent_games"]
    agg["recent_avg_margin"] = (agg["recent_margin"] - agg["recent_opp"]) / agg["recent_games"]
    return agg[["Season", "TeamID", "recent_win_pct", "recent_avg_margin"]]


# ============================================================
# 4. Massey Ordinals (individual + aggregate)
# ============================================================

TOP_MASSEY = ["POM", "SAG", "MOR", "WLK", "DOL", "COL", "RTH", "WOL", "CNG", "MB"]

def compute_massey_features(massey_df: pd.DataFrame) -> pd.DataFrame:
    last_day = massey_df.groupby(["Season", "SystemName"])["RankingDayNum"].max().reset_index()
    last_day.columns = ["Season", "SystemName", "MaxDay"]
    massey_last = massey_df.merge(last_day,
                                  left_on=["Season", "SystemName", "RankingDayNum"],
                                  right_on=["Season", "SystemName", "MaxDay"])

    # Aggregate features from top systems
    top = massey_last[massey_last["SystemName"].isin(TOP_MASSEY)]
    top_agg = top.groupby(["Season", "TeamID"])["OrdinalRank"].agg(
        massey_top_mean="mean", massey_top_median="median",
        massey_top_min="min", massey_top_max="max", massey_top_std="std",
    ).reset_index()

    all_agg = massey_last.groupby(["Season", "TeamID"])["OrdinalRank"].agg(
        massey_all_mean="mean", massey_all_median="median",
    ).reset_index()

    result = top_agg.merge(all_agg, on=["Season", "TeamID"], how="outer")

    # Individual top systems as separate features
    for sys_name in ["POM", "SAG", "MOR"]:
        sys_data = massey_last[massey_last["SystemName"] == sys_name][["Season", "TeamID", "OrdinalRank"]]
        sys_data = sys_data.rename(columns={"OrdinalRank": f"massey_{sys_name.lower()}"})
        result = result.merge(sys_data, on=["Season", "TeamID"], how="left")

    return result


# ============================================================
# 5. Seeds, SOS, Conference Strength
# ============================================================

def parse_seed_num(seed_str: str) -> int:
    return int(seed_str[1:3])

def get_seed_features(seeds_df: pd.DataFrame) -> pd.DataFrame:
    s = seeds_df.copy()
    s["seed_num"] = s["Seed"].apply(parse_seed_num)
    return s[["Season", "TeamID", "seed_num"]]

def compute_sos(rs_compact: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    w = rs_compact[["Season", "WTeamID", "LTeamID"]].rename(
        columns={"WTeamID": "TeamID", "LTeamID": "OppID"})
    l = rs_compact[["Season", "LTeamID", "WTeamID"]].rename(
        columns={"LTeamID": "TeamID", "WTeamID": "OppID"})
    matchups = pd.concat([w, l])
    matchups = matchups.merge(elo_df.rename(columns={"TeamID": "OppID", "elo": "opp_elo"}),
                               on=["Season", "OppID"], how="left")
    sos = matchups.groupby(["Season", "TeamID"]).agg(
        sos=("opp_elo", "mean"),
        sos_std=("opp_elo", "std"),
    ).reset_index()
    return sos


def compute_conf_strength(conf_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """Conference average Elo as a team feature."""
    merged = conf_df.merge(elo_df, on=["Season", "TeamID"], how="left")
    conf_avg = merged.groupby(["Season", "ConfAbbrev"])["elo"].agg(
        conf_elo_mean="mean", conf_elo_median="median",
    ).reset_index()
    # Join back to teams
    result = conf_df.merge(conf_avg, on=["Season", "ConfAbbrev"], how="left")
    return result[["Season", "TeamID", "conf_elo_mean", "conf_elo_median"]]


# ============================================================
# 6. Build team-level feature table
# ============================================================

def build_team_features(dfs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = {}
    for g in ["M", "W"]:
        print(f"  Building {g} features...")

        # Elo (with tournament carry-over)
        elo = compute_elo_all(dfs[f"{g}_rs_compact"], dfs[f"{g}_tourney"])

        # Season stats
        stats_detail = compute_season_stats_vec(dfs[f"{g}_rs_detail"])
        stats_compact = compute_season_stats_compact(dfs[f"{g}_rs_compact"])
        detail_keys = set(zip(stats_detail["Season"], stats_detail["TeamID"]))
        compact_only = stats_compact[
            stats_compact.apply(lambda r: (r["Season"], r["TeamID"]) not in detail_keys, axis=1)
        ]
        feat = pd.concat([stats_detail, compact_only], ignore_index=True)

        # Merge features
        feat = feat.merge(elo, on=["Season", "TeamID"], how="left")
        feat = feat.merge(compute_sos(dfs[f"{g}_rs_compact"], elo), on=["Season", "TeamID"], how="left")
        feat = feat.merge(compute_recent_form(dfs[f"{g}_rs_compact"]), on=["Season", "TeamID"], how="left")
        feat = feat.merge(get_seed_features(dfs[f"{g}_seeds"]), on=["Season", "TeamID"], how="left")
        feat = feat.merge(compute_conf_strength(dfs[f"{g}_conf"], elo), on=["Season", "TeamID"], how="left")

        if g == "M":
            feat = feat.merge(compute_massey_features(dfs["M_massey"]), on=["Season", "TeamID"], how="left")

        results[g] = feat
    return results["M"], results["W"]


# ============================================================
# 7. Matchup features (vectorized)
# ============================================================

def build_matchup_df(matchups: pd.DataFrame, team_feats: pd.DataFrame,
                     gender_flag: int) -> pd.DataFrame:
    feat_cols = [c for c in team_feats.columns if c not in ("Season", "TeamID")]

    rename_a = {c: f"A_{c}" for c in feat_cols}
    rename_b = {c: f"B_{c}" for c in feat_cols}

    m = matchups.merge(
        team_feats.rename(columns=rename_a),
        left_on=["Season", "TeamA"], right_on=["Season", "TeamID"], how="left"
    ).drop("TeamID", axis=1, errors="ignore")

    m = m.merge(
        team_feats.rename(columns=rename_b),
        left_on=["Season", "TeamB"], right_on=["Season", "TeamID"], how="left"
    ).drop("TeamID", axis=1, errors="ignore")

    # Diffs
    for c in feat_cols:
        m[f"diff_{c}"] = m[f"A_{c}"] - m[f"B_{c}"]

    # Elo win probability (logistic)
    if "A_elo" in m.columns and "B_elo" in m.columns:
        elo_diff = m["A_elo"] - m["B_elo"]
        m["elo_win_prob"] = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

    # Seed interaction features
    if "A_seed_num" in m.columns and "B_seed_num" in m.columns:
        m["seed_product"] = m["A_seed_num"] * m["B_seed_num"]
        m["seed_sum"] = m["A_seed_num"] + m["B_seed_num"]

    # Drop A_/B_ columns
    drop_cols = [f"A_{c}" for c in feat_cols] + [f"B_{c}" for c in feat_cols]
    m = m.drop(columns=drop_cols)

    m["gender"] = gender_flag
    return m


# ============================================================
# 8. Training (ensemble)
# ============================================================

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.02,
    "num_leaves": 8,
    "max_depth": 5,
    "min_child_samples": 30,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "verbose": -1,
    "n_jobs": -1,
}

# Blend weight: final = w * elo_prob + (1-w) * lgbm
ELO_BLEND_WEIGHT = 0.15


def build_tourney_matchups(tourney_df: pd.DataFrame) -> pd.DataFrame:
    w = tourney_df["WTeamID"].values
    l = tourney_df["LTeamID"].values
    a = np.minimum(w, l)
    b = np.maximum(w, l)
    label = (w == a).astype(int)
    return pd.DataFrame({
        "Season": tourney_df["Season"].values,
        "TeamA": a, "TeamB": b, "label": label,
    })


def train_and_validate(train_df: pd.DataFrame) -> tuple:
    feature_cols = [c for c in train_df.columns
                    if c not in ("Season", "TeamA", "TeamB", "label")]

    X = train_df[feature_cols].astype(np.float32)
    y = train_df["label"].values
    seasons = train_df["Season"].values

    val_seasons = [2021, 2022, 2023, 2024, 2025]
    train_mask = ~np.isin(seasons, val_seasons)
    val_mask = np.isin(seasons, val_seasons)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    seasons_val = seasons[val_mask]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    # --- Single model for validation ---
    params = {**LGBM_PARAMS, "seed": 42}
    dtrain = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, y_val, feature_name=feature_cols, reference=dtrain)

    model = lgb.train(
        params, dtrain, num_boost_round=5000,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    best_iter = model.best_iteration

    preds_val_raw = model.predict(X_val)

    # Blend with Elo win probability
    elo_wp_val = X_val["elo_win_prob"].values if "elo_win_prob" in feature_cols else None
    if elo_wp_val is not None:
        preds_val = np.clip(
            (1 - ELO_BLEND_WEIGHT) * preds_val_raw + ELO_BLEND_WEIGHT * elo_wp_val,
            0.02, 0.98)
        preds_val_noblend = np.clip(preds_val_raw, 0.02, 0.98)
        brier_noblend = np.mean((preds_val_noblend - y_val) ** 2)
        print(f"  (no blend Brier: {brier_noblend:.4f})")
    else:
        preds_val = np.clip(preds_val_raw, 0.02, 0.98)

    # --- Ensemble for final model ---
    print(f"\n  Training {N_ENSEMBLE}-model ensemble (best_iter={best_iter})...")
    models = []
    ensemble_preds = np.zeros(len(X_val))

    for i in range(N_ENSEMBLE):
        seed = 42 + i * 17
        p = {**LGBM_PARAMS, "seed": seed, "subsample_seed": seed,
             "colsample_bytree": 0.6 + (i % 3) * 0.1,
             "subsample": 0.65 + (i % 3) * 0.05}

        # Train on all data
        m_full = lgb.train(p, lgb.Dataset(X, y, feature_name=feature_cols),
                           num_boost_round=best_iter)
        models.append(m_full)

        # Also get val prediction for this seed
        m_val = lgb.train(p, lgb.Dataset(X_train, y_train, feature_name=feature_cols),
                          num_boost_round=best_iter)
        ensemble_preds += m_val.predict(X_val)

    ensemble_preds /= N_ENSEMBLE

    # Blend ensemble with Elo
    if elo_wp_val is not None:
        ensemble_preds = (1 - ELO_BLEND_WEIGHT) * ensemble_preds + ELO_BLEND_WEIGHT * elo_wp_val
    ensemble_preds = np.clip(ensemble_preds, 0.02, 0.98)

    # Metrics
    brier_single = np.mean((preds_val - y_val) ** 2)
    brier_ens = np.mean((ensemble_preds - y_val) ** 2)

    logloss_single = -np.mean(y_val * np.log(preds_val + 1e-15) + (1 - y_val) * np.log(1 - preds_val + 1e-15))
    logloss_ens = -np.mean(y_val * np.log(ensemble_preds + 1e-15) + (1 - y_val) * np.log(1 - ensemble_preds + 1e-15))

    acc_single = np.mean((preds_val > 0.5) == y_val)
    acc_ens = np.mean((ensemble_preds > 0.5) == y_val)

    print(f"\n=== Validation (seasons {val_seasons}) ===")
    print(f"  Single:   Brier={brier_single:.4f}, LogLoss={logloss_single:.4f}, Acc={acc_single:.4f}")
    print(f"  Ensemble: Brier={brier_ens:.4f}, LogLoss={logloss_ens:.4f}, Acc={acc_ens:.4f}")

    for s in val_seasons:
        ms = seasons_val == s
        if ms.sum() > 0:
            b = np.mean((ensemble_preds[ms] - y_val[ms]) ** 2)
            a = np.mean((ensemble_preds[ms] > 0.5) == y_val[ms])
            print(f"  {s}: Brier={b:.4f}, Acc={a:.4f}, N={ms.sum()}")

    # Feature importance (from single model)
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print(f"\n=== Top 20 Features ===")
    print(imp.head(20).to_string(index=False))

    return models, feature_cols, {
        "brier_single": brier_single, "brier_ensemble": brier_ens,
        "logloss_single": logloss_single, "logloss_ensemble": logloss_ens,
        "accuracy_single": acc_single, "accuracy_ensemble": acc_ens,
        "best_iteration": best_iter,
    }


# ============================================================
# 9. Submission
# ============================================================

def generate_submission(models: list, feature_cols: list,
                        m_feats: pd.DataFrame, w_feats: pd.DataFrame) -> pd.DataFrame:
    sample = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
    parsed = sample["ID"].str.split("_", expand=True)
    sample["Season"] = parsed[0].astype(int)
    sample["TeamA"] = parsed[1].astype(int)
    sample["TeamB"] = parsed[2].astype(int)

    m_mask = sample["TeamA"] < 3000
    m_matchups = sample[m_mask][["Season", "TeamA", "TeamB"]].copy()
    w_matchups = sample[~m_mask][["Season", "TeamA", "TeamB"]].copy()

    print(f"  Men: {len(m_matchups)}, Women: {len(w_matchups)}")

    m_pred_df = build_matchup_df(m_matchups, m_feats, gender_flag=0)
    w_pred_df = build_matchup_df(w_matchups, w_feats, gender_flag=1)
    pred_df = pd.concat([m_pred_df, w_pred_df], ignore_index=True)

    for c in feature_cols:
        if c not in pred_df.columns:
            pred_df[c] = np.nan

    X_pred = pred_df[feature_cols].astype(np.float32)

    # Ensemble prediction
    preds = np.zeros(len(X_pred))
    for m in models:
        preds += m.predict(X_pred)
    preds /= len(models)

    # Blend with Elo
    if "elo_win_prob" in pred_df.columns:
        elo_wp = pred_df["elo_win_prob"].values
        preds = (1 - ELO_BLEND_WEIGHT) * preds + ELO_BLEND_WEIGHT * elo_wp
    preds = np.clip(preds, 0.02, 0.98)

    pred_df["Pred"] = preds
    pred_df["ID"] = (pred_df["Season"].astype(str) + "_" +
                     pred_df["TeamA"].astype(str) + "_" +
                     pred_df["TeamB"].astype(str))

    submission = sample[["ID"]].merge(pred_df[["ID", "Pred"]], on="ID", how="left")

    out_path = OUT_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Stats: mean={preds.mean():.4f}, std={preds.std():.4f}, "
          f"min={preds.min():.4f}, max={preds.max():.4f}")
    return submission


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("March Machine Learning Mania 2026 - Baseline v3")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    dfs = load_data()

    print("\n[2/4] Building team features...")
    m_feats, w_feats = build_team_features(dfs)
    print(f"  M: {m_feats.shape}, W: {w_feats.shape}")

    print("\n[3/4] Training model...")
    m_tourney = build_tourney_matchups(dfs["M_tourney"])
    w_tourney = build_tourney_matchups(dfs["W_tourney"])

    m_train = build_matchup_df(m_tourney, m_feats, gender_flag=0)
    m_train["label"] = m_tourney["label"].values
    w_train = build_matchup_df(w_tourney, w_feats, gender_flag=1)
    w_train["label"] = w_tourney["label"].values

    train_df = pd.concat([m_train, w_train], ignore_index=True)
    print(f"  Training data: {train_df.shape}")

    models, feature_cols, metrics = train_and_validate(train_df)

    # Save first model
    models[0].save_model(str(OUT_DIR / "model.lgb"))

    print("\n[4/4] Generating submission...")
    generate_submission(models, feature_cols, m_feats, w_feats)

    with open(OUT_DIR / "metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"\n{'='*60}")
    print(f"Brier (single): {metrics['brier_single']:.4f}")
    print(f"Brier (ensemble): {metrics['brier_ensemble']:.4f}")
    print(f"Accuracy (ensemble): {metrics['accuracy_ensemble']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
