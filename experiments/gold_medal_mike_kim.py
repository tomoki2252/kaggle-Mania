"""
Gold Medal Solution reproduction - Mike Kim
Reference: reference/gold-medal-solution-mike-kim.ipynb

モデル概要:
1. データ準備: men/women統合、box scoreをT1/T2形式に変換（勝敗両方向）
2. Easy特徴量: シード番号、シード差
3. Medium特徴量: シーズン平均box score、直近14日勝率、アウェイ勝率、対戦成績、勝率ベース強さ指標
4. Hard特徴量: Eloレーティング
5. Hardest特徴量: GLM team quality (statsmodels)
6. メタ特徴量: NuSVC + KNN (men/womenクロス学習)
7. 最終モデル: XGBoost (Cauchy損失) + leaf-based LogisticRegression calibration
8. 推論: 全LOSOモデルのアンサンブル
"""

import gc
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import tqdm
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC

warnings.filterwarnings("ignore")
pd.set_option("display.max_column", 999)

# ============================================================
# Config
# ============================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "submission"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SEASON = 2003  # cutoff year as in original notebook

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 60)
print("1. Loading data...")
print("=" * 60)

M_regular_results = pd.read_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv")
M_tourney_results = pd.read_csv(DATA_DIR / "MNCAATourneyDetailedResults.csv")
M_seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")

W_regular_results = pd.read_csv(DATA_DIR / "WRegularSeasonDetailedResults.csv")
W_tourney_results = pd.read_csv(DATA_DIR / "WNCAATourneyDetailedResults.csv")
W_seeds = pd.read_csv(DATA_DIR / "WNCAATourneySeeds.csv")

# join men's and women's data
regular_results = pd.concat([M_regular_results, W_regular_results])
tourney_results = pd.concat([M_tourney_results, W_tourney_results])
seeds = pd.concat([M_seeds, W_seeds])

# filter by season
regular_results = regular_results.loc[regular_results["Season"] >= MIN_SEASON]
tourney_results = tourney_results.loc[tourney_results["Season"] >= MIN_SEASON]
seeds = seeds.loc[seeds["Season"] >= MIN_SEASON]

# Home/Away encoding
wloc = {"H": 1, "A": -1, "N": np.nan}
regular_results["WHome"] = regular_results["WLoc"].map(lambda x: wloc[x])
tourney_results["WHome"] = tourney_results["WLoc"].map(lambda x: wloc[x])

print(f"  Regular season games: {len(regular_results)}")
print(f"  Tourney games: {len(tourney_results)}")
print(f"  Seeds: {len(seeds)}")

# ============================================================
# 2. Prepare Data (double dataset with swapped team positions)
# ============================================================
print("\n" + "=" * 60)
print("2. Preparing data...")
print("=" * 60)


def prepare_data(df):
    df = df[
        [
            "Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "NumOT", "WHome",
            "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
            "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
        ]
    ]

    # adjustment factor for overtimes
    adjot = (40 + 5 * df["NumOT"]) / 40
    adjcols = [
        "LScore", "WScore",
        "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
        "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
    ]
    for col in adjcols:
        df[col] = df[col] / adjot

    dfswap = df.copy()
    df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in list(df.columns)]
    dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in list(dfswap.columns)]
    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output["PointDiff"] = output["T1_Score"] - output["T2_Score"]
    output["win"] = (output["PointDiff"] > 0) * 1
    output["men_women"] = (output["T1_TeamID"].apply(lambda t: str(t).startswith("1"))) * 1  # 0: women, 1: men
    return output


regular_data = prepare_data(regular_results.copy())
tourney_data = prepare_data(tourney_results.copy())

print(f"  Regular data (doubled): {len(regular_data)}")
print(f"  Tourney data (doubled): {len(tourney_data)}")

# ============================================================
# 3. Easy Features: Seeds
# ============================================================
print("\n" + "=" * 60)
print("3. Easy features: Seeds...")
print("=" * 60)

seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))

seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]]
tourney_data = pd.merge(tourney_data, seeds_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, seeds_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

print(f"  Tourney data shape: {tourney_data.shape}")

# ============================================================
# 4. Medium Features: Box score averages & win ratios
# ============================================================
print("\n" + "=" * 60)
print("4. Medium features: Box score averages, win ratios...")
print("=" * 60)

boxcols = [
    "T1_Home", "T2_Home",
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
    "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
    "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
    "PointDiff",
]

# --- WinRatio14d (last 14 days of regular season) ---
def summarize_cols(x):
    return pd.Series({
        "WinRatio14d": x["win14days"].sum() / (1 + x["last14days"].sum()),
    })


regular_data["win14days"] = (regular_data["DayNum"] > 118) & (regular_data["T1_Score"] > regular_data["T2_Score"])
regular_data["win14days"] = regular_data["win14days"].map(int)
regular_data["last14days"] = (regular_data["DayNum"] > 118).map(int)

ss_new = regular_data.groupby(["Season", "T1_TeamID"]).apply(summarize_cols, include_groups=False).reset_index()
ss_new["season_team"] = ss_new["Season"].map(str) + "_" + ss_new["T1_TeamID"].map(str)
ratio_dict = dict(zip(ss_new["season_team"].tolist(), ss_new["WinRatio14d"].tolist()))

# --- Matchup history (Laplace) ---
s = regular_data["Season"].tolist()
t1 = regular_data["T1_TeamID"].tolist()
t2 = regular_data["T2_TeamID"].tolist()
s1 = regular_data["T1_Score"].tolist()
s2 = regular_data["T2_Score"].tolist()
d = defaultdict(lambda: [1, 1])
for i in range(len(s)):
    tup = str(s[i]) + "_" + str(t1[i]) + "_" + str(t2[i])
    if s1[i] > s2[i]:
        d[tup][0] += 1
    else:
        d[tup][1] += 1

match_dict = defaultdict(lambda: 0.5)
for k, v in d.items():
    match_dict[k] = v[0] / v[1]

# --- Away wins ---
l1 = regular_data["T1_TeamID"].tolist()
l2 = regular_data["T2_TeamID"].tolist()
s0 = regular_data["Season"].tolist()
s1_scores = regular_data["T1_Score"].tolist()
s2_scores = regular_data["T2_Score"].tolist()
h1 = regular_data["T1_Home"].tolist()
h2 = regular_data["T2_Home"].tolist()

home_dict = defaultdict(lambda: [0, 0])
for i in range(len(l1)):
    k = str(s0[i]) + "_" + str(l1[i])
    if s1_scores[i] > s2_scores[i] and h1[i] == -1:
        home_dict[k][0] += 1
    elif s1_scores[i] < s2_scores[i] and h1[i] == -1:
        home_dict[k][1] += 1

for i in range(len(l2)):
    k = str(s0[i]) + "_" + str(l2[i])
    if s1_scores[i] < s2_scores[i] and h2[i] == -1:
        home_dict[k][0] += 1
    elif s1_scores[i] > s2_scores[i] and h2[i] == -1:
        home_dict[k][1] += 1

away_dict = {}
for k, v in home_dict.items():
    away_dict[k] = v[0] / (v[0] + v[1])

# --- Win-ratio based strength (awins) ---
l1 = regular_data["T1_TeamID"].tolist()
l2 = regular_data["T2_TeamID"].tolist()
s0 = regular_data["Season"].tolist()
s1_scores = regular_data["T1_Score"].tolist()
s2_scores = regular_data["T2_Score"].tolist()

# First pass: raw win ratio
l_dict = defaultdict(list)
for i in range(len(l1)):
    k = str(s0[i]) + "_" + str(l1[i])
    if s1_scores[i] < s2_scores[i]:
        l_dict[k].append(0)
    else:
        l_dict[k].append(1)

for i in range(len(l2)):
    k = str(s0[i]) + "_" + str(l2[i])
    if s1_scores[i] > s2_scores[i]:
        l_dict[k].append(0)
    else:
        l_dict[k].append(1)

awins_dict = defaultdict(lambda: np.nan)
for k, v in l_dict.items():
    awins_dict[k] = np.mean(l_dict[k])

# Second pass: strength of schedule weighted wins
l_dict = defaultdict(list)
for i in range(len(l1)):
    k = str(s0[i]) + "_" + str(l1[i])
    if s1_scores[i] < s2_scores[i]:
        l_dict[k].append(0)
    else:
        l_dict[k].append(awins_dict[str(s0[i]) + "_" + str(l2[i])])

for i in range(len(l2)):
    k = str(s0[i]) + "_" + str(l2[i])
    if s1_scores[i] > s2_scores[i]:
        l_dict[k].append(0)
    else:
        l_dict[k].append(awins_dict[str(s0[i]) + "_" + str(l1[i])])

awins_dict = defaultdict(lambda: np.nan)
for k, v in l_dict.items():
    awins_dict[k] = np.mean(l_dict[k])

print("  Win ratio, matchup, away wins, awins features computed.")

# --- Season averages ---
ss = regular_data.groupby(["Season", "T1_TeamID"])[boxcols].agg("mean").reset_index()

ss_T1 = ss.copy()
ss_T1.columns = ["T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(ss_T1.columns)]
ss_T1 = ss_T1.rename({"T1_avg_Season": "Season", "T1_avg_TeamID": "T1_TeamID"}, axis=1)

ss_T1["T1_WinRatio14d"] = ss_T1["Season"].map(str) + "_" + ss_T1["T1_TeamID"].map(str)
ss_T1["T1_away_wins"] = ss_T1["T1_WinRatio14d"].map(away_dict)
ss_T1["T1_awins"] = ss_T1["T1_WinRatio14d"].map(awins_dict)
ss_T1["T1_WinRatio14d"] = ss_T1["T1_WinRatio14d"].map(ratio_dict)

ss_T2 = ss.copy()
ss_T2.columns = ["T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(ss_T2.columns)]
ss_T2 = ss_T2.rename({"T2_avg_Season": "Season", "T2_avg_TeamID": "T2_TeamID"}, axis=1)

ss_T2["T2_WinRatio14d"] = ss_T2["Season"].map(str) + "_" + ss_T2["T2_TeamID"].map(str)
ss_T2["T2_away_wins"] = ss_T2["T2_WinRatio14d"].map(away_dict)
ss_T2["T2_awins"] = ss_T2["T2_WinRatio14d"].map(awins_dict)
ss_T2["T2_WinRatio14d"] = ss_T2["T2_WinRatio14d"].map(ratio_dict)

tourney_data = pd.merge(tourney_data, ss_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, ss_T2, on=["Season", "T2_TeamID"], how="left")

tourney_data["laplace_matchup"] = (
    tourney_data["Season"].map(str)
    + "_"
    + tourney_data["T1_TeamID"].map(str)
    + "_"
    + tourney_data["T2_TeamID"].map(str)
)
tourney_data["laplace_matchup"] = tourney_data["laplace_matchup"].map(match_dict)

print(f"  Tourney data shape after medium features: {tourney_data.shape}")

# ============================================================
# 5. Hard Features: Elo Ratings
# ============================================================
print("\n" + "=" * 60)
print("5. Hard features: Elo ratings...")
print("=" * 60)


def update_elo(winner_elo, loser_elo):
    expected_win = expected_result(winner_elo, loser_elo)
    change_in_elo = k_factor * (1 - expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo


def expected_result(elo_a, elo_b):
    return 1.0 / (1 + 10 ** ((elo_b - elo_a) / elo_width))


base_elo = 1000
elo_width = 400
k_factor = 100

elos = []
for season in sorted(set(seeds["Season"])):
    ss_elo = regular_data.loc[regular_data["Season"] == season]
    ss_elo = ss_elo.loc[ss_elo["win"] == 1].reset_index(drop=True)
    teams = set(ss_elo["T1_TeamID"]) | set(ss_elo["T2_TeamID"])
    elo = dict(zip(teams, [base_elo] * len(teams)))
    for i in range(ss_elo.shape[0]):
        w_team, l_team = ss_elo.loc[i, "T1_TeamID"], ss_elo.loc[i, "T2_TeamID"]
        w_elo, l_elo = elo[w_team], elo[l_team]
        w_elo_new, l_elo_new = update_elo(w_elo, l_elo)
        elo[w_team] = w_elo_new
        elo[l_team] = l_elo_new
    elo = pd.DataFrame.from_dict(elo, orient="index").reset_index()
    elo = elo.rename({"index": "TeamID", 0: "elo"}, axis=1)
    elo["Season"] = season
    elos.append(elo)
elos = pd.concat(elos)

elos_T1 = elos.copy().rename({"TeamID": "T1_TeamID", "elo": "T1_elo"}, axis=1)
elos_T2 = elos.copy().rename({"TeamID": "T2_TeamID", "elo": "T2_elo"}, axis=1)
tourney_data = pd.merge(tourney_data, elos_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, elos_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]

print(f"  Elo ratings computed for {len(elos)} team-seasons.")

# ============================================================
# 6. Hardest Features: GLM Team Quality
# ============================================================
print("\n" + "=" * 60)
print("6. Hardest features: GLM team quality...")
print("=" * 60)

regular_data["ST1"] = regular_data.apply(
    lambda t: str(int(t["Season"])) + "/" + str(int(t["T1_TeamID"])), axis=1
)
regular_data["ST2"] = regular_data.apply(
    lambda t: str(int(t["Season"])) + "/" + str(int(t["T2_TeamID"])), axis=1
)
seeds_T1["ST1"] = seeds_T1.apply(
    lambda t: str(int(t["Season"])) + "/" + str(int(t["T1_TeamID"])), axis=1
)
seeds_T2["ST2"] = seeds_T2.apply(
    lambda t: str(int(t["Season"])) + "/" + str(int(t["T2_TeamID"])), axis=1
)

# collect tourney teams
st = set(seeds_T1["ST1"]) | set(seeds_T2["ST2"])
# append non-tourney teams which were able to beat tourney team at least once
st = st | set(
    regular_data.loc[
        (regular_data["T1_Score"] > regular_data["T2_Score"]) & (regular_data["ST2"].isin(st)),
        "ST1",
    ]
)


def team_quality(season, men_women, dt):
    formula = "PointDiff~-1+T1_TeamID+T2_TeamID"
    glm = sm.GLM.from_formula(
        formula=formula,
        data=dt.loc[(dt["Season"] == season) & (dt["men_women"] == men_women), :],
        family=sm.families.Gaussian(),
    ).fit()

    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ["TeamID", "quality"]
    quality["Season"] = season
    quality = quality.loc[quality.TeamID.str.contains("T1_")].reset_index(drop=True)
    quality["TeamID"] = quality["TeamID"].apply(lambda x: x[10:14]).astype(int)
    # Explicitly free the GLM model to avoid memory accumulation
    del glm
    gc.collect()
    return quality


glm_quality = []

dt = regular_data.loc[regular_data["ST1"].isin(st) | regular_data["ST2"].isin(st)].copy()
dt["T1_TeamID"] = dt["T1_TeamID"].astype(str)
dt["T2_TeamID"] = dt["T2_TeamID"].astype(str)
dt.loc[~dt["ST1"].isin(st), "T1_TeamID"] = "0000"
dt.loc[~dt["ST2"].isin(st), "T2_TeamID"] = "0000"
seasons = sorted(set(seeds["Season"]))
for s_year in tqdm.tqdm(seasons, unit="season"):
    if s_year >= 2010:  # min season for women
        try:
            glm_quality.append(team_quality(s_year, 0, dt))
        except Exception as e:
            print(f"  Warning: GLM women season {s_year} skipped: {e}")
    if s_year >= 2003:  # min season for men
        try:
            glm_quality.append(team_quality(s_year, 1, dt))
        except Exception as e:
            print(f"  Warning: GLM men season {s_year} skipped: {e}")

glm_quality = pd.concat(glm_quality).reset_index(drop=True)
# Free dt to reclaim memory (large multi-season filtered dataframe)
del dt
gc.collect()

glm_quality_T1 = glm_quality.copy()
glm_quality_T2 = glm_quality.copy()
glm_quality_T1.columns = ["T1_TeamID", "T1_quality", "Season"]
glm_quality_T2.columns = ["T2_TeamID", "T2_quality", "Season"]
tourney_data = pd.merge(tourney_data, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["diff_quality"] = tourney_data["T1_quality"] - tourney_data["T2_quality"]

print(f"  GLM quality computed. Tourney data shape: {tourney_data.shape}")

# ============================================================
# 7. Meta Features: NuSVC + KNN (cross men/women)
# ============================================================
print("\n" + "=" * 60)
print("7. Meta features: NuSVC + KNN...")
print("=" * 60)

cat_features = [
    "T1_avg_FGM", "T1_avg_FGM3", "T1_avg_FGA3", "T1_avg_FTM", "T1_avg_FTA",
    "T1_avg_OR", "T1_avg_DR", "T1_avg_Ast", "T1_avg_TO", "T1_avg_Stl",
    "T1_avg_opponent_Score", "T1_avg_opponent_FGM", "T1_avg_opponent_FGM3",
    "T1_avg_opponent_FGA3", "T1_avg_opponent_FTM", "T1_avg_opponent_FTA",
    "T1_avg_opponent_OR", "T1_avg_opponent_DR", "T1_avg_opponent_Ast",
    "T1_avg_opponent_TO", "T1_avg_opponent_Stl",
    "T2_avg_FGM", "T2_avg_FGM3", "T2_avg_FGA3", "T2_avg_FTM", "T2_avg_FTA",
    "T2_avg_OR", "T2_avg_DR", "T2_avg_Ast", "T2_avg_TO", "T2_avg_Stl",
    "T2_avg_opponent_Score", "T2_avg_opponent_FGM", "T2_avg_opponent_FGM3",
    "T2_avg_opponent_FGA3", "T2_avg_opponent_FTM", "T2_avg_opponent_FTA",
    "T2_avg_opponent_OR", "T2_avg_opponent_DR", "T2_avg_opponent_Ast",
    "T2_avg_opponent_TO", "T2_avg_opponent_Stl",
]

np.random.seed(0)

# Train on men, predict on women (and vice versa) - cross-gender meta features
# Model 1a/1b: trained on men's data
cb_model1a = make_pipeline(
    StandardScaler(), NuSVC(probability=True, nu=0.6, kernel="poly", gamma="scale", degree=2)
)
cb_model1b = make_pipeline(
    StandardScaler(), KNeighborsRegressor(n_neighbors=100, weights="uniform", p=2, metric="minkowski")
)

x_train = tourney_data.loc[tourney_data["men_women"] == 1, cat_features].values
y_train = tourney_data.loc[tourney_data["men_women"] == 1, "PointDiff"].values
y_train1 = (y_train > 0) * 1.0
cb_model1a.fit(x_train, y_train1)
cb_model1b.fit(x_train, y_train)

x_test = tourney_data.loc[tourney_data["men_women"] != 1, cat_features].values
preds1a = cb_model1a.predict_proba(x_test)[:, 1]
preds1b = cb_model1b.predict(x_test)

# Model 2a/2b: trained on women's data
cb_model2a = make_pipeline(
    StandardScaler(), NuSVC(probability=True, nu=0.6, kernel="poly", gamma="scale", degree=2)
)
cb_model2b = make_pipeline(
    StandardScaler(), KNeighborsRegressor(n_neighbors=100, weights="uniform", p=2, metric="minkowski")
)

x_train = tourney_data.loc[tourney_data["men_women"] != 1, cat_features].values
y_train = tourney_data.loc[tourney_data["men_women"] != 1, "PointDiff"].values
y_train2 = (y_train > 0) * 1.0
cb_model2a.fit(x_train, y_train2)
cb_model2b.fit(x_train, y_train)

x_test = tourney_data.loc[tourney_data["men_women"] == 1, cat_features].values
preds2a = cb_model2a.predict_proba(x_test)[:, 1]
preds2b = cb_model2b.predict(x_test)

# Assign cross-predictions as features
# Note: men's predictions come from women-trained model and vice versa
tourney_data["cb_preds1"] = np.nan
tourney_data.loc[tourney_data["men_women"] == 1, "cb_preds1"] = preds2a
tourney_data.loc[tourney_data["men_women"] != 1, "cb_preds1"] = preds1a

tourney_data["cb_preds2"] = np.nan
tourney_data.loc[tourney_data["men_women"] == 1, "cb_preds2"] = preds2b
tourney_data.loc[tourney_data["men_women"] != 1, "cb_preds2"] = preds1b

print(f"  NuSVC preds: {tourney_data['cb_preds1'].describe()}")
print(f"  KNN preds: {tourney_data['cb_preds2'].describe()}")

# ============================================================
# 8. XGBoost Model (Cauchy loss) - Leave-One-Season-Out
# ============================================================
print("\n" + "=" * 60)
print("8. XGBoost training (LOSO CV)...")
print("=" * 60)

features = [
    ### EASY FEATURES ###
    "men_women",
    "T1_seed",
    "T2_seed",
    "Seed_diff",
    ### MEDIUM FEATURES ###
    "T1_avg_Score",
    "T1_avg_FGA",
    "T1_awins", "T2_awins",
    "T1_away_wins", "T2_away_wins",
    "laplace_matchup",
    "T1_WinRatio14d",
    "T2_WinRatio14d",
    "cb_preds1", "cb_preds2",
    "T1_avg_Blk",
    "T1_avg_PF",
    "T1_avg_opponent_FGA",
    "T1_avg_opponent_Blk",
    "T1_avg_opponent_PF",
    "T1_avg_PointDiff",
    "T2_avg_Score",
    "T2_avg_FGA",
    "T2_avg_Blk",
    "T2_avg_PF",
    "T2_avg_opponent_FGA",
    "T2_avg_opponent_Blk",
    "T2_avg_opponent_PF",
    "T2_avg_PointDiff",
    ### HARD FEATURES ###
    "T1_elo",
    "T2_elo",
    "elo_diff",
    ### HARDEST FEATURES ###
    "T1_quality",
    "T2_quality",
]

print(f"  Number of features: {len(features)}")


# Cauchy objective function
def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000
    x = preds - labels
    x2 = x**2
    c2 = c**2
    grad = x / (x2 / c2 + 1)
    hess = -c2 * (x2 - c2) / (x2 + c2) ** 2
    return grad, hess


xgb_parameters = {
    "eval_metric": "mae",
    "eta": 0.02,
    "subsample": 0.35,
    "colsample_bytree": 0.7,
    "num_parallel_tree": 10,
    "min_child_weight": 40,
    "max_depth": 4,
    "gamma": 10,
}

models = {}
oof_mae = []
oof_preds = []
oof_targets = []
oof_ss = []

# leave-one-season out models
for oof_season in sorted(set(tourney_data.Season)):
    x_train = tourney_data.loc[tourney_data["Season"] != oof_season, features].values
    y_train = tourney_data.loc[tourney_data["Season"] != oof_season, "PointDiff"].values
    x_val = tourney_data.loc[tourney_data["Season"] == oof_season, features].values
    y_val = tourney_data.loc[tourney_data["Season"] == oof_season, "PointDiff"].values
    s_val = tourney_data.loc[tourney_data["Season"] == oof_season, "Season"].values

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)

    models[oof_season] = xgb.train(
        params=xgb_parameters,
        dtrain=dtrain,
        num_boost_round=400,
        obj=cauchyobj,
    )

    preds = models[oof_season].predict(dval)
    mae = mean_absolute_error(y_val, preds)
    print(f"  oof season {oof_season} mae: {mae:.4f}")
    oof_mae.append(mae)
    oof_preds += list(preds)
    oof_targets += list(y_val)
    oof_ss += list(s_val)

print(f"  Average MAE: {np.mean(oof_mae):.4f}")

# ============================================================
# 9. Leaf-based LogisticRegression Calibration (LOSO)
# ============================================================
print("\n" + "=" * 60)
print("9. Leaf-based LogisticRegression calibration (LOSO)...")
print("=" * 60)


def make_strings(x):
    output_list = []
    for i in range(x.shape[0]):
        tmp_str = ""
        for j in range(x.shape[1]):
            tmp_str += str(j) + "_" + str(int(x[i][j])) + " "
        output_list.append(tmp_str)
    return output_list


brier_list = []
logistic_dict = dict()

for oof_season in sorted(set(tourney_data.Season)):
    gc.collect()
    x_train = tourney_data.loc[tourney_data["Season"] != oof_season, features].values
    y_train = tourney_data.loc[tourney_data["Season"] != oof_season, "PointDiff"].values
    x_val = tourney_data.loc[tourney_data["Season"] == oof_season, features].values
    y_val = tourney_data.loc[tourney_data["Season"] == oof_season, "PointDiff"].values

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)

    tmp = models[oof_season].predict(dtrain, pred_leaf=True)
    tmp = make_strings(tmp)
    calibrater = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(solver="liblinear", random_state=0, max_iter=300),
    )
    y_val_bin = (y_train > 0) * 1
    calibrater.fit(tmp, y_val_bin)
    logistic_dict[oof_season] = calibrater

    tmp = models[oof_season].predict(dval, pred_leaf=True)
    tmp = make_strings(tmp)
    preds = calibrater.predict_proba(tmp)[:, 1]
    y_val_bin = (y_val > 0) * 1
    bscore = brier_score_loss(y_val_bin, preds)
    brier_list.append(bscore)
    print(f"  oof season {oof_season} brier: {bscore:.6f}")

avg_brier = np.mean(brier_list)
print(f"\n  Average Brier Score (LOSO): {avg_brier:.6f}")

# Save OOF results
oof_df = pd.DataFrame({
    "Season": oof_ss,
    "PointDiff": oof_targets,
    "margin_pred": oof_preds,
})
oof_df.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)

# ============================================================
# 10. Make Submission
# ============================================================
print("\n" + "=" * 60)
print("10. Making submission...")
print("=" * 60)

# Try Stage2 first, fall back to Stage1
stage2_path = DATA_DIR / "SampleSubmissionStage2.csv"
stage1_path = DATA_DIR / "SampleSubmissionStage1.csv"
if stage2_path.exists():
    X = pd.read_csv(stage2_path)
    print(f"  Using Stage2 submission template: {X.shape}")
elif stage1_path.exists():
    X = pd.read_csv(stage1_path)
    print(f"  Using Stage1 submission template: {X.shape}")
else:
    raise FileNotFoundError("No submission template found.")

# Parse submission IDs
X["Season"] = X["ID"].apply(lambda t: int(t.split("_")[0]))
X["T1_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[1]))
X["T2_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[2]))
# men's TeamIDs start with "1" → men_women=1; women's start with "3" → men_women=0
X["men_women"] = X["T1_TeamID"].apply(lambda t: 1 if str(t)[0] == "1" else 0)

# --- Compute submission-season features if not already in lookups ---
sub_season = X["Season"].max()
print(f"  Submission season: {sub_season}")

# Check if submission season has Elo computed
if sub_season not in set(elos["Season"]):
    print(f"  Computing Elo for season {sub_season}...")
    ss_elo_sub = regular_data.loc[regular_data["Season"] == sub_season]
    ss_elo_sub = ss_elo_sub.loc[ss_elo_sub["win"] == 1].reset_index(drop=True)
    teams_sub = set(ss_elo_sub["T1_TeamID"]) | set(ss_elo_sub["T2_TeamID"])
    elo_sub = dict(zip(teams_sub, [base_elo] * len(teams_sub)))
    for i in range(ss_elo_sub.shape[0]):
        w_team, l_team = ss_elo_sub.loc[i, "T1_TeamID"], ss_elo_sub.loc[i, "T2_TeamID"]
        w_elo_val, l_elo_val = elo_sub[w_team], elo_sub[l_team]
        w_elo_new, l_elo_new = update_elo(w_elo_val, l_elo_val)
        elo_sub[w_team] = w_elo_new
        elo_sub[l_team] = l_elo_new
    elo_sub_df = pd.DataFrame.from_dict(elo_sub, orient="index").reset_index()
    elo_sub_df = elo_sub_df.rename({"index": "TeamID", 0: "elo"}, axis=1)
    elo_sub_df["Season"] = sub_season
    elos = pd.concat([elos, elo_sub_df])
    elos_T1 = elos.copy().rename({"TeamID": "T1_TeamID", "elo": "T1_elo"}, axis=1)
    elos_T2 = elos.copy().rename({"TeamID": "T2_TeamID", "elo": "T2_elo"}, axis=1)

# Check if submission season has GLM quality for BOTH genders
sub_glm = glm_quality[glm_quality["Season"] == sub_season]
sub_glm_men = sub_glm[sub_glm["TeamID"].astype(str).str.startswith("1")]
sub_glm_women = sub_glm[sub_glm["TeamID"].astype(str).str.startswith("3")]
need_men_glm = len(sub_glm_men) == 0
need_women_glm = len(sub_glm_women) == 0

if need_men_glm or need_women_glm:
    print(f"  Computing GLM quality for season {sub_season} (men={need_men_glm}, women={need_women_glm})...")
    dt_sub = regular_data.loc[regular_data["Season"] == sub_season].copy()
    dt_sub["T1_TeamID"] = dt_sub["T1_TeamID"].astype(str)
    dt_sub["T2_TeamID"] = dt_sub["T2_TeamID"].astype(str)
    if need_women_glm:
        try:
            q = team_quality(sub_season, 0, dt_sub)
            glm_quality = pd.concat([glm_quality, q]).reset_index(drop=True)
            print(f"    Women's GLM quality computed: {len(q)} teams")
        except Exception as e:
            print(f"    Warning: GLM women quality failed: {e}")
    if need_men_glm:
        try:
            q = team_quality(sub_season, 1, dt_sub)
            glm_quality = pd.concat([glm_quality, q]).reset_index(drop=True)
            print(f"    Men's GLM quality computed: {len(q)} teams")
        except Exception as e:
            print(f"    Warning: GLM men quality failed: {e}")
    del dt_sub
    glm_quality_T1 = glm_quality.copy()
    glm_quality_T2 = glm_quality.copy()
    glm_quality_T1.columns = ["T1_TeamID", "T1_quality", "Season"]
    glm_quality_T2.columns = ["T2_TeamID", "T2_quality", "Season"]

# Merge features
X = pd.merge(X, ss_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, ss_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, seeds_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, seeds_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, elos_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, elos_T2, on=["Season", "T2_TeamID"], how="left")
X["Seed_diff"] = X["T2_seed"] - X["T1_seed"]
X["elo_diff"] = X["T1_elo"] - X["T2_elo"]
X["diff_quality"] = X["T1_quality"] - X["T2_quality"]

X["laplace_matchup"] = (
    X["Season"].map(str) + "_" + X["T1_TeamID"].map(str) + "_" + X["T2_TeamID"].map(str)
)
X["laplace_matchup"] = X["laplace_matchup"].map(match_dict)

# Filter to teams with key features FIRST (before expensive NuSVC inference)
current_season = X["Season"].max()
seeds_set = set(seeds[seeds["Season"] == current_season]["TeamID"].tolist())
if len(seeds_set) > 0:
    b1 = X["T1_TeamID"].isin(seeds_set)
    b2 = X["T2_TeamID"].isin(seeds_set)
    b3 = b1 & b2
else:
    # No seeds: filter to teams with elo (played regular season games)
    print(f"  No seeds for season {current_season}, using teams with Elo ratings.")
    b3 = X["T1_elo"].notna() & X["T2_elo"].notna()

X_select = X[b3].reset_index(drop=True).copy()
X_not = X[~b3].reset_index(drop=True).copy()
X_not["Pred"] = 0.5

print(f"  Teams with features matchups: {X_select.shape[0]}")
print(f"  Teams without features matchups: {X_not.shape[0]}")

# For teams without seeds, fill with median seed (8) for submission
X_select["T1_seed"] = X_select["T1_seed"].fillna(8)
X_select["T2_seed"] = X_select["T2_seed"].fillna(8)
X_select["Seed_diff"] = X_select["T2_seed"] - X_select["T1_seed"]

# NuSVC/KNN predictions on filtered data only (much faster)
print("  Computing NuSVC/KNN predictions (batched to avoid OOM)...")
X_select["cb_preds1"] = np.nan
X_select["cb_preds2"] = np.nan

BATCH_SIZE = 5000
for mw_val, model_a, model_b in [(0, cb_model1a, cb_model1b), (1, cb_model2a, cb_model2b)]:
    mask = X_select["men_women"] == mw_val
    if mask.sum() > 0:
        idx = X_select.index[mask].tolist()
        preds_a_list, preds_b_list = [], []
        for start in range(0, len(idx), BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            x_batch = X_select.loc[batch_idx, cat_features].values
            preds_a_list.append(model_a.predict_proba(x_batch)[:, 1])
            preds_b_list.append(model_b.predict(x_batch))
            gc.collect()
        X_select.loc[idx, "cb_preds1"] = np.concatenate(preds_a_list)
        X_select.loc[idx, "cb_preds2"] = np.concatenate(preds_b_list)

# Fill missing cb_preds with defaults
X_select["cb_preds1"] = X_select["cb_preds1"].fillna(0.5)
X_select["cb_preds2"] = X_select["cb_preds2"].fillna(0.0)

# Fill remaining NaN in features for X_select
for col in features:
    if X_select[col].isna().any():
        fill_val = tourney_data[col].median() if col in tourney_data.columns else 0
        X_select[col] = X_select[col].fillna(fill_val)

# Run all LOSO models and average predictions (batched to avoid OOM)
loso_seasons = sorted(set(tourney_data.Season))
pred_sum = np.zeros(len(X_select))
LEAF_BATCH = 2000  # batch size for leaf prediction to control memory

print(f"  Running {len(loso_seasons)} LOSO models on {len(X_select)} matchups...")
for oof_season in loso_seasons:
    gc.collect()
    calibrater = logistic_dict[oof_season]
    tfidf = calibrater.named_steps["tfidfvectorizer"]
    lr_step = calibrater.named_steps["logisticregression"]

    for start in range(0, len(X_select), LEAF_BATCH):
        end = min(start + LEAF_BATCH, len(X_select))
        dtest_batch = xgb.DMatrix(X_select.iloc[start:end][features].values)
        leaf_batch = models[oof_season].predict(dtest_batch, pred_leaf=True)
        leaf_strs = make_strings(leaf_batch)
        tfidf_feats = tfidf.transform(leaf_strs)
        batch_probs = lr_step.predict_proba(tfidf_feats)[:, 1]
        pred_sum[start:end] += batch_probs
        del dtest_batch, leaf_batch, leaf_strs, tfidf_feats, batch_probs
    gc.collect()

X_select["Pred"] = pred_sum / len(loso_seasons)

# Combine
X_both = pd.concat(
    [X_select[["ID", "Pred"]], X_not[["ID", "Pred"]]], axis=0
).reset_index(drop=True)

print(f"  Total submission rows: {X_both.shape[0]}")

# ============================================================
# Post-processing: Team probability adjustment (logit-space)
# ============================================================
ADJUST_TEAMS = {
    1276: 0.3,  # Michigan (+0.3 logit boost)
}

def logit_shift(p, delta):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + delta)))

for team_id, delta in ADJUST_TEAMS.items():
    mask_t1 = X_both["ID"].str.contains(f"_{team_id}_")
    mask_t2 = X_both["ID"].str.endswith(f"_{team_id}")
    before_t1 = X_both.loc[mask_t1, "Pred"].mean()
    before_t2 = (1 - X_both.loc[mask_t2, "Pred"]).mean()
    # T1 (lower ID) = team: increase Pred
    X_both.loc[mask_t1, "Pred"] = logit_shift(X_both.loc[mask_t1, "Pred"].values, +delta)
    # T2 (higher ID) = team: decrease opponent's Pred
    X_both.loc[mask_t2, "Pred"] = logit_shift(X_both.loc[mask_t2, "Pred"].values, -delta)
    after_t1 = X_both.loc[mask_t1, "Pred"].mean()
    after_t2 = (1 - X_both.loc[mask_t2, "Pred"]).mean()
    n = mask_t1.sum() + mask_t2.sum()
    print(f"  Team {team_id}: delta={delta:+.1f}, {n} matchups adjusted")
    print(f"    T1 win prob: {before_t1:.4f} → {after_t1:.4f}")
    print(f"    T2 win prob: {before_t2:.4f} → {after_t2:.4f}")

# Also apply to select_predictions
X_select_adj = X_select[["ID", "Pred"]].copy()
for team_id, delta in ADJUST_TEAMS.items():
    mask_t1 = X_select_adj["ID"].str.contains(f"_{team_id}_")
    mask_t2 = X_select_adj["ID"].str.endswith(f"_{team_id}")
    X_select_adj.loc[mask_t1, "Pred"] = logit_shift(X_select_adj.loc[mask_t1, "Pred"].values, +delta)
    X_select_adj.loc[mask_t2, "Pred"] = logit_shift(X_select_adj.loc[mask_t2, "Pred"].values, -delta)

# Save
X_select_adj.to_csv(OUTPUT_DIR / "select_predictions.csv", index=False)
X_both[["ID", "Pred"]].to_csv(OUTPUT_DIR / "submission.csv", index=False)

# Save metrics
with open(OUTPUT_DIR / "metrics.txt", "w") as f:
    f.write(f"Average MAE (LOSO): {np.mean(oof_mae):.6f}\n")
    f.write(f"Average Brier Score (LOSO): {avg_brier:.6f}\n")
    f.write(f"Number of features: {len(features)}\n")
    f.write(f"Number of LOSO seasons: {len(models)}\n")
    for i, season in enumerate(sorted(set(tourney_data.Season))):
        f.write(f"  Season {season} - Brier: {brier_list[i]:.6f}\n")

print("\n" + "=" * 60)
print("Done!")
print(f"  Submission saved to: {OUTPUT_DIR / 'submission.csv'}")
print(f"  Select predictions saved to: {OUTPUT_DIR / 'select_predictions.csv'}")
print(f"  OOF predictions saved to: {OUTPUT_DIR / 'oof_predictions.csv'}")
print(f"  Metrics saved to: {OUTPUT_DIR / 'metrics.txt'}")
print("=" * 60)
