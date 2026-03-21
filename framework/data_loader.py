"""データ読み込みとトーナメント対戦ペア生成"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_all_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """公式コンペデータを読み込む

    Returns:
        辞書: キーは "{M|W}_{データ名}" 形式
    """
    dfs = {}
    for prefix in ["M", "W"]:
        dfs[f"{prefix}_rs_compact"] = pd.read_csv(
            data_dir / f"{prefix}RegularSeasonCompactResults.csv"
        )
        dfs[f"{prefix}_tourney"] = pd.read_csv(
            data_dir / f"{prefix}NCAATourneyCompactResults.csv"
        )
        dfs[f"{prefix}_seeds"] = pd.read_csv(
            data_dir / f"{prefix}NCAATourneySeeds.csv"
        )
        dfs[f"{prefix}_teams"] = pd.read_csv(data_dir / f"{prefix}Teams.csv")
        dfs[f"{prefix}_conf"] = pd.read_csv(
            data_dir / f"{prefix}TeamConferences.csv"
        )

        # DetailedResults は存在する場合のみ
        detail_path = data_dir / f"{prefix}RegularSeasonDetailedResults.csv"
        if detail_path.exists():
            dfs[f"{prefix}_rs_detail"] = pd.read_csv(detail_path)

        # セカンダリトーナメント
        sec_path = data_dir / f"{prefix}SecondaryTourneyCompactResults.csv"
        if sec_path.exists():
            dfs[f"{prefix}_secondary"] = pd.read_csv(sec_path)

        # カンファレンストーナメント
        conf_tourney_path = data_dir / f"{prefix}ConferenceTourneyGames.csv"
        if conf_tourney_path.exists():
            dfs[f"{prefix}_conf_tourney"] = pd.read_csv(conf_tourney_path)

    # Massey (男子のみ)
    massey_path = data_dir / "MMasseyOrdinals.csv"
    if massey_path.exists():
        dfs["M_massey"] = pd.read_csv(massey_path)

    # Coach data (男子のみ)
    coach_path = data_dir / "MTeamCoaches.csv"
    if coach_path.exists():
        dfs["M_coaches"] = pd.read_csv(coach_path)

    return dfs


def build_tourney_matchups(
    tourney_df: pd.DataFrame, gender: str
) -> pd.DataFrame:
    """トーナメント結果から対戦ペアとラベルを生成

    提出形式に合わせて TeamA < TeamB (TeamID の小さい方が TeamA)
    label = 1 なら TeamA の勝ち

    Args:
        tourney_df: トーナメント結果 (WTeamID, LTeamID, Season, ...)
        gender: "M" or "W"

    Returns:
        DataFrame: Season, TeamA, TeamB, label, Gender
    """
    w = tourney_df["WTeamID"].values
    l = tourney_df["LTeamID"].values
    a = np.minimum(w, l)
    b = np.maximum(w, l)
    label = (w == a).astype(int)

    return pd.DataFrame(
        {
            "Season": tourney_df["Season"].values,
            "TeamA": a,
            "TeamB": b,
            "label": label,
            "Gender": gender,
        }
    )


def build_submission_matchups(
    data_dir: Path, stage: str = "stage2"
) -> pd.DataFrame:
    """サンプル提出ファイルから全対戦ペアを生成

    Args:
        data_dir: データディレクトリ
        stage: "stage1" or "stage2"

    Returns:
        DataFrame: Season, TeamA, TeamB, Gender
    """
    filename = (
        "SampleSubmissionStage1.csv" if stage == "stage1" else "SampleSubmissionStage2.csv"
    )
    sample = pd.read_csv(data_dir / filename)
    parsed = sample["ID"].str.split("_", expand=True)
    sample["Season"] = parsed[0].astype(int)
    sample["TeamA"] = parsed[1].astype(int)
    sample["TeamB"] = parsed[2].astype(int)
    # 男子の TeamID < 3000, 女子 >= 3000
    sample["Gender"] = sample["TeamA"].apply(lambda x: "M" if x < 3000 else "W")
    return sample[["Season", "TeamA", "TeamB", "Gender"]]
