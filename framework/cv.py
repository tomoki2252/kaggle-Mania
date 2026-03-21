"""Season-based Cross Validation"""

import pandas as pd


def season_cv_split(
    df: pd.DataFrame,
    val_seasons: list[int],
    min_train_season: int = 2003,
) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    """シーズンベースのCV分割を生成

    各 val_season に対して、それより前のシーズンを学習データとする。
    リーク防止: val_season のトーナメント結果は学習に含まれない。

    Args:
        df: 全対戦データ (Season カラムを含む)
        val_seasons: バリデーションに使うシーズンのリスト
        min_train_season: 学習データの最小シーズン

    Yields:
        (train_df, val_df, val_season) のタプル
    """
    splits = []
    for val_season in sorted(val_seasons):
        train_mask = (df["Season"] >= min_train_season) & (df["Season"] < val_season)
        val_mask = df["Season"] == val_season

        train_df = df[train_mask]
        val_df = df[val_mask]

        if len(train_df) == 0 or len(val_df) == 0:
            continue

        splits.append((train_df, val_df, val_season))

    return splits
