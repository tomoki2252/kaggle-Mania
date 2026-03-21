"""Matchup Rating 特徴量ビルダー

Team Rating から対戦ペアの特徴量を構築する。
runner の matchup_feature_fn として使用する。

生成する matchup 特徴量:
- off_A_minus_def_B: TeamA の攻撃力 - TeamB の守備力
- off_B_minus_def_A: TeamB の攻撃力 - TeamA の守備力
- pace_interaction: 両チームの pace の平均 (試合ペースの推定)
- net_rating_diff: TeamA と TeamB の net_rating の差
- efficiency_diff: (off_A - def_B) - (off_B - def_A)
"""

import pandas as pd


def build_matchup_rating_features(
    df: pd.DataFrame,
    diff_cols: list[str],
    method_prefix: str = "ridge",
) -> tuple[pd.DataFrame, list[str]]:
    """対戦ペア DataFrame にマッチアップ rating 特徴量を追加する。

    runner の matchup_feature_fn として使用するため、
    (df, diff_cols) -> (df, new_diff_cols) のシグネチャに合わせる。

    前提: df には diff_{prefix}_off, diff_{prefix}_def 等が存在する。
    """
    df = df.copy()
    new_cols = list(diff_cols)

    off_key = f"diff_{method_prefix}_off"
    def_key = f"diff_{method_prefix}_def"
    net_key = f"diff_{method_prefix}_net"
    pace_key = f"diff_{method_prefix}_pace"
    consistency_key = f"diff_{method_prefix}_consistency"

    # これらの diff カラムは A - B の差分。
    # diff_off = A_off - B_off, diff_def = A_def - B_def
    # 必要なのは:
    #   off_A_minus_def_B = A_off - B_def (A の攻撃が B の守備をどれだけ上回るか)
    #   off_B_minus_def_A = B_off - A_def (B の攻撃が A の守備をどれだけ上回るか)
    # これらは直接 diff からは計算できないので、
    # A_*, B_* のカラムを直接参照する必要がある。

    # ただし runner の _build_matchup_features は diff_* しか返さない。
    # => 実験スクリプト側で直接 team_features を join して計算する方が良い。
    # ここでは diff_* から近似的に計算する。

    # net_rating_diff は diff_net そのもの (既に存在)
    # efficiency_diff = (off_A - def_B) - (off_B - def_A) = diff_off - diff_def
    if off_key in df.columns and def_key in df.columns:
        eff_diff_name = f"{method_prefix}_efficiency_diff"
        df[eff_diff_name] = df[off_key] - df[def_key]
        new_cols.append(eff_diff_name)

    return df, new_cols


def build_full_matchup_features(
    matchups: pd.DataFrame,
    team_features: pd.DataFrame,
    method_prefix: str = "ridge",
) -> tuple[pd.DataFrame, list[str]]:
    """team_features を直接 join して matchup 特徴量を構築する。

    runner の _build_matchup_features を使わず、独自に特徴量を構築する。
    off_A - def_B 等の cross matchup 特徴量を正確に計算できる。

    Args:
        matchups: Season, TeamA, TeamB, label, Gender
        team_features: Season, TeamID, {prefix}_off, {prefix}_def, etc.
        method_prefix: "ridge" or "iter"

    Returns:
        (df_with_features, feature_column_names)
    """
    off_col = f"{method_prefix}_off"
    def_col = f"{method_prefix}_def"
    net_col = f"{method_prefix}_net"
    pace_col = f"{method_prefix}_pace"
    consistency_col = f"{method_prefix}_consistency"

    rating_cols = [off_col, def_col, net_col, pace_col, consistency_col]
    available_cols = [c for c in rating_cols if c in team_features.columns]

    result = matchups.copy()

    # TeamA の rating を結合
    rename_a = {c: f"A_{c}" for c in available_cols}
    result = result.merge(
        team_features[["Season", "TeamID"] + available_cols].rename(
            columns={**rename_a, "TeamID": "TeamA_ID"}
        ),
        left_on=["Season", "TeamA"],
        right_on=["Season", "TeamA_ID"],
        how="left",
    ).drop(columns=["TeamA_ID"], errors="ignore")

    # TeamB の rating を結合
    rename_b = {c: f"B_{c}" for c in available_cols}
    result = result.merge(
        team_features[["Season", "TeamID"] + available_cols].rename(
            columns={**rename_b, "TeamID": "TeamB_ID"}
        ),
        left_on=["Season", "TeamB"],
        right_on=["Season", "TeamB_ID"],
        how="left",
    ).drop(columns=["TeamB_ID"], errors="ignore")

    # Matchup 特徴量を計算
    feature_names = []

    if off_col in available_cols and def_col in available_cols:
        # off_A - def_B: A の攻撃力が B の守備力をどれだけ上回るか
        name = f"{method_prefix}_off_A_minus_def_B"
        result[name] = result[f"A_{off_col}"] - result[f"B_{def_col}"]
        feature_names.append(name)

        # off_B - def_A: B の攻撃力が A の守備力をどれだけ上回るか
        name = f"{method_prefix}_off_B_minus_def_A"
        result[name] = result[f"B_{off_col}"] - result[f"A_{def_col}"]
        feature_names.append(name)

        # efficiency_diff: (off_A - def_B) - (off_B - def_A)
        name = f"{method_prefix}_efficiency_diff"
        result[name] = (
            (result[f"A_{off_col}"] - result[f"B_{def_col}"])
            - (result[f"B_{off_col}"] - result[f"A_{def_col}"])
        )
        feature_names.append(name)

    if net_col in available_cols:
        # net_rating_diff: A の net - B の net
        name = f"{method_prefix}_net_diff"
        result[name] = result[f"A_{net_col}"] - result[f"B_{net_col}"]
        feature_names.append(name)

    if pace_col in available_cols:
        # pace_interaction: 両チームの pace の平均
        name = f"{method_prefix}_pace_avg"
        result[name] = (result[f"A_{pace_col}"] + result[f"B_{pace_col}"]) / 2
        feature_names.append(name)

        # pace_diff
        name = f"{method_prefix}_pace_diff"
        result[name] = result[f"A_{pace_col}"] - result[f"B_{pace_col}"]
        feature_names.append(name)

    if consistency_col in available_cols:
        # consistency_diff
        name = f"{method_prefix}_consistency_diff"
        result[name] = result[f"A_{consistency_col}"] - result[f"B_{consistency_col}"]
        feature_names.append(name)

    # A_*, B_* の中間カラムを削除
    drop_cols = [f"A_{c}" for c in available_cols] + [f"B_{c}" for c in available_cols]
    result = result.drop(columns=drop_cols)

    return result, feature_names
