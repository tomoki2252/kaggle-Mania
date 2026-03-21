"""実験ランナー - 特徴量生成からCV評価・submission生成まで"""

from __future__ import annotations

import inspect
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from framework.config import ExperimentConfig
from framework.cv import season_cv_split
from framework.data_loader import (
    build_submission_matchups,
    build_tourney_matchups,
    load_all_data,
)
from framework.evaluation import (
    evaluate_predictions,
    format_evaluation_report,
)
from framework.features.base import FeatureGenerator
from framework.models.base import ModelWrapper


def _accepts_gender(factory: Callable) -> bool:
    """model_factory が gender キーワード引数を受け取るか判定"""
    try:
        sig = inspect.signature(factory)
        return "gender" in sig.parameters
    except (ValueError, TypeError):
        return False


def _call_model_factory(factory: Callable, gender: str) -> ModelWrapper:
    """model_factory を呼び出す (gender対応を自動判定)"""
    if _accepts_gender(factory):
        return factory(gender=gender)
    return factory()


def _build_matchup_features(
    matchups: pd.DataFrame,
    team_features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """対戦ペアに対してチーム特徴量の差分を計算する

    Args:
        matchups: Season, TeamA, TeamB を持つ対戦ペア
        team_features: Season, TeamID + 各特徴量
        feature_columns: 特徴量カラム名リスト

    Returns:
        matchups に diff_* カラムが追加された DataFrame
    """
    result = matchups.copy()

    # TeamA の特徴量を結合
    rename_a = {c: f"A_{c}" for c in feature_columns}
    result = result.merge(
        team_features[["Season", "TeamID"] + feature_columns].rename(
            columns={**rename_a, "TeamID": "TeamA_ID"}
        ),
        left_on=["Season", "TeamA"],
        right_on=["Season", "TeamA_ID"],
        how="left",
    ).drop(columns=["TeamA_ID"], errors="ignore")

    # TeamB の特徴量を結合
    rename_b = {c: f"B_{c}" for c in feature_columns}
    result = result.merge(
        team_features[["Season", "TeamID"] + feature_columns].rename(
            columns={**rename_b, "TeamID": "TeamB_ID"}
        ),
        left_on=["Season", "TeamB"],
        right_on=["Season", "TeamB_ID"],
        how="left",
    ).drop(columns=["TeamB_ID"], errors="ignore")

    # 差分特徴量を計算
    diff_cols = []
    for c in feature_columns:
        diff_name = f"diff_{c}"
        result[diff_name] = result[f"A_{c}"] - result[f"B_{c}"]
        diff_cols.append(diff_name)

    # A_/B_ カラムを削除
    drop_cols = [f"A_{c}" for c in feature_columns] + [f"B_{c}" for c in feature_columns]
    result = result.drop(columns=drop_cols)

    return result, diff_cols


def _generate_all_features(
    feature_generators: list[FeatureGenerator],
    data: dict[str, pd.DataFrame],
    gender: str,
    max_season: int,
) -> tuple[pd.DataFrame, list[str]]:
    """全特徴量ジェネレータを実行して結合する"""
    all_feature_cols = []
    team_features = None

    for gen in feature_generators:
        feat = gen.generate(data, gender, max_season)
        cols = gen.feature_columns
        all_feature_cols.extend(cols)

        if team_features is None:
            team_features = feat
        else:
            team_features = team_features.merge(
                feat, on=["Season", "TeamID"], how="outer"
            )

    return team_features, all_feature_cols


def run_experiment(
    config: ExperimentConfig,
    feature_generators: list[FeatureGenerator],
    model_factory: Callable,
    verbose: bool = True,
    matchup_feature_fn: Callable | None = None,
) -> dict:
    """実験を実行する

    Args:
        config: 実験設定
        feature_generators: 特徴量ジェネレータのリスト
        model_factory: ModelWrapper を返すファクトリ関数 (gender_mode ごとに新しいインスタンス)
        verbose: 進捗表示
        matchup_feature_fn: 対戦ペアDataFrameに追加の特徴量を付与する関数
            (df, diff_cols) -> (df, new_diff_cols) を返す

    Returns:
        実験結果の辞書
    """
    start_time = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"=== 実験: {config.name} ===")
        print(f"出力先: {output_dir}")

    # === 1. データ読み込み ===
    if verbose:
        print("\n[1/5] データ読み込み...")
    data = load_all_data(config.data_dir)

    # === 2. トーナメント対戦ペア作成 ===
    if verbose:
        print("[2/5] トーナメント対戦ペア作成...")
    matchups_list = []
    for gender in ["M", "W"]:
        m = build_tourney_matchups(data[f"{gender}_tourney"], gender)
        matchups_list.append(m)
    all_matchups = pd.concat(matchups_list, ignore_index=True)

    # === 3. CV実行 ===
    if verbose:
        print(f"[3/5] Season-based CV (val_seasons={config.val_seasons})...")

    splits = season_cv_split(
        all_matchups, config.val_seasons, config.min_train_season
    )

    oof_records = []  # OOF prediction を蓄積
    cv_results = []

    for train_matchups, val_matchups, val_season in splits:
        if verbose:
            print(f"\n  --- Val Season: {val_season} ---")

        # gender_mode に応じてモデルを分ける
        if config.gender_mode == "separate":
            genders_to_train = ["M", "W"]
        elif config.gender_mode == "combined":
            genders_to_train = ["combined"]
        else:  # "both" - 両方実行して比較
            genders_to_train = ["combined", "M", "W"]

        for mode in genders_to_train:
            # 学習・検証データを gender でフィルタ
            if mode in ("M", "W"):
                tr = train_matchups[train_matchups["Gender"] == mode]
                va = val_matchups[val_matchups["Gender"] == mode]
                if len(tr) == 0 or len(va) == 0:
                    continue
                genders_for_features = [mode]
            else:
                tr = train_matchups
                va = val_matchups
                genders_for_features = ["M", "W"]

            # 特徴量生成 (val_season のトーナメント情報は使わない)
            all_team_features = []
            all_feature_cols = None
            for g in genders_for_features:
                team_feat, feat_cols = _generate_all_features(
                    feature_generators, data, g, val_season
                )
                all_team_features.append(team_feat)
                all_feature_cols = feat_cols

            team_features = pd.concat(all_team_features, ignore_index=True)

            # 対戦特徴量を構築
            tr_features, diff_cols = _build_matchup_features(
                tr, team_features, all_feature_cols
            )
            va_features, _ = _build_matchup_features(
                va, team_features, all_feature_cols
            )

            # カスタム matchup 特徴量
            if matchup_feature_fn is not None:
                tr_features, diff_cols = matchup_feature_fn(tr_features, diff_cols)
                va_features, _ = matchup_feature_fn(va_features, diff_cols)

            # モデル作成 (handles_nan を確認するため先に作成)
            model = _call_model_factory(model_factory, mode)

            # NaN 処理: モデルが NaN を扱えない場合のみ除外
            if model.handles_nan:
                X_train = tr_features[diff_cols].reset_index(drop=True)
                y_train = tr["label"].values
                X_val = va_features[diff_cols].reset_index(drop=True)
                y_val = va["label"].values
                va_info = va.reset_index(drop=True)
            else:
                X_train = tr_features[diff_cols].dropna()
                y_train = tr["label"].values[X_train.index]
                X_train = X_train.reset_index(drop=True)

                valid_idx = va_features[diff_cols].dropna().index
                X_val = va_features[diff_cols].iloc[valid_idx].reset_index(drop=True)
                y_val = va["label"].values[valid_idx]
                va_info = va.iloc[valid_idx].reset_index(drop=True)

            if len(X_train) == 0 or len(X_val) == 0:
                continue
            model.fit(X_train, y_train)

            # 予測
            preds = model.predict_proba(X_val)
            preds = np.clip(preds, config.clip_min, config.clip_max)

            # OOF record
            for i in range(len(X_val)):
                oof_records.append({
                    "Season": int(va_info.iloc[i]["Season"]),
                    "TeamA": int(va_info.iloc[i]["TeamA"]),
                    "TeamB": int(va_info.iloc[i]["TeamB"]),
                    "Gender": va_info.iloc[i]["Gender"],
                    "label": int(y_val[i]),
                    "pred": float(preds[i]),
                    "model_mode": mode,
                    "val_season": val_season,
                })

            if verbose:
                from framework.evaluation import brier_score
                bs = brier_score(y_val, preds)
                print(f"    mode={mode}: train={len(X_train)}, val={len(X_val)}, Brier={bs:.4f}")

    # === 4. 全体評価 ===
    if verbose:
        print(f"\n[4/5] 全体評価...")

    oof_df = pd.DataFrame(oof_records)
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    experiment_results = {}
    for mode in oof_df["model_mode"].unique():
        mode_df = oof_df[oof_df["model_mode"] == mode]
        eval_result = evaluate_predictions(
            y_true=mode_df["label"].values,
            y_pred=mode_df["pred"].values,
            seasons=mode_df["val_season"].values,
            genders=mode_df["Gender"].values,
        )
        experiment_results[mode] = eval_result

        if verbose:
            print(f"\n  [{mode}]")
            print(f"  {format_evaluation_report(eval_result)}")

    # === 5. Submission 生成 ===
    if verbose:
        print(f"\n[5/5] Submission 生成...")

    submission = _generate_submission(
        config, data, feature_generators, model_factory, all_matchups, verbose,
        matchup_feature_fn,
    )

    # === 結果保存 ===
    elapsed = time.time() - start_time

    summary = {
        "experiment_name": config.name,
        "config": {
            "val_seasons": config.val_seasons,
            "min_train_season": config.min_train_season,
            "gender_mode": config.gender_mode,
            "clip_min": config.clip_min,
            "clip_max": config.clip_max,
        },
        "features": [g.name for g in feature_generators],
        "feature_columns": [f"diff_{c}" for gen in feature_generators for c in gen.feature_columns],
        "model": _call_model_factory(model_factory, "M").name,
        "results": experiment_results,
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    if verbose:
        print(f"\n=== 完了 ({elapsed:.1f}秒) ===")
        print(f"結果: {output_dir / 'results.json'}")
        print(f"OOF: {output_dir / 'oof_predictions.csv'}")
        print(f"Submission: {output_dir / 'submission.csv'}")

    return summary


def _generate_submission(
    config: ExperimentConfig,
    data: dict[str, pd.DataFrame],
    feature_generators: list[FeatureGenerator],
    model_factory: Callable,
    all_matchups: pd.DataFrame,
    verbose: bool,
    matchup_feature_fn: Callable | None = None,
) -> pd.DataFrame:
    """submission ファイルを生成する

    全学習データでモデルを再学習し、全対戦ペアの予測を行う。
    gender_mode="separate" の場合は男女別モデルで予測する。
    """
    output_dir = config.output_dir
    sub_matchups = build_submission_matchups(config.data_dir, config.submission_stage)

    # submission用: 全データで学習 (submission_season より前の全データ)
    max_season_for_sub = config.submission_season

    results = []
    for gender in ["M", "W"]:
        g_sub = sub_matchups[sub_matchups["Gender"] == gender]
        if len(g_sub) == 0:
            continue

        # 特徴量生成
        team_feat, feat_cols = _generate_all_features(
            feature_generators, data, gender, max_season_for_sub
        )

        # 学習データ (全トーナメント結果)
        g_train = all_matchups[
            (all_matchups["Gender"] == gender)
            & (all_matchups["Season"] < max_season_for_sub)
        ]
        if len(g_train) == 0:
            continue

        tr_features, diff_cols = _build_matchup_features(g_train, team_feat, feat_cols)
        sub_features, _ = _build_matchup_features(g_sub, team_feat, feat_cols)

        # カスタム matchup 特徴量
        if matchup_feature_fn is not None:
            tr_features, diff_cols = matchup_feature_fn(tr_features, diff_cols)
            sub_features, _ = matchup_feature_fn(sub_features, diff_cols)

        # モデル学習
        model = _call_model_factory(model_factory, gender)

        if model.handles_nan:
            X_train = tr_features[diff_cols].reset_index(drop=True)
            y_train = g_train["label"].values
        else:
            X_train = tr_features[diff_cols].dropna()
            y_train = g_train["label"].values[X_train.index]
            X_train = X_train.reset_index(drop=True)

        if len(X_train) == 0:
            continue

        model.fit(X_train, y_train)

        # 予測 (NaN を含む行は 0.5 にフォールバック)
        X_sub = sub_features[diff_cols]
        if model.handles_nan:
            preds = model.predict_proba(X_sub)
        else:
            has_nan = X_sub.isna().any(axis=1)
            preds = np.full(len(X_sub), 0.5)
            if (~has_nan).sum() > 0:
                preds[~has_nan] = model.predict_proba(X_sub[~has_nan])
        preds = np.clip(preds, config.clip_min, config.clip_max)

        sub_features = sub_features.copy()
        sub_features["Pred"] = preds
        results.append(sub_features[["Season", "TeamA", "TeamB", "Pred"]])

    if not results:
        if verbose:
            print("  Warning: submission データが生成されませんでした")
        return pd.DataFrame()

    pred_df = pd.concat(results, ignore_index=True)
    pred_df["ID"] = (
        pred_df["Season"].astype(str) + "_"
        + pred_df["TeamA"].astype(str) + "_"
        + pred_df["TeamB"].astype(str)
    )

    # サンプル提出ファイルの ID 順に並べる
    filename = (
        "SampleSubmissionStage1.csv"
        if config.submission_stage == "stage1"
        else "SampleSubmissionStage2.csv"
    )
    sample = pd.read_csv(config.data_dir / filename)
    submission = sample[["ID"]].merge(pred_df[["ID", "Pred"]], on="ID", how="left")
    submission["Pred"] = submission["Pred"].fillna(0.5)

    submission.to_csv(output_dir / "submission.csv", index=False)

    if verbose:
        print(f"  Submission: {len(submission)} 行")
        print(f"  Pred stats: mean={submission['Pred'].mean():.4f}, "
              f"std={submission['Pred'].std():.4f}")

    return submission
