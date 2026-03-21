"""Team Rating 実験

攻撃力・守備力・ペースを分離した team rating モデルの実験。
2つの rating 推定方式 (Ridge, Iterative) と、
2つの予測モデル (LogisticRegression, LightGBM) を比較する。

使い方:
    uv run python experiments/team_rating_experiment.py
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    brier_score,
    evaluate_predictions,
    format_evaluation_report,
)
from framework.features.team_rating import TeamRatingFeature
from framework.features.matchup_rating import build_full_matchup_features
from framework.models.logistic import LogisticModel
from framework.models.lightgbm_model import LightGBMModel


def _generate_team_features(
    data: dict[str, pd.DataFrame],
    gender: str,
    max_season: int,
    method: str = "both",
) -> pd.DataFrame:
    """Team Rating 特徴量を生成"""
    gen = TeamRatingFeature(method=method)
    return gen.generate(data, gender, max_season)


def _build_features_for_matchups(
    matchups: pd.DataFrame,
    team_features: pd.DataFrame,
    method_prefixes: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """matchups に対して全 method の matchup 特徴量を構築"""
    all_feature_names = []
    result = matchups.copy()

    for prefix in method_prefixes:
        result, feat_names = build_full_matchup_features(
            result, team_features, method_prefix=prefix
        )
        all_feature_names.extend(feat_names)

    return result, all_feature_names


def run_team_rating_experiment(
    config: ExperimentConfig,
    rating_method: str = "both",
    model_type: str = "logistic",
    verbose: bool = True,
) -> dict:
    """Team Rating 実験を実行

    Args:
        config: 実験設定
        rating_method: "ridge", "iterative", "both"
        model_type: "logistic", "lightgbm"
        verbose: 進捗表示

    Returns:
        実験結果
    """
    start_time = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"=== Team Rating 実験: {config.name} ===")
        print(f"  rating_method={rating_method}, model_type={model_type}")
        print(f"  出力先: {output_dir}")

    # データ読み込み
    data = load_all_data(config.data_dir)

    # rating method に応じた prefix
    if rating_method == "ridge":
        method_prefixes = ["ridge"]
    elif rating_method == "iterative":
        method_prefixes = ["iter"]
    else:
        method_prefixes = ["ridge", "iter"]

    # トーナメント対戦ペア作成
    matchups_list = []
    for gender in ["M", "W"]:
        m = build_tourney_matchups(data[f"{gender}_tourney"], gender)
        matchups_list.append(m)
    all_matchups = pd.concat(matchups_list, ignore_index=True)

    # CV 分割
    splits = season_cv_split(all_matchups, config.val_seasons, config.min_train_season)

    # gender_mode に応じた実行モード
    if config.gender_mode == "separate":
        gender_modes = [("M",), ("W",)]
    elif config.gender_mode == "combined":
        gender_modes = [("M", "W")]
    else:  # "both"
        gender_modes = [("M", "W"), ("M",), ("W",)]

    oof_records = []

    for train_matchups, val_matchups, val_season in splits:
        if verbose:
            print(f"\n  --- Val Season: {val_season} ---")

        for gender_tuple in gender_modes:
            mode_label = "combined" if len(gender_tuple) > 1 else gender_tuple[0]

            # gender フィルタ
            tr = train_matchups[train_matchups["Gender"].isin(gender_tuple)]
            va = val_matchups[val_matchups["Gender"].isin(gender_tuple)]
            if len(tr) == 0 or len(va) == 0:
                continue

            # 特徴量生成 (各 gender について生成して結合)
            team_feat_list = []
            for g in gender_tuple:
                tf = _generate_team_features(data, g, val_season, rating_method)
                team_feat_list.append(tf)
            team_features = pd.concat(team_feat_list, ignore_index=True)

            # matchup 特徴量構築
            tr_features, feat_cols = _build_features_for_matchups(
                tr, team_features, method_prefixes
            )
            va_features, _ = _build_features_for_matchups(
                va, team_features, method_prefixes
            )

            # モデル作成
            if model_type == "lightgbm":
                model = LightGBMModel(num_boost_round=500, n_ensemble=3)
            else:
                model = LogisticModel(C=1.0)

            # NaN 処理
            if model.handles_nan:
                X_train = tr_features[feat_cols].reset_index(drop=True)
                y_train = tr["label"].values
                X_val = va_features[feat_cols].reset_index(drop=True)
                y_val = va["label"].values
                va_info = va.reset_index(drop=True)
            else:
                valid_train = tr_features[feat_cols].dropna().index
                X_train = tr_features[feat_cols].iloc[valid_train].reset_index(drop=True)
                y_train = tr["label"].values[valid_train]

                valid_val = va_features[feat_cols].dropna().index
                X_val = va_features[feat_cols].iloc[valid_val].reset_index(drop=True)
                y_val = va["label"].values[valid_val]
                va_info = va.iloc[valid_val].reset_index(drop=True)

            if len(X_train) == 0 or len(X_val) == 0:
                continue

            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)
            preds = np.clip(preds, config.clip_min, config.clip_max)

            bs = brier_score(y_val, preds)
            if verbose:
                print(f"    mode={mode_label}: train={len(X_train)}, val={len(X_val)}, Brier={bs:.4f}")

            for i in range(len(X_val)):
                oof_records.append({
                    "Season": int(va_info.iloc[i]["Season"]),
                    "TeamA": int(va_info.iloc[i]["TeamA"]),
                    "TeamB": int(va_info.iloc[i]["TeamB"]),
                    "Gender": va_info.iloc[i]["Gender"],
                    "label": int(y_val[i]),
                    "pred": float(preds[i]),
                    "model_mode": mode_label,
                    "val_season": val_season,
                })

    # 全体評価
    if verbose:
        print(f"\n=== 全体評価 ===")

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

    # Rating 分布統計
    if verbose:
        print(f"\n=== Rating 分布統計 ===")
    rating_stats = _compute_rating_stats(data, config, rating_method, method_prefixes)
    if verbose:
        for prefix, stats in rating_stats.items():
            print(f"\n  [{prefix}]")
            for col, s in stats.items():
                print(f"    {col}: mean={s['mean']:.2f}, std={s['std']:.2f}, "
                      f"min={s['min']:.2f}, max={s['max']:.2f}")

    # Submission 生成
    if verbose:
        print(f"\n=== Submission 生成 ===")
    submission = _generate_submission(
        config, data, rating_method, method_prefixes, model_type,
        all_matchups, verbose
    )

    # Feature importance (LightGBM の場合)
    feature_importance = None
    if model_type == "lightgbm":
        # 最後のモデルから取得 (概要として)
        feature_importance = model.get_params().get("feature_importance", {})

    # 結果保存
    elapsed = time.time() - start_time
    summary = {
        "experiment_name": config.name,
        "rating_method": rating_method,
        "model_type": model_type,
        "config": {
            "val_seasons": config.val_seasons,
            "min_train_season": config.min_train_season,
            "gender_mode": config.gender_mode,
        },
        "feature_columns": [f for f in (
            _get_feature_names(method_prefixes)
        )],
        "results": experiment_results,
        "rating_stats": rating_stats,
        "feature_importance": feature_importance,
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    if verbose:
        print(f"\n=== 完了 ({elapsed:.1f}秒) ===")

    return summary


def _get_feature_names(method_prefixes: list[str]) -> list[str]:
    """method_prefix に応じた特徴量名リストを返す"""
    names = []
    for prefix in method_prefixes:
        names.extend([
            f"{prefix}_off_A_minus_def_B",
            f"{prefix}_off_B_minus_def_A",
            f"{prefix}_efficiency_diff",
            f"{prefix}_net_diff",
            f"{prefix}_pace_avg",
            f"{prefix}_pace_diff",
            f"{prefix}_consistency_diff",
        ])
    return names


def _compute_rating_stats(
    data: dict[str, pd.DataFrame],
    config: ExperimentConfig,
    rating_method: str,
    method_prefixes: list[str],
) -> dict:
    """Rating の分布統計を計算"""
    stats = {}
    for gender in ["M", "W"]:
        tf = _generate_team_features(data, gender, config.submission_season, rating_method)
        for prefix in method_prefixes:
            key = f"{prefix}_{gender}"
            cols = [c for c in tf.columns if c.startswith(prefix) and c != f"{prefix}_home_adv_off" and c != f"{prefix}_home_adv"]
            stats[key] = {}
            for col in cols:
                vals = tf[col].dropna()
                if len(vals) > 0:
                    stats[key][col] = {
                        "mean": float(vals.mean()),
                        "std": float(vals.std()),
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                        "median": float(vals.median()),
                    }
    return stats


def _generate_submission(
    config: ExperimentConfig,
    data: dict[str, pd.DataFrame],
    rating_method: str,
    method_prefixes: list[str],
    model_type: str,
    all_matchups: pd.DataFrame,
    verbose: bool,
) -> pd.DataFrame:
    """Submission を生成"""
    output_dir = config.output_dir
    sub_matchups = build_submission_matchups(config.data_dir, config.submission_stage)
    max_season = config.submission_season

    results = []
    for gender in ["M", "W"]:
        g_sub = sub_matchups[sub_matchups["Gender"] == gender]
        if len(g_sub) == 0:
            continue

        # 特徴量生成
        team_features = _generate_team_features(data, gender, max_season, rating_method)

        # 学習データ
        g_train = all_matchups[
            (all_matchups["Gender"] == gender)
            & (all_matchups["Season"] < max_season)
        ]
        if len(g_train) == 0:
            continue

        tr_features, feat_cols = _build_features_for_matchups(
            g_train, team_features, method_prefixes
        )
        sub_features, _ = _build_features_for_matchups(
            g_sub, team_features, method_prefixes
        )

        # モデル
        if model_type == "lightgbm":
            model = LightGBMModel(num_boost_round=500, n_ensemble=3)
        else:
            model = LogisticModel(C=1.0)

        if model.handles_nan:
            X_train = tr_features[feat_cols].reset_index(drop=True)
            y_train = g_train["label"].values
        else:
            valid_idx = tr_features[feat_cols].dropna().index
            X_train = tr_features[feat_cols].iloc[valid_idx].reset_index(drop=True)
            y_train = g_train["label"].values[valid_idx]

        if len(X_train) == 0:
            continue

        model.fit(X_train, y_train)

        X_sub = sub_features[feat_cols]
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


# =============================================================================
# メイン: 複数条件を比較実行
# =============================================================================

def main():
    """全条件を実行して比較"""
    print("=" * 60)
    print("Team Rating 実験")
    print("=" * 60)

    conditions = [
        # (name, rating_method, model_type)
        ("team_rating_ridge_logistic", "ridge", "logistic"),
        ("team_rating_iter_logistic", "iterative", "logistic"),
        ("team_rating_both_logistic", "both", "logistic"),
        ("team_rating_ridge_lgbm", "ridge", "lightgbm"),
        ("team_rating_iter_lgbm", "iterative", "lightgbm"),
        ("team_rating_both_lgbm", "both", "lightgbm"),
    ]

    all_results = {}

    for name, rating_method, model_type in conditions:
        print(f"\n{'=' * 60}")
        config = ExperimentConfig(
            name=name,
            gender_mode="both",
        )

        result = run_team_rating_experiment(
            config=config,
            rating_method=rating_method,
            model_type=model_type,
            verbose=True,
        )
        all_results[name] = result

    # 比較サマリー
    print(f"\n{'=' * 60}")
    print("=== 比較サマリー ===")
    print(f"{'=' * 60}")
    print(f"{'実験名':<40} {'mode':<12} {'Brier':>8} {'Acc':>8} {'N':>6}")
    print("-" * 76)

    for name, result in all_results.items():
        for mode, eval_result in result["results"].items():
            ov = eval_result["overall"]
            print(f"{name:<40} {mode:<12} {ov['brier_score']:>8.4f} "
                  f"{ov['accuracy']:>8.4f} {ov['n_samples']:>6}")


if __name__ == "__main__":
    main()
