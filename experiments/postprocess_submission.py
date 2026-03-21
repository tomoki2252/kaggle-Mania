"""
提出ファイルのポスト処理スクリプト
パイプライン再実行なしに、チームごとの勝率をロジット空間で調整する。

前提:
  先に gold_medal_mike_kim.py を実行し、submission/ に以下が生成されていること:
    - submission.csv (全マッチアップ予測)
    - select_predictions.csv (トーナメントシード付きチームのみ, 任意)

使い方:
  uv run python experiments/postprocess_submission.py

調整設定:
  ADJUST_TEAMS に {TeamID: delta} を追加・変更するだけ。
  delta > 0 → そのチームの勝率を上げる
  delta < 0 → そのチームの勝率を下げる
  delta の目安: 0.1=微小, 0.3=中程度, 0.5=やや強め
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 設定
# ============================================================
SUBMISSION_DIR = Path(__file__).resolve().parent.parent / "submission"

# チームIDとdelta値の設定 (delta > 0 で勝率UP, < 0 で勝率DOWN)
ADJUST_TEAMS = {
    1276: 0.3,  # Michigan (+0.3 logit boost)
}

# 処理対象ファイル: (入力, 出力) のリスト
FILES = [
    ("submission.csv", "submission_adjusted.csv"),
    ("select_predictions.csv", "select_predictions_adjusted.csv"),
]


# ============================================================
# 処理
# ============================================================
def logit_shift(p: np.ndarray, delta: float) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + delta)))


def adjust(df: pd.DataFrame, team_id: int, delta: float, verbose: bool = False) -> pd.DataFrame:
    df = df.copy()
    mask_t1 = df["ID"].str.contains(f"_{team_id}_")
    mask_t2 = df["ID"].str.endswith(f"_{team_id}")

    if verbose:
        before_t1 = df.loc[mask_t1, "Pred"].mean() if mask_t1.any() else float("nan")
        before_t2 = (1 - df.loc[mask_t2, "Pred"]).mean() if mask_t2.any() else float("nan")

    df.loc[mask_t1, "Pred"] = logit_shift(df.loc[mask_t1, "Pred"].values, +delta)
    df.loc[mask_t2, "Pred"] = logit_shift(df.loc[mask_t2, "Pred"].values, -delta)

    if verbose:
        after_t1 = df.loc[mask_t1, "Pred"].mean() if mask_t1.any() else float("nan")
        after_t2 = (1 - df.loc[mask_t2, "Pred"]).mean() if mask_t2.any() else float("nan")
        n = mask_t1.sum() + mask_t2.sum()
        print(f"  Team {team_id} (delta={delta:+.1f}): {n} matchups調整")
        print(f"    T1側 平均勝率: {before_t1:.4f} → {after_t1:.4f}")
        print(f"    T2側 平均勝率: {before_t2:.4f} → {after_t2:.4f}")

    return df


def main():
    print("=" * 50)
    print("Submission Post-processing")
    print("=" * 50)

    processed = 0
    for in_name, out_name in FILES:
        in_path = SUBMISSION_DIR / in_name
        out_path = SUBMISSION_DIR / out_name

        if not in_path.exists():
            print(f"\n[{in_name}] スキップ (ファイルなし)")
            continue

        df = pd.read_csv(in_path)
        print(f"\n[{in_name}] {len(df)} rows")
        for team_id, delta in ADJUST_TEAMS.items():
            df = adjust(df, team_id, delta, verbose=("select" in in_name))
        df.to_csv(out_path, index=False)
        print(f"  → 保存: {out_name}")
        processed += 1

    if processed == 0:
        print("\n処理対象ファイルがありません。先に gold_medal_mike_kim.py を実行してください。")
    else:
        print(f"\n完了! ({processed} files)")


if __name__ == "__main__":
    main()
