# March Machine Learning Mania 2026

Kaggle コンペティション: NCAA バスケットボールトーナメント (男女) の全対戦結果を予測する。
評価指標は Brier score。

## 最終モデル

**Gold Medal Solution (Mike Kim 再現) + ポスト処理**

- ソース: [experiments/gold_medal_mike_kim.py](experiments/gold_medal_mike_kim.py)
- 参考ノートブック: [reference/gold-medal-solution-mike-kim.ipynb](reference/gold-medal-solution-mike-kim.ipynb)
- **LOSO CV Brier Score: 0.1672** (22シーズン平均, 2003-2025)
- 提出ファイル: [submission/submission.csv](submission/submission.csv) ※ミシガン大 (TeamID=1276) の勝率を delta=+0.3 でロジット空間調整済み

### モデル構成

1. **Easy 特徴量**: シード番号・シード差
2. **Medium 特徴量**: シーズン平均 box score、直近14日勝率、アウェイ勝率、対戦成績、勝率ベース強さ指標
3. **Hard 特徴量**: Elo レーティング
4. **Hardest 特徴量**: GLM team quality (statsmodels)
5. **メタ特徴量**: NuSVC + KNN (men/women クロス学習)
6. **最終モデル**: XGBoost (Cauchy 損失) + leaf-based LogisticRegression calibration
7. **推論**: 全 LOSO モデルのアンサンブル (22モデル平均)

### ポスト処理

[experiments/postprocess_submission.py](experiments/postprocess_submission.py) でチーム別の勝率調整を実施。
ロジット空間でのシフトにより確率が [0,1] を超えることなく自然に調整される。

## ディレクトリ構成

```
.
├── data/                    # 公式コンペデータ (gitignore)
├── docs/                    # プロジェクトドキュメント
├── reference/               # 参考ノートブック
├── framework/               # 検証基盤 (特徴量・モデル・CV)
├── experiments/             # 実験スクリプト
│   ├── gold_medal_mike_kim.py       # ★ 最終モデル
│   ├── postprocess_submission.py    # ★ ポスト処理
│   ├── final_ensemble_submission.py # Framework アンサンブル
│   ├── baseline_logistic.py         # シード差 + Logistic
│   ├── baseline_lgbm.py            # 全特徴量 + LightGBM
│   ├── elo_baseline.py             # Elo (固定パラメータ)
│   ├── elo_optimized.py            # Elo (最適化パラメータ)
│   ├── massey_baseline.py          # Massey + Logistic/Ridge
│   ├── calibration_experiment.py   # キャリブレーション検証
│   ├── coach_experiment_v2.py      # コーチ特徴量実験
│   ├── ensemble_experiment.py      # アンサンブルブレンド実験
│   └── team_rating_experiment.py   # Team Rating 実験
├── submission/              # 最終提出物
│   ├── submission.csv       # Kaggle 提出ファイル (調整済み)
│   ├── oof_predictions.csv  # OOF 予測
│   └── metrics.txt          # CV 結果詳細
└── outputs/                 # 中間実験結果 (gitignore, 再生成可能)
```

## セットアップ

```bash
uv sync
```

## 実行方法

```bash
# 最終モデル (Gold Medal 再現)
uv run python experiments/gold_medal_mike_kim.py

# ポスト処理 (チーム別勝率調整)
uv run python experiments/postprocess_submission.py

# Framework ベースの実験
uv run python experiments/baseline_logistic.py
uv run python experiments/elo_optimized.py
uv run python experiments/final_ensemble_submission.py
```

## スコア一覧

### 最終モデル (Gold Medal 再現, LOSO全シーズン)

| Season | Brier Score |
|--------|-------------|
| 2003 | 0.1893 |
| 2004 | 0.1760 |
| 2005 | 0.1640 |
| 2006 | 0.1943 |
| 2007 | 0.1454 |
| 2008 | 0.1477 |
| 2009 | 0.1667 |
| 2010 | 0.1687 |
| 2011 | 0.1867 |
| 2012 | 0.1569 |
| 2013 | 0.1665 |
| 2014 | 0.1748 |
| 2015 | 0.1456 |
| 2016 | 0.1736 |
| 2017 | 0.1590 |
| 2018 | 0.1815 |
| 2019 | 0.1420 |
| 2021 | 0.1814 |
| 2022 | 0.1889 |
| 2023 | 0.1892 |
| 2024 | 0.1603 |
| 2025 | 0.1205 |
| **平均** | **0.1672** |

### Framework ベース (Season-based CV, val=2021-2025)

| モデル | Combined | Men | Women |
|--------|----------|-----|-------|
| 3モデルアンサンブル (L2+L1+Elo) | **0.1640** | 0.1910 | 0.1368 |
| 全特徴量+Logistic C=0.05+Isotonic | 0.1663 | 0.1925 | 0.1398 |
| Elo optimized (separate) | 0.1684 | 0.1946 | 0.1393 |
| Seed + Logistic | 0.1781 | 0.2030 | 0.1466 |
| LightGBM (全特徴量) | 0.1874 | 0.2094 | 0.1751 |

## リーク防止策

- Season-based LOSO CV: 各シーズンの検証ではそのシーズン以降のデータを使用しない
- トーナメント結果はトレーニング時のみ使用し、特徴量生成にはレギュラーシーズンのデータのみを使用
- 2026年の特徴量は2025年以前のデータから生成

## 特徴量一覧 (Gold Medal モデル, 34特徴量)

- シード番号 (T1/T2), シード差
- シーズン平均 box score (得点, アシスト, リバウンド, ブロック, スティール, ターンオーバー, FG%, 3P%, FT%)
- 直近14日勝率
- アウェイ勝率, アウェイ対戦勝率
- 対戦成績 (過去の直接対決)
- 勝率ベース強さ指標
- Elo レーティング
- GLM team quality
- NuSVC メタ特徴量, KNN メタ特徴量
