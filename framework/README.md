# 検証基盤フレームワーク

March Machine Learning Mania の実験を統一的に実行・比較するための基盤。

## ファイル一覧

```
framework/
├── __init__.py
├── config.py            # ExperimentConfig (実験設定 dataclass)
├── data_loader.py       # データ読み込み、対戦ペア生成
├── cv.py                # Season-based CV splitter
├── evaluation.py        # Brier score, LogLoss, Accuracy + レポート生成
├── runner.py            # 実験オーケストレーション
├── features/
│   ├── base.py          # FeatureGenerator 抽象基底クラス
│   ├── seed_diff.py     # シード差特徴量
│   ├── elo.py           # Eloレーティング + SOS + カンファレンス強度
│   ├── season_stats.py  # シーズン統計 (Detailed/Compact)
│   ├── massey.py        # Massey Ordinals (男子のみ)
│   ├── recent_form.py   # 直近フォーム
│   ├── team_rating.py   # チーム Rating (Ridge / 反復調整法)
│   ├── matchup_rating.py # Matchup Rating 特徴量ビルダー
│   └── coach.py         # コーチ特徴量 (男子のみ, minimal/tournament/full)
└── models/
    ├── base.py           # ModelWrapper 抽象基底クラス
    ├── logistic.py       # ロジスティック回帰
    ├── elo_predictor.py  # Elo勝率公式 (学習不要)
    ├── lightgbm_model.py # LightGBM
    ├── ridge_model.py    # ScaledLogistic + Ridge (NaN imputation付き)
    └── calibrated.py     # 汎用キャリブレーションラッパー
```

## 実行方法

```bash
# Framework ベースの実験
uv run python experiments/baseline_logistic.py
uv run python experiments/elo_baseline.py
uv run python experiments/elo_optimized.py
uv run python experiments/baseline_lgbm.py
uv run python experiments/team_rating_experiment.py
uv run python experiments/massey_baseline.py
uv run python experiments/calibration_experiment.py
uv run python experiments/coach_experiment_v2.py
uv run python experiments/ensemble_experiment.py
uv run python experiments/final_ensemble_submission.py
```

結果は `outputs/<実験名>/` に保存される:
- `results.json` - 全体の評価結果
- `oof_predictions.csv` - OOF (Out-of-Fold) 予測
- `submission.csv` - Kaggle 提出用ファイル

## CV 設計

**Season-based CV (時系列分割)**:
- 各バリデーションシーズンに対し、それより前のシーズンのみで学習
- 例: val=2023 → train=2003-2022
- デフォルト val_seasons: [2021, 2022, 2023, 2024, 2025]

**gender_mode**:
- `"combined"`: 男女統合モデル
- `"separate"`: 男女別モデル
- `"both"`: 両方を実行して比較

## リーク防止策

1. CV の各 fold で、val_season 以降のトーナメント結果は学習データに含まない
2. 特徴量生成時に `max_season` パラメータで使用可能なデータ範囲を制限
3. Elo特徴量: NCAAトーナメント開始直前のスナップショットを使用。当該シーズンのトーナメント結果はスナップショットに含まれない
4. シード情報: トーナメント前に公開される情報のため使用可
5. `FeatureGenerator.generate()` の `max_season` 引数で各実装がリーク防止を保証

## 特徴量一覧

| 特徴量クラス | カラム | 説明 |
|-------------|--------|------|
| SeedDiffFeature | seed_num | NCAAトーナメントシード番号 |
| EloFeature | elo, [sos, sos_std, conf_elo_mean, conf_elo_median] | Eloレーティング + オプション |
| SeasonStatsFeature | win_pct, ppg, margin, fg_pct 等25列 | シーズン統計 |
| MasseyFeature | massey_top_mean 等10列 | Massey Ordinals (男子のみ) |
| RecentFormFeature | recent_win_pct, recent_avg_margin | 直近N試合 |
| TeamRatingFeature | ridge_off/def/net/pace/consistency, iter_off/def/net/pace/consistency | チーム Rating |
| matchup_rating | off_A_minus_def_B, off_B_minus_def_A, efficiency_diff 等7列 | Matchup Rating |
| CoachFeature | coach_tenure, coach_is_new 等 | コーチ特徴量 (男子のみ) |

## 新しい特徴量の追加方法

`framework/features/base.py` の `FeatureGenerator` を継承:

```python
class MyFeature(FeatureGenerator):
    @property
    def name(self) -> str:
        return "my_feature"

    @property
    def feature_columns(self) -> list[str]:
        return ["feat_a", "feat_b"]

    def generate(self, data, gender, max_season):
        # return: DataFrame with ["Season", "TeamID", "feat_a", "feat_b"]
        ...
```

## 新しいモデルの追加方法

`framework/models/base.py` の `ModelWrapper` を継承:

```python
class MyModel(ModelWrapper):
    @property
    def name(self) -> str:
        return "my_model"

    @property
    def handles_nan(self) -> bool:
        return True  # NaN を扱えるモデルの場合

    def fit(self, X, y):
        ...

    def predict_proba(self, X):
        # return: np.ndarray (shape: n_samples,)
        ...
```

## キャリブレーション

`CalibratedModelWrapper` で任意の `ModelWrapper` にキャリブレーションを追加できる:

```python
from framework.models.calibrated import CalibratedModelWrapper
from framework.models.ridge_model import ScaledLogisticModel

model_factory = lambda: CalibratedModelWrapper(
    base_factory=lambda: ScaledLogisticModel(C=0.05),
    method="isotonic",  # "isotonic" or "sigmoid"
    cv=5,
)
```

## 2026年の実験結果 (Season-based CV, val=2021-2025)

| モデル | Combined | Men | Women |
|--------|----------|-----|-------|
| **3モデルアンサンブル (L2+L1+Elo, 男女別重み)** | **0.1640** | **0.1910** | **0.1368** |
| 2モデルアンサンブル (L2+Elo, 男女別重み) | 0.1644 | 0.1910 | 0.1377 |
| 全特徴量+Coach(minimal)+Logistic C=0.05+Isotonic | 0.1657 | 0.1910 | 0.1401 |
| 全特徴量+Logistic C=0.05+Isotonic | 0.1663 | 0.1925 | 0.1398 |
| Elo optimized (separate) | 0.1684 | 0.1946 | 0.1393 |
| LightGBM (全特徴量) | 0.1874 | 0.2094 | 0.1751 |
