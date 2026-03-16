# モデル設計書 - Baseline v3 (Final)

## 目標
Brier Score (= MSE of predicted probabilities) を最小化する。
P(LowerTeamID wins) を全132,133マッチアップに対して予測する。

## アーキテクチャ

### 概要
LightGBMによる二値分類モデル (logloss目的関数)。
男女共通パイプライン + 5モデルアンサンブル + Elo勝率ブレンド。

### 学習データ
- NCAAトーナメント結果 (男子: 1985-2025, 女子: 1998-2025)
- 合計4,302試合
- TeamA = LowerTeamID として正規化 (提出形式と一致)

### バリデーション戦略
- 直近5年 (2021-2025) をテストセット、それ以前で学習
- 2020はCOVIDでトーナメント中止のため自動的に除外

## 特徴量設計 (51特徴量)

### 1. Elo Rating (importance: #2)
- K=32, 初期値1500, margin-of-victory調整
- **RS + トーナメント結果**の両方でEloを更新 (次シーズンへのキャリーオーバー)
- シーズン開始時に平均回帰 (factor=0.75)
- `diff_elo`, `elo_win_prob` (ロジスティック変換) の2つ

### 2. シード情報 (importance: #1)
- シード番号 (1-16) の差分: `diff_seed_num`
- 非線形相互作用: `seed_product`, `seed_sum`
- 2026年はseed未確定 → NaN → LightGBMがNaN処理

### 3. レギュラーシーズン統計 (25特徴量)
各チームのシーズン通算から差分を計算:
- 効率性: OffEff, DefEff, NetEff
- シューティング: FG%, 3P%, FT%, eFG%, TS%
- ボックススコア: リバウンド率, アシスト, TO, スティール, ブロック
- 安定性: score_std, margin_std
- AST/TO ratio, pace

### 4. Massey Ordinals (男子のみ, 10特徴量)
- Top 10システム (POM, SAG, MOR, WLK, DOL, COL, RTH, WOL, CNG, MB)
- 集約: mean, median, min, max, std
- 個別: POM, SAG, MOR の生ランク
- 全システム集約: mean, median

### 5. Strength of Schedule (importance: #4)
- `sos`: 対戦相手の平均Elo
- `sos_std`: 対戦相手のElo標準偏差 (スケジュールの多様性)

### 6. カンファレンス強度 (importance: #6-7)
- `conf_elo_mean`, `conf_elo_median`: 所属カンファレンスの平均/中央値Elo

### 7. 直近フォーム
- `recent_win_pct`: 直近10試合の勝率
- `recent_avg_margin`: 直近10試合の平均得失点差

### 8. その他
- `gender`: 男子=0, 女子=1

## ハイパーパラメータ (グリッドサーチで最適化)

```python
lgbm_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.02,    # 低めの学習率
    'num_leaves': 8,          # 浅い木 (過学習抑制)
    'max_depth': 5,
    'min_child_samples': 30,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 1.0,
    'reg_lambda': 10.0,       # 強めのL2正則化
}
# best_iteration: 375
```

## アンサンブル
- 5モデルを異なるseedとサブサンプリング率で学習
- 予測値の平均を最終予測とする

## ブレンド
- `final = 0.85 * lgbm_ensemble + 0.15 * elo_win_prob`
- Eloの安定した予測力をベースとして活用

## バリデーション結果

| 指標 | Single | Ensemble |
|------|--------|----------|
| Brier Score | **0.1705** | 0.1707 |
| Log Loss | 0.506 | 0.507 |
| Accuracy | 74.4% | **74.9%** |

### シーズン別 (Ensemble)
| Season | Brier | Accuracy | N |
|--------|-------|----------|---|
| 2021 | 0.1890 | 70.5% | 129 |
| 2022 | 0.1852 | 71.6% | 134 |
| 2023 | 0.1897 | 74.6% | 134 |
| 2024 | 0.1600 | 73.9% | 134 |
| 2025 | 0.1302 | 83.6% | 134 |

### Top 10 Feature Importance
1. diff_seed_num (16,289)
2. diff_elo (8,558)
3. elo_win_prob (7,101)
4. diff_sos (3,047)
5. diff_margin (2,574)
6. diff_conf_elo_mean (647)
7. diff_conf_elo_median (540)
8. diff_sos_std (382)
9. diff_recent_avg_margin (340)
10. diff_to_pg (333)

## 予測時の注意
- 2026年はシード未確定 → Elo + RS統計 + Massey で代替
- 提出は全チーム間の仮想対戦 → トーナメント非出場チームも含む
- クリッピング: [0.02, 0.98]

## 改善の方向性
- Selection Sunday後にシード情報を追加して再学習
- レギュラーシーズン試合も学習データに追加 (データ量大幅増)
- ニューラルネット (embeddings) による非線形モデル
- 女子専用モデルの分離 (ホーム/アウェイ効果が大きい)
