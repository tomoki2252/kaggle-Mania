# 過去のMarch Machine Learning Mania 解法調査

過去のKaggle March Machine Learning Maniaコンペ（および類似のNCAA予測プロジェクト）で使われた手法を調査し、モデル設計の参考としてまとめる。

---

## 1. 手法の全体像

過去の上位解法は、大きく以下の3パターンに分類できる。

### パターンA: 外部ランキング + シンプルモデル（最も多い成功パターン）

複数の外部ランキングシステム（KenPom, Sagarin, MasseyOrdinals等）のスコア差をロジスティック回帰に入力する手法。

- **代表例**: 1位解法（Andrew Landgraf）、Top 1% Gold 2023（maze508）
- **特徴量**: 外部ランキングのスコア差、シード差
- **モデル**: ロジスティック回帰
- **考え方**: 自分でチーム評価指標を構築するよりも、既に高精度な外部ランキングシステムを活用した方が精度が出る

### パターンB: レギュラーシーズン統計 + 勾配ブースティング（最も特徴量が多い）

レギュラーシーズンの各種統計を差分特徴量として大量に作り、XGBoost/LightGBMで学習する手法。

- **代表例**: 2021年解法（John Edwards）、複数のKaggle Notebookソリューション
- **特徴量**: Elo, SRS, 勝率, 得失点差, Four Factors, MasseyOrdinals, シード等 約50-100個
- **モデル**: XGBoost / LightGBM
- **考え方**: 多数の特徴量を用意し、モデルに重要度の選択を任せる

### パターンC: Eloレーティング + キャリブレーション（最もシンプル）

Eloレーティングを独自に計算し、ロジスティック関数で確率に変換する手法。

- **代表例**: The Data Jocks 2025モデル、Conor Dewey
- **特徴量**: Eloスコア差のみ（+ 外部ランキング数個をブレンドする場合あり）
- **モデル**: Eloのロジスティック変換（実質パラメトリックモデル）
- **考え方**: 全試合を時系列で処理し、モメンタム（直近の調子）を自然に反映

---

## 2. 各解法の詳細

### 2a. 1位解法 — Andrew Landgraf

| 項目 | 内容 |
|------|------|
| 順位 | **1位** |
| モデル | ロジスティック回帰 |
| 核心的アイデア | 自分の予測精度だけでなく、**他の参加者の提出傾向もモデル化**し、賞金最大化の観点で最適な提出を選択 |
| 特徴量 | 外部ランキングシステムのスコア差 |

> この解法のユニークな点は、Brier scoreの絶対値ではなく**順位を最大化する戦略**を取ったこと。他の参加者がコンセンサス的な予測を出す中で、あえて差別化された予測を出すことで上位に入れるという洞察。

### 2b. Top 1% Gold 2023 — maze508

| 項目 | 内容 |
|------|------|
| 順位 | **Top 1% Gold** |
| モデル | XGBoost + ロジスティック回帰 + ランダムフォレスト のアンサンブル |
| 特徴量 | 外部ランキング（POM, SAG, MOR等のTop10システム）、勝率、得失点差、ホーム/アウェイ成績、ボックススコア集計 |
| 特徴量選択 | Recursive Feature Elimination（RFE）で精度を落とす特徴量を除外 |

**試行錯誤の過程**:
- 独自のEloシステムを構築したが、外部ランキング（MasseyOrdinals）に勝てなかった
- トーナメントのラウンドを考慮したElo更新も試みたが、改善しなかった
- 最終的に「歴史的に最も精度が高い外部ランキングTop10の活用」に落ち着いた

> **教訓**: 独自Elo構築に時間をかけるより、MasseyOrdinalsに含まれる既存の196システムを活用する方が効率的。

### 2c. 4位解法 — Erik Forseth

| 項目 | 内容 |
|------|------|
| 順位 | **4位** |
| モデル | ロジスティック回帰 + ニューラルネットワーク のアンサンブル |

> ニューラルネットワークを使った解法は少数だが、ロジスティック回帰との組み合わせで上位に入っている。

### 2d. 2021年解法 — John Edwards

| 項目 | 内容 |
|------|------|
| モデル | XGBoost（木の深さ2, 学習率0.012, 木数401） |
| バリデーション | 2015-2019をテスト、5-fold CV（層化） |
| スコア | Log Loss 0.5008（2019トーナメント） |

**特徴量の構成**:
| カテゴリ | 具体的な特徴量 |
|---------|---------------|
| チーム強度指標 | Elo, LRMC, SRS, RPI, TrueSkill, Colley行列法, LME4（混合効果モデル） |
| ボックススコア | FG%, 3P%, FT%, リバウンド, アシスト, TO, スティール, ブロック（ポゼッションあたりに正規化） |
| MasseyOrdinals | 65システムのDayNum=133時点のランキングの平均 |
| シード | シード番号差 |

**Eloの実装詳細**:
- ホームコートアドバンテージ ≈ 135ポイント相当
- Margin of Victory（得点差）を更新量に反映
- シーズン間のキャリーオーバーあり

**独自の工夫**:
- 予測確率を[0.05, 0.95]にクリッピング
- 45%-55%の拮抗ゲームで確信度を人為的に調整
- **怪我情報の手動反映**: スター選手欠場時に約3ポイント分の補正を加えた

### 2e. The Data Jocks 2025モデル

| 項目 | 内容 |
|------|------|
| モデル | Eloレーティング（ロジスティック変換） |
| 核心的アイデア | **シーズン開始時のElo初期値の工夫** |

**Eloの工夫点**:
- チームごとに異なるprior（事前情報）を設定し、シーズン開始から精度が高い
- K値を小さくして安定させる（大きなpriorがあるため急激な変動が不要）
- Margin of Victory を非線形に反映（20点勝ちは2点勝ちよりも大きく反映）

### 2f. Machine Learning Madness — Conor Dewey

| 項目 | 内容 |
|------|------|
| モデル | ロジスティック回帰 |
| 特徴量 | Sagarin + Pomeroy + Moore + Whitlock のランキング平均 + Elo |

**アプローチ**:
- 各ランキングシステムを個別にテストし、最も精度が高いものを選定
- スコアを0-1に正規化してから合成
- 最終的にスコア差分をロジスティック回帰に入力

---

## 3. 共通する知見のまとめ

### 特徴量について

| 特徴量 | 重要度 | 備考 |
|--------|--------|------|
| **シード差** | 最重要 | ほぼ全ての上位解法で使用。単体でBrier≈0.17 |
| **外部ランキング差**（MasseyOrdinals） | 最重要 | Top1%解法の多くが核心的特徴量として使用。POM, SAG, MORが特に有用 |
| **Eloレーティング差** | 重要 | シードがないチームにも適用可能な点が強み |
| 勝率・得失点差 | 重要 | 基本的だが安定して効く |
| ボックススコア統計差 | 中程度 | Four Factors等。ポゼッションあたりに正規化が必須 |
| SOS（対戦相手の強さ） | 中程度 | Eloベースで算出する例が多い |
| カンファレンス強度 | 低〜中 | 補助的 |
| 直近フォーム | 低 | 過学習しやすい |

### モデルについて

| モデル | 使用頻度 | 長所 | 短所 |
|--------|---------|------|------|
| **ロジスティック回帰** | 最多 | キャリブレーションが良い、過学習しにくい | 非線形関係を捉えにくい |
| **XGBoost / LightGBM** | 多い | 非線形・交互作用を自動で捉える | 少ないデータ(4,302試合)で過学習しやすい |
| ニューラルネットワーク | 少ない | Embeddingで複雑な関係を学習 | データ量不足、チューニングが難しい |
| Eloのみ | 少ない | シンプルで解釈しやすい | 特徴量追加の余地がない |

> **傾向**: 特徴量が少ない（ランキング差のみ等）場合はロジスティック回帰、多数の特徴量を使う場合はXGBoost/LightGBMが選ばれている。

### 評価指標の変遷

| 年 | 評価指標 |
|----|---------|
| 〜2022 | Log Loss |
| 2023〜 | **Brier Score** |

> Brier scoreへの変更により、極端な確率（0.99等）を出すインセンティブが減った。Log Lossは「完全に正しい予測」への報酬が非常に大きかったが、Brier scoreでは「そこそこ正しい予測」の方が安全。

---

## 4. 我々のモデル設計への示唆

### 今回のコンペ（2026）で有効と考えられるアプローチ

1. **MasseyOrdinalsの活用を最優先にする（男子）**
   - 過去の上位解法の多くが「独自Eloより外部ランキングの方が強い」と結論
   - POM, SAG, MOR を中心に、DayNum=128-133付近のランキングを使用

2. **モデルはロジスティック回帰をベースに、LightGBMと比較**
   - データ量(4,302試合)を考えるとシンプルなモデルが有利
   - 特徴量が多くなる場合はLightGBM、少ない場合はロジスティック回帰

3. **Eloは補助特徴量として使用**
   - MasseyOrdinalsがない女子モデルでは、Eloがメイン特徴量になる
   - シーズン開始時のprior設定が重要（前シーズンのEloからの回帰）

4. **予測確率のクリッピング**
   - [0.02, 0.98] 程度にクリッピングして極端な予測を避ける
   - Brier scoreではLog Lossほど極端な予測のペナルティは大きくないが、安全策として有効

---

## 出典

- [Top 1% Gold - March Machine Learning Mania 2023 Solution Writeup (maze508)](https://medium.com/@maze508/top-1-gold-kaggle-march-machine-learning-mania-2023-solution-writeup-2c0273a62a78)
- [March Machine Learning Mania, 1st Place Winner's Interview: Andrew Landgraf](https://medium.com/kaggle-blog/march-machine-learning-mania-1st-place-winners-interview-andrew-landgraf-f18214efc659)
- [March Machine Learning Mania, 4th Place Winner's Interview: Erik Forseth](https://medium.com/kaggle-blog/march-machine-learning-mania-4th-place-winners-interview-erik-forseth-8d915d8cea57)
- [2021 March Madness Kaggle Solution (John Edwards)](https://johnbedwards.io/blog/march_madness_2021/)
- [March Madness 2025 Model (The Data Jocks)](https://thedatajocks.com/march-madness-2025-model/)
- [Machine Learning Madness (Conor Dewey)](https://www.conordewey.com/blog/machine-learning-madness-predicting-every-ncaa-tournament-matchup)
- [March-ML-Mania 2026 (GitHub)](https://github.com/ngusadeep/March-ML-Mania)
- [Kaggle March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)
