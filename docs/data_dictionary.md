# データ辞書 - March Machine Learning Mania 2026

## 全体概要

| 項目 | 値 |
|------|-----|
| 総ファイル数 | 35 CSV |
| 総レコード数 | 約721万行 |
| 最大ファイル | MMasseyOrdinals.csv (120.9 MB, 576万行) |
| 男子データ期間 | 1985-2026 (42シーズン) |
| 女子データ期間 | 1998-2026 (29シーズン) |
| TeamID体系 | 男子: 1101-1481 / 女子: 3101-3481 (重複なし) |
| Null値 | 全ファイルでなし |

---

## ファイル構成

データは大きく4カテゴリに分かれる。

```
data/
├── [共通] Cities.csv, Conferences.csv
├── [提出] SampleSubmissionStage1.csv, SampleSubmissionStage2.csv
├── [男子 M*] 17ファイル
└── [女子 W*] 14ファイル
```

男子(M)と女子(W)は基本的に同じ構造のペアになっている。
男子のみに存在するファイルが3つある（後述）。

---

## 1. 共通ファイル

### Cities.csv (509行)
試合開催都市のマスタ。

| カラム | 型 | 説明 |
|--------|-----|------|
| CityID | int | 都市ID (4001-4531) |
| City | str | 都市名 |
| State | str | 州名 (68種) |

### Conferences.csv (51行)
カンファレンスのマスタ。

| カラム | 型 | 説明 |
|--------|-----|------|
| ConfAbbrev | str | カンファレンス略称 (主キー) |
| Description | str | カンファレンス正式名 |

---

## 2. 提出ファイル

### SampleSubmissionStage1.csv (519,144行)
Stage1用サンプル。2022-2025年の過去データでスコアリングされる。

### SampleSubmissionStage2.csv (132,133行)
Stage2用サンプル（本番提出用）。2026年のみ。全チーム間の仮想対戦を含む。

| カラム | 型 | 説明 |
|--------|-----|------|
| ID | str | `{Season}_{LowerTeamID}_{HigherTeamID}` 形式 |
| Pred | float | LowerTeamID側が勝つ確率 (初期値0.5) |

---

## 3. マスタデータ (M/W共通構造)

### Teams.csv (M: 381チーム / W: 379チーム)
チームマスタ。

| カラム | 型 | M | W | 説明 |
|--------|-----|---|---|------|
| TeamID | int | o | o | チームID |
| TeamName | str | o | o | チーム名 |
| FirstD1Season | int | o | **x** | D1参加開始シーズン |
| LastD1Season | int | o | **x** | D1参加最終シーズン |

> **注意**: WTeamsにはFirstD1Season/LastD1Season列がない（唯一の構造差異）。

### Seasons.csv (M: 42行 / W: 29行)
シーズンマスタ。

| カラム | 型 | 説明 |
|--------|-----|------|
| Season | int | シーズン年 |
| DayZero | date | シーズン基準日 (DayNum=0の日付) |
| RegionW | str | ブラケット地域名 W |
| RegionX | str | ブラケット地域名 X |
| RegionY | str | ブラケット地域名 Y |
| RegionZ | str | ブラケット地域名 Z |

> DayNumは全試合データで使われる。`実際の日付 = DayZero + DayNum日`。

### TeamConferences.csv (M: 13,753行 / W: 10,073行)
各シーズンのチーム所属カンファレンス。

| カラム | 型 | 説明 |
|--------|-----|------|
| Season | int | シーズン年 |
| TeamID | int | チームID |
| ConfAbbrev | str | カンファレンス略称 (→ Conferences.csv) |

### TeamSpellings.csv (M: 1,178行 / W: 1,144行)
チーム名の表記ゆれマッピング。外部データとの結合用。

| カラム | 型 | 説明 |
|--------|-----|------|
| TeamNameSpelling | str | 表記パターン (小文字) |
| TeamID | int | 正規チームID |

---

## 4. レギュラーシーズン試合結果

### RegularSeasonCompactResults.csv (M: 196,823行 / W: 140,825行)
試合結果の基本情報。

| カラム | 型 | 説明 |
|--------|-----|------|
| Season | int | シーズン年 |
| DayNum | int | 日番号 (0-132) |
| WTeamID | int | 勝利チームID |
| WScore | int | 勝利チームスコア |
| LTeamID | int | 敗北チームID |
| LScore | int | 敗北チームスコア |
| WLoc | str | 勝利チームの場所 (**H**ome / **A**way / **N**eutral) |
| NumOT | int | 延長戦回数 |

### RegularSeasonDetailedResults.csv (M: 122,775行 2003~ / W: 85,505行 2010~)
CompactResultsの全カラム + 以下のボックススコア統計 (W=勝者, L=敗者)。

| カラム接頭辞 | W側 | L側 | 説明 |
|-------------|------|------|------|
| FGM | WFGM | LFGM | フィールドゴール成功数 |
| FGA | WFGA | LFGA | フィールドゴール試投数 |
| FGM3 | WFGM3 | LFGM3 | 3ポイント成功数 |
| FGA3 | WFGA3 | LFGA3 | 3ポイント試投数 |
| FTM | WFTM | LFTM | フリースロー成功数 |
| FTA | WFTA | LFTA | フリースロー試投数 |
| OR | WOR | LOR | オフェンスリバウンド |
| DR | WDR | LDR | ディフェンスリバウンド |
| Ast | WAst | LAst | アシスト |
| TO | WTO | LTO | ターンオーバー |
| Stl | WStl | LStl | スティール |
| Blk | WBlk | LBlk | ブロック |
| PF | WPF | LPF | パーソナルファウル |

> 合計34カラム (基本8 + ボックススコア26)。

---

## 5. NCAAトーナメント

### NCAATourneyCompactResults.csv (M: 2,585行 / W: 1,717行)
トーナメント試合結果。構造はRegularSeasonCompactResultsと同一。
- DayNum: 134-154 (レギュラーシーズン後)
- 男子はWLoc=N(中立地)のみ。**女子はH/A/Nの3値**（上位シードがホーム開催の場合あり）。

### NCAATourneyDetailedResults.csv (M: 1,449行 / W: 961行)
トーナメント試合の詳細統計。構造はRegularSeasonDetailedResultsと同一。

### NCAATourneySeeds.csv (M: 2,626行 / W: 1,744行)
各シーズンのトーナメントシード。

| カラム | 型 | 説明 |
|--------|-----|------|
| Season | int | シーズン年 |
| Seed | str | シード文字列 (例: "W01", "X16a") |
| TeamID | int | チームID |

> Seedフォーマット: `{Region}{SeedNum}{PlayIn}`
> - Region: W/X/Y/Z (Seasons.csvの地域名に対応)
> - SeedNum: 01-16
> - PlayIn: a/b (プレイインゲーム用、なければ空)

### NCAATourneySlots.csv (M: 2,586行 / W: 1,780行)
トーナメントブラケット構造の定義。

| カラム | 型 | 説明 |
|--------|-----|------|
| Season | int | シーズン年 |
| Slot | str | スロット名 (例: "R1W1", "R2W1") |
| StrongSeed | str | 上位シードのスロット/シード名 |
| WeakSeed | str | 下位シードのスロット/シード名 |

---

## 6. カンファレンストーナメント

### ConferenceTourneyGames.csv (M: 6,793行 2001~ / W: 4,160行 2001~)

| カラム | 型 | 説明 |
|--------|-----|------|
| Season | int | シーズン年 |
| ConfAbbrev | str | カンファレンス略称 |
| DayNum | int | 日番号 |
| WTeamID | int | 勝利チームID |
| LTeamID | int | 敗北チームID |

> スコアは含まれない（勝敗のみ）。

---

## 7. その他のトーナメント (NIT等)

### SecondaryTourneyCompactResults.csv (M: 1,865行 / W: 906行)
NIT等のセカンダリトーナメント結果。構造はCompactResultsと同一 + SecondaryTourney列。

| 追加カラム | 型 | 説明 |
|-----------|-----|------|
| SecondaryTourney | str | トーナメント種別 |

男子の種別: NIT, CBI, CIT, V16, WBI, NInv, NCAAT2nd (7種)
女子の種別: WNIT, WBI, WCBI (3種)

### SecondaryTourneyTeams.csv (M: 1,895行 / W: 904行)
セカンダリトーナメント出場チームリスト。

| カラム | 型 | 説明 |
|--------|-----|------|
| Season | int | シーズン年 |
| SecondaryTourney | str | トーナメント種別 |
| TeamID | int | チームID |

---

## 8. 開催地情報

### GameCities.csv (M: 90,684行 2010~ / W: 54,790行 2010~)

| カラム | 型 | 説明 |
|--------|-----|------|
| Season | int | シーズン年 |
| DayNum | int | 日番号 |
| WTeamID | int | 勝利チームID |
| LTeamID | int | 敗北チームID |
| CRType | str | 試合種別 (Regular / NCAA / Secondary) |
| CityID | int | 開催都市ID (→ Cities.csv) |

---

## 9. 男子のみのデータ (3ファイル)

### MMasseyOrdinals.csv (5,761,702行, 120.9 MB) -- 最大ファイル
外部ランキングシステムによるチーム順位。2003年以降。

| カラム | 型 | 説明 |
|--------|-----|------|
| Season | int | シーズン年 |
| RankingDayNum | int | ランキング算出日 |
| SystemName | str | ランキングシステム名 (196種) |
| TeamID | int | チームID (372チーム) |
| OrdinalRank | int | 順位 (1-365) |

> 196種のランキングシステム x 複数日付 x 全チーム で膨大な行数になっている。
> 代表的なシステム: AP, RPI, SAG, POM, MOR 等。

### MNCAATourneySeedRoundSlots.csv (776行)
シードからラウンド・対戦スロットへの対応表。

| カラム | 型 | 説明 |
|--------|-----|------|
| Seed | str | シード文字列 |
| GameRound | int | ラウンド番号 |
| GameSlot | str | ゲームスロット名 |
| EarlyDayNum | int | 試合可能最早日 |
| LateDayNum | int | 試合可能最遅日 |

### MTeamCoaches.csv (13,898行, 1,657コーチ)
チームのコーチ履歴。

| カラム | 型 | 説明 |
|--------|-----|------|
| Season | int | シーズン年 |
| TeamID | int | チームID |
| FirstDayNum | int | 指揮開始日 |
| LastDayNum | int | 指揮終了日 |
| CoachName | str | コーチ名 |

---

## 10. M/W対応表

| ファイル名 (接尾辞) | M | W | 構造一致 |
|---------------------|---|---|---------|
| Teams | o | o | **差異あり** (M にのみ FirstD1Season, LastD1Season) |
| Seasons | o | o | 一致 |
| TeamConferences | o | o | 一致 |
| TeamSpellings | o | o | 一致 |
| RegularSeasonCompactResults | o | o | 一致 |
| RegularSeasonDetailedResults | o | o | 一致 |
| NCAATourneyCompactResults | o | o | 一致 |
| NCAATourneyDetailedResults | o | o | 一致 |
| NCAATourneySeeds | o | o | 一致 |
| NCAATourneySlots | o | o | 一致 |
| ConferenceTourneyGames | o | o | 一致 |
| GameCities | o | o | 一致 |
| SecondaryTourneyCompactResults | o | o | 一致 |
| SecondaryTourneyTeams | o | o | 一致 |
| MasseyOrdinals | o | **x** | 男子のみ |
| NCAATourneySeedRoundSlots | o | **x** | 男子のみ |
| TeamCoaches | o | **x** | 男子のみ |

---

## 主要なリレーション (ER概要)

```
Teams ──< TeamConferences >── Conferences
  │
  ├──< RegularSeasonResults (WTeamID / LTeamID)
  │         └──> GameCities ──> Cities
  │
  ├──< NCAATourneyResults (WTeamID / LTeamID)
  │
  ├──< NCAATourneySeeds
  │         └──> NCAATourneySlots (Seed ↔ StrongSeed/WeakSeed)
  │
  ├──< ConferenceTourneyGames
  │
  ├──< SecondaryTourneyResults
  │         └──> SecondaryTourneyTeams
  │
  ├──< MasseyOrdinals (男子のみ)
  │
  └──< TeamCoaches (男子のみ)

Seasons ── DayZero で全 DayNum を実日付に変換
```

---

## DayNum体系

全ての試合日付は `DayNum` (シーズン内の日番号) で表現される。

| 範囲 | 期間 |
|------|------|
| 0-132 | レギュラーシーズン |
| 134-154 | NCAAトーナメント |

実日付 = `Seasons.DayZero + DayNum日`

---

## EDAのポイント

1. **DetailedResults は CompactResults の部分集合**: Detailed は2003年(男子)/2010年(女子)以降のみ
2. **MasseyOrdinals は男子のみ**: 女子の外部ランキングデータは含まれない
3. **女子トーナメントにはホーム開催あり**: WLoc に H/A が含まれる (男子は N のみ)
4. **M/WのTeamIDは重複しない**: 提出ファイルで男女統合されているが、IDで自然に分離可能
5. **2026年データ**: RegularSeasonのみ存在 (トーナメントはまだ未開催)
