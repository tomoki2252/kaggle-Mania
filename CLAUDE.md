ユーザはkaggleのコンペティションに参加しています。
あなたには、ユーザの分析をサポートする役割があります。
会話は日本語で行い、ドキュメントをする際も日本語を使用してください。

以下は、kaggleのコンペティション概要の原文です。
---

# Overview
You will be forecasting the outcomes of both the men's and women's 2026 collegiate basketball tournaments, by submitting predictions for every possible tournament matchup.

# Evaluation
Submissions are evaluated on the Brier score between the predicted probabilities and the actual game outcomes (this is equivalent to mean squared error in this context).

Submission File
As a reminder, the submission file format also has a revised format from prior iterations:

1. We have combined the Men's and Women's tournaments into one single competition. Your submission file should contain predictions for both.
2. You will be predicting the hypothetical results for every possible team matchup, not just teams that are selected for the NCAA tournament. This change was enacted to provide a longer time window to submit predictions for the 2026 tournament. Previously, the short time between Selection Sunday and the tournament tipoffs would require participants to quickly turn around updated predictions. By forecasting every possible outcome between every team, you can now submit a valid prediction at any point leading up to the tournaments.
3. You may submit as many times as you wish before the tournaments start, but make sure to select the one submission you want to count towards scoring. Do not rely on automatic selection to pick your submissions.

As with prior years, each game has a unique ID created by concatenating the season in which the game was played and the two team's respective TeamIds. For example, "2026_1101_1102" indicates a hypothetical matchup between team 1101 and 1102 in the year 2026. You must predict the probability that the team with the lower TeamId beats the team with the higher TeamId. Note that the men's teams and women's TeamIds do not overlap.

The resulting submission format looks like the following, where Pred represents the predicted probability that the first team will win:

```
ID,Pred
2026_1101_1102,0.5
2026_1101_1103,0.5
2026_1101_1104,0.5
...
```

Your 2026 submissions will score 0.0 if you have submitted predictions in the right format. The leaderboard of this competition will be only meaningful once the 2026 tournaments begin and Kaggle rescores your predictions!

---

データは、data/にあります。
ドキュメントは、docs/内に配置してください。
分析用のスクリプトは、/home/t-cho/dev/kaggle/Maniaに配置してください。
