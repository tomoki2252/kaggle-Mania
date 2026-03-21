"""特徴量生成の抽象基底クラス"""

from abc import ABC, abstractmethod

import pandas as pd


class FeatureGenerator(ABC):
    """特徴量生成の基底クラス

    各特徴量クラスは以下を実装する:
    - name: 特徴量名
    - generate(): (Season, TeamID) をキーとするチーム特徴量テーブルを返す

    リーク防止の責任は各クラスが負う:
    - generate() に渡される max_season 以降のトーナメント情報は使わないこと
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """特徴量セットの名前"""

    @property
    @abstractmethod
    def feature_columns(self) -> list[str]:
        """生成される特徴量カラム名のリスト"""

    @abstractmethod
    def generate(
        self,
        data: dict[str, pd.DataFrame],
        gender: str,
        max_season: int,
    ) -> pd.DataFrame:
        """チームレベルの特徴量テーブルを生成する

        Args:
            data: データローダーが返す辞書
            gender: "M" or "W"
            max_season: この season 以前のデータのみ使用可能
                        (この season のレギュラーシーズンデータは使用可)

        Returns:
            DataFrame with columns: ["Season", "TeamID"] + feature_columns
        """
