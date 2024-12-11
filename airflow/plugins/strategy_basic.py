import abc
import decimal
import random
from typing import Any

import pandas as pd

from gym_env_single_asset import Actions


class Strategy(abc.ABC):

    def get_direction(self, idx: int, features_df: pd.DataFrame) -> int:
        return 0

    def get_action(
            self,
            observation: Any | None = None,
            info: dict[str, Any] | None = None) -> Actions:
        return Actions.Hold


class StrategyBuyAndHold(Strategy):

    def get_direction(self, idx: int, features_df: pd.DataFrame) -> int:
        return 1

    def get_action(
            self,
            observation: Any | None = None,
            info: dict[str, Any] | None = None) -> Actions:
        return Actions.Buy


class StrategyRandomWalk(Strategy):
    def __init__(self, seed: int | None = None):
        super().__init__()
        if seed is None:
            self._rng = random.Random()
        else:
            self._rng = random.Random(seed)

    def get_direction(self, idx: int, features_df: pd.DataFrame) -> int:
        direction = self._rng.randint(-1, 1)
        return direction

    def get_action(
            self,
            observation: Any | None = None,
            info: dict[str, Any] | None = None) -> Actions:
        direction = self._rng.randint(-1, 1)
        if direction == 0:
            return Actions.Hold
        if direction == 1:
            return Actions.Buy
        if direction == -1:
            return Actions.Sell


class StrategyTrendFollowing(Strategy):

    def __init__(
            self,
            price_delta_threshold: decimal = 0.01):
        super().__init__()
        self._price_delta_threshold = price_delta_threshold

    def get_direction(self, idx: int, features_df: pd.DataFrame) -> int:
        row = features_df.iloc[idx]
        return self._get_direction_by_price(row["open"], row["close"])

    def get_action(
            self,
            observation: Any | None = None,
            info: dict[str, Any] | None = None) -> Actions:
        if info is None:
            return Actions.Hold
        direction = self._get_direction_by_price(info["open_price"], info["close_price"])
        if direction == 1:
            return Actions.Buy
        if direction == -1:
            return Actions.Sell
        return Actions.Hold

    def _get_direction_by_price(self, open_price: decimal, close_price: decimal) -> int:
        price_delta = (close_price - open_price)
        if abs(price_delta) < self._price_delta_threshold:
            return 0
        if price_delta > 0:
            return 1
        else:
            return -1
