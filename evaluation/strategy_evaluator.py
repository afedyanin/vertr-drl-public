import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.append("../airflow/plugins")

from strategy_basic import Strategy
from strategy_predictor import StrategyPredictor
from synthetic_data_adapter import DataAdapter


class StrategyEvaluator:
    def __init__(
            self,
            data_adapter: DataAdapter,
            strategy: Strategy):
        self.data_adapter = data_adapter
        self.strategy = strategy
        self.predictor = StrategyPredictor(self.data_adapter, self.strategy)

    def evaluate(
            self,
            start_time_utc: datetime,
            end_time_utc: datetime,
            commission: float = 0.03) -> pd.DataFrame:

        df = self.predictor.evaluate(start_time_utc, end_time_utc)
        df["commission_amount"] = df["open"] * commission
        df["profit_amount"] = (df["close"] - df["open"]).abs() * df["reward"]
        df["profit_amount_with_comission"] = df["profit_amount"] - df["commission_amount"] * df["reward"]
        df["profit_percent"] = df["profit_amount"]/df["open"]
        df["profit_percent_with_comission"] = df["profit_amount_with_comission"]/df["open"]
        df["profit_percent_cum"] = df["profit_percent"].cumsum()
        df["profit_percent_cum_with_comission"] = df["profit_percent_with_comission"].cumsum()

        return df
