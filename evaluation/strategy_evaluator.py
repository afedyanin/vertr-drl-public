import sys
from datetime import datetime, timezone

import pandas as pd

from gym_env_single_asset import Actions

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
            commission: float = 0.0) -> pd.DataFrame:
        df = self.predictor.evaluate(start_time_utc, end_time_utc)
        df["commission_amount"] = df["open"] * commission
        df["profit_amount"] = (df["close"] - df["open"]).abs() * df["reward"]
        df["profit_amount_with_comission"] = (df["profit_amount"] - df["commission_amount"]) * df["reward"]
        df["profit_percent"] = df["profit_amount"] / df["open"]
        df["profit_percent_with_comission"] = df["profit_amount_with_comission"] / df["open"]
        df["profit_percent_cum"] = df["profit_percent"].cumsum()
        df["profit_percent_cum_with_comission"] = df["profit_percent_with_comission"].cumsum()

        return df

    def get_operations(
            self,
            start_time_utc: datetime,
            end_time_utc: datetime,
            quantity_lots: int,
            lot_size: int,
            commission: float = 0.0) -> pd.DataFrame:
        df = self.predictor.evaluate(start_time_utc, end_time_utc)
        time_index = []
        prices = []
        quantities = []
        commissions = []

        position = 0

        for index, row in df.iterrows():
            action = row["action_pred"]
            order_quantity = self._get_order_quantity(position, action, quantity_lots, lot_size)
            if order_quantity == 0:
                continue
            quantity = order_quantity * lot_size
            time_index.append(index)
            price = (row["open"] + row["close"]) / 2.0
            prices.append(price)
            commissions.append(commission)
            quantities.append(quantity)
            position += quantity

        df_ops = pd.DataFrame(
            index=time_index,
            data={
                "price": prices,
                "quantity": quantities,
                "commission": commissions})

        return df_ops

    @staticmethod
    def _get_order_quantity(
            position_balance: int,
            action: Actions,
            quantity_lots: int,
            lot_size: int) -> int:

        signal = StrategyEvaluator._get_signal(action)
        signal_quantity = signal * quantity_lots

        if position_balance is None or position_balance == 0:
            # just open position
            return signal_quantity

        position_balance_lots = position_balance // lot_size

        if signal_quantity > 0:
            if position_balance > 0:
                # same direction - keep position
                return 0
            else:
                # diff direction - swap position
                return position_balance_lots * (-1) + signal_quantity

        if signal_quantity < 0:
            if position_balance > 0:
                # diff direction swap position
                return position_balance_lots * (-1) + signal_quantity
            else:
                # same direction - keep position
                return 0

        return 0

    @staticmethod
    def _get_signal(action: Actions) -> int:
        if action == Actions.Buy:
            return 1
        if action == Actions.Sell:
            return -1
        return 0



