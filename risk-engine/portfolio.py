import sys
from datetime import datetime, timezone, timedelta

import pandas as pd
from tinkoff.invest.utils import money_to_decimal, quotation_to_decimal

sys.path.append("../airflow/plugins")

from db_connection import DbConnection
from domain_model import Instrument, Interval
from tinvest_sandbox_adapter import TinvestSandboxAdapter
from operations_sql_adapter import OperationsSqlAdapter


class PortfolioSummary:
    def __init__(self):
        self._tinvest_adapter = TinvestSandboxAdapter()

    def dump(self):
        print('----------')
        print('Portfolio')
        portfolio = self._tinvest_adapter.get_portfolio_snapshot()
        print(f'portfolio={portfolio.portfolio:,.2f}')
        print(f'currencies={portfolio.currencies:,.2f}')
        print(f'shares={portfolio.shares:,.2f}')
        print(f'expected_yield(%)={portfolio.expected_yield:.4f}')
        print('----------')
        print(f'Positions')
        positions = self._tinvest_adapter.get_positions_snapshot()
        for position in positions:
            print(f'{position.instrument_uid}{position.instrument_str}={position.balance:,.2f}')
        print('----------')


class Portfolio:
    def __init__(self,
                 db_connection: DbConnection,
                 instrument: Instrument,
                 interval: Interval,
                 ):
        self._db_connection = db_connection
        self.instrument = instrument
        self.interval = interval
        self._tinvest_adapter = TinvestSandboxAdapter()
        self._operations_adapter = OperationsSqlAdapter(self._db_connection)

    def get_pnl(self) -> pd.DataFrame:
        trades = self.get_trading_pnl()
        first_date = trades.index[0]
        last_date = trades.index[-1] + timedelta(hours=2)
        candles = self._get_candles(first_date, last_date)

        pnl_df = candles.join(trades, how='outer')
        pnl_df.fillna(value=0, inplace=True)
        pnl_df["quantity_cum"] = pnl_df["quantity"].cumsum()
        pnl_df['mtm_pnl'] = (pnl_df['close'] - pnl_df['close'].shift(1)) * pnl_df['quantity_cum']
        pnl_df['total_pnl'] = pnl_df['trading_pnl'] + pnl_df['mtm_pnl']
        pnl_df['total_pnl_cum'] = pnl_df['total_pnl'].cumsum()
        return pnl_df

    def get_trading_pnl(
            self,
            grouped: bool = True) -> pd.DataFrame:
        operations_df = self._get_operations()
        first_date = operations_df.index[0]
        last_date = operations_df.index[-1] + timedelta(hours=2)
        candles_df = self._get_candles(first_date, last_date)

        trading_df = operations_df.join(candles_df, how='left')
        trading_df.fillna(value=0, inplace=True)
        trading_df['trading_pnl'] = (trading_df['close'] - trading_df['price']) * trading_df['quantity']

        if grouped:
            trading_df = trading_df.groupby('time_utc')[['quantity', 'commission', 'trading_pnl']].sum()

        return trading_df

    @staticmethod
    def get_yield_from_operations(
            operations_df: pd.DataFrame,
            initial_balance: float) -> pd.DataFrame:
        yield_df = operations_df.copy()
        yield_df['volume'] = (yield_df['price'] * yield_df['quantity'])
        yield_df['volume_cum'] = yield_df['volume'].cumsum()
        yield_df['shares_qty'] = yield_df['quantity'].cumsum()
        yield_df['commission_cum'] = yield_df['commission'].cumsum()
        yield_df['currencies'] = initial_balance - yield_df['volume_cum']
        yield_df['currencies_with_commission'] = yield_df['currencies'] + yield_df['commission_cum']
        yield_df['shares_sum'] = yield_df['shares_qty'] * yield_df['price']
        yield_df['portfolio'] = yield_df['currencies'] + yield_df['shares_sum']
        yield_df['portfolio_with_commission'] = yield_df['currencies_with_commission'] + yield_df[
            'shares_sum']
        yield_df['yield'] = ((yield_df['portfolio'] - initial_balance) / initial_balance) * 100
        yield_df['yield_with_commission'] = ((yield_df[
                                                  'portfolio_with_commission'] - initial_balance) / initial_balance) * 100
        yield_df.drop(columns=['price', 'quantity', 'commission',
                               'volume', 'volume_cum', 'shares_qty',
                               'commission_cum', 'currencies', 'currencies_with_commission',
                               'shares_sum', 'portfolio', 'portfolio_with_commission'], inplace=True)

        return yield_df

    def get_yield(self, initial_balance: float) -> pd.DataFrame:
        ops_df = self._get_operations(grouped=True)
        return self.get_yield_from_operations(ops_df, initial_balance)

    def _get_operations(self, grouped: bool = False) -> pd.DataFrame:
        operations_df = self._operations_adapter.get_operations(
            account_id=self._tinvest_adapter.account_id,
            instrument_id=self.instrument.instrument_id)

        operations_df.drop(
            columns=['id', 'account_id', 'parent_operation_id', 'state', 'quantity_rest', 'currency', 'figi',
                     'instrument_type', 'instrument_uid', 'asset_uid', 'position_uid', 'operation_json'], inplace=True)
        operations_df['time_utc'] = operations_df['date'].apply(
            lambda x: datetime(x.year, x.month, x.day, x.hour, (x.minute // 10) * 10, tzinfo=timezone.utc))
        operations_df['direction'] = operations_df['operation_type'].apply(self._get_direction)
        operations_df['quantity'] = operations_df['quantity'].mul(operations_df['direction'])
        operations_df['commission'] = operations_df['operation_type'].apply(
            self._get_commission).mul(operations_df['payment'])

        operations_df.drop(columns=['date', 'payment', 'type', 'operation_type', 'direction'], inplace=True)
        operations_df.set_index('time_utc', inplace=True)

        if grouped:
            operations_df = operations_df.groupby('time_utc')[['price', 'quantity', 'commission']].sum()

        operations_df.sort_index(inplace=True)
        return operations_df

    def _get_candles(self,
                     first_date: datetime,
                     last_date: datetime) -> pd.DataFrame:

        # TODO: Use candles from SQL DB
        candles_df = self._tinvest_adapter.get_candles(
            instrument=self.instrument,
            start_date_utc=first_date,
            end_date_utc=last_date)

        candles_df.drop(columns=['high', 'low', 'volume', 'is_complete'], inplace=True)
        candles_df.set_index('time_utc', inplace=True)
        candles_df.sort_index(inplace=True)
        return candles_df

    @staticmethod
    def _get_direction(operation_type: int) -> int:
        if operation_type == 15:
            return 1
        if operation_type == 22:
            return -1
        return 0

    @staticmethod
    def _get_commission(operation_type: int) -> int:
        if operation_type == 19:
            return 1
        return 0
