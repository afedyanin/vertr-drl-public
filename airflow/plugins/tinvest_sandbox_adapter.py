"""
https://github.com/RussianInvestments/invest-python
https://russianinvestments.github.io/invest-python/examples/
"""

import decimal
import uuid
from datetime import datetime, timezone, timedelta
import pandas as pd

from tinkoff.invest.sandbox.client import SandboxClient
from tinkoff.invest import Account, OrderDirection, OrderType, OperationState, Operation, OperationTrade, OrderState, \
    PositionsResponse, Instrument, Client, InstrumentIdType, CandleInterval
from tinkoff.invest import MoneyValue
from tinkoff.invest.schemas import CandleSource, PortfolioResponse
from tinkoff.invest.utils import money_to_decimal, quotation_to_decimal

from domain_model import Order, Operation as DomainOperation, Trade, Instrument as DomainInstrument, PortfolioSnapshot, \
    PositionSnapshot
from json_utils import dataclass_to_json

_SB_TOKEN_ = 't.8DpIsag8_t2bHcaPEXZiAxDLdxbyqP7MXvDwoamPBWSDBD7dgQeMNutgas5Ay83YOlLsA-m8qSPm8Sz-FMaNuw'
SB_TRADING_ACCOUNT_ID = 'fc66cf9b-8fb8-4d9e-ba79-a5e8b87c5aa7'

class TinvestSandboxAdapter:
    def __init__(self,
                 account_id: str | None = None,
                 token: str = _SB_TOKEN_):
        self._token = token
        self.account_id = account_id if account_id else SB_TRADING_ACCOUNT_ID

    def get_instrument(
            self,
            ticker: str = "SBER",
            class_code: str = "TQBR") -> Instrument:
        with Client(self._token) as client:
            response = client.instruments.get_instrument_by(
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                class_code=class_code,
                id=ticker)
            return response.instrument

    def find_instrument(self, query: str) -> list[Instrument]:
        with Client(self._token) as client:
            response = client.instruments.find_instrument(query=query)
            return response.instruments

    def get_candles(
            self,
            instrument: DomainInstrument,
            start_date_utc: datetime | None = None,
            end_date_utc: datetime | None = None,
            interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_10_MIN) -> pd.DataFrame:

        if end_date_utc is None:
            end_date_utc = datetime.now(timezone.utc)

        if start_date_utc is None:
            start_date_utc = end_date_utc - timedelta(days=2)

        with Client(self._token) as client:
            candles = client.get_all_candles(
                instrument_id=instrument.instrument_id,
                from_=start_date_utc,
                to=end_date_utc,
                interval=interval,
                candle_source_type=CandleSource.CANDLE_SOURCE_EXCHANGE,
            )
            df = pd.DataFrame([{
                'time_utc': c.time,
                'open': pd.to_numeric(quotation_to_decimal(c.open)),
                'high': pd.to_numeric(quotation_to_decimal(c.high)),
                'low': pd.to_numeric(quotation_to_decimal(c.low)),
                'close': pd.to_numeric(quotation_to_decimal(c.close)),
                'volume': pd.to_numeric(c.volume),
                'is_complete': c.is_complete
            } for c in candles])

            return df

    def open_sandbox_account(self) -> uuid:
        with SandboxClient(self._token) as client:
            response = client.sandbox.open_sandbox_account()
            return response.account_id

    def close_sandbox_account(self, account_id: str):
        with SandboxClient(self._token) as client:
            response = client.sandbox.close_sandbox_account(account_id=account_id)
            return response

    def get_accounts(self) -> list[Account]:
        with SandboxClient(self._token) as client:
            response = client.users.get_accounts()
            return response.accounts

    def deposit_sandbox_account(self, amount: decimal) -> decimal:
        money_value = MoneyValue(currency="rub", units=amount)
        with SandboxClient(self._token) as client:
            response = client.sandbox.sandbox_pay_in(
                account_id=self.account_id,
                amount=money_value)
            balance = money_to_decimal(response.balance)
            return balance

    def post_market_order(
            self,
            order_request_id: uuid,
            instrument: DomainInstrument,
            quantity: int = 1) -> Order:

        if quantity > 0:
            direction = OrderDirection.ORDER_DIRECTION_BUY
        elif quantity < 0:
            direction = OrderDirection.ORDER_DIRECTION_SELL
        else:
            raise ValueError("Quantity must be positive or negative")

        quantity = abs(quantity)

        with SandboxClient(self._token) as client:
            response = client.orders.post_order(
                order_id=str(order_request_id),
                quantity=quantity,
                direction=direction,
                account_id=self.account_id,
                order_type=OrderType.ORDER_TYPE_MARKET,
                instrument_id=instrument.instrument_id)

            return Order(
                order_id=uuid.UUID(response.order_id),
                account_id=self.account_id,
                order_request_id=uuid.UUID(response.order_request_id),
                direction=response.direction,
                executed_commission=money_to_decimal(response.executed_commission),
                executed_order_price=money_to_decimal(response.executed_order_price),
                execution_report_status=response.execution_report_status,
                initial_commission=money_to_decimal(response.initial_commission),
                instrument_uid=response.instrument_uid,
                lots_executed=response.lots_executed,
                lots_requested=response.lots_requested,
                message=response.message,
                order_type=response.order_type,
                server_time=response.response_metadata.server_time,
                order_json=dataclass_to_json(response))

    def get_order_state(self, order_id: uuid) -> OrderState:
        with SandboxClient(self._token) as client:
            response = client.orders.get_order_state(account_id=self.account_id, order_id=order_id)
            return response

    def get_positions(self) -> PositionsResponse:
        with SandboxClient(self._token) as client:
            response = client.operations.get_positions(account_id=self.account_id)
            return response

    def get_portfolio(self) -> PortfolioResponse:
        with SandboxClient(self._token) as client:
            response = client.operations.get_portfolio(account_id=self.account_id)
            return response

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        with SandboxClient(self._token) as client:
            response = client.operations.get_portfolio(account_id=self.account_id)
            return PortfolioSnapshot(
                account_id=self.account_id,
                time_utc=datetime.now(tz=timezone.utc),
                shares=money_to_decimal(response.total_amount_shares),
                bonds=money_to_decimal(response.total_amount_bonds),
                futures=money_to_decimal(response.total_amount_futures),
                options=money_to_decimal(response.total_amount_options),
                currencies=money_to_decimal(response.total_amount_currencies),
                portfolio=money_to_decimal(response.total_amount_portfolio),
                expected_yield=quotation_to_decimal(response.expected_yield),
            )

    def get_positions_snapshot(self) -> list[PositionSnapshot]:
        with SandboxClient(self._token) as client:
            response = client.operations.get_positions(account_id=self.account_id)
            positions = []
            current_time = datetime.now(tz=timezone.utc)
            for money in response.money:
                position = PositionSnapshot(
                    account_id=self.account_id,
                    time_utc=current_time,
                    instrument_str=money.currency,
                    balance=money_to_decimal(money),
                    instrument_uid="")
                positions.append(position)

            for security in response.securities:
                position = PositionSnapshot(
                    account_id=self.account_id,
                    time_utc=current_time,
                    instrument_str="",
                    balance=security.balance,
                    instrument_uid=security.instrument_uid)
                positions.append(position)

            for security in response.futures:
                position = PositionSnapshot(
                    account_id=self.account_id,
                    time_utc=current_time,
                    instrument_str="",
                    balance=security.balance,
                    instrument_uid=security.instrument_uid)
                positions.append(position)

            for security in response.options:
                position = PositionSnapshot(
                    account_id=self.account_id,
                    time_utc=current_time,
                    instrument_str="",
                    balance=security.balance,
                    instrument_uid=security.instrument_uid)
                positions.append(position)

            return positions

    def get_position_balance(self, instrument: DomainInstrument):
        with SandboxClient(self._token) as client:
            positions = client.operations.get_positions(account_id=self.account_id)
            if instrument.engine == "stock":
                item = next((obj for obj in positions.securities if
                             getattr(obj, 'instrument_uid', None) == instrument.instrument_id), None)
            elif instrument.engine == "futures":
                item = next((obj for obj in positions.futures if
                             getattr(obj, 'instrument_uid', None) == instrument.instrument_id), None)
            elif instrument.engine == "options":
                item = next((obj for obj in positions.options if
                             getattr(obj, 'instrument_uid', None) == instrument.instrument_id), None)
            return 0 if item is None else item.balance

    def get_operations(self,
                       date_from: datetime | None = None,
                       date_to: datetime | None = None,
                       instrument_figi: str = "") -> list[DomainOperation]:
        with SandboxClient(self._token) as client:
            response = client.operations.get_operations(
                account_id=self.account_id,
                from_=date_from,
                to=date_to,
                state=OperationState.OPERATION_STATE_UNSPECIFIED,
                figi=instrument_figi)
            operations = []
            for operation in response.operations:
                op = self._create_operation(operation)
                operations.append(op)
            return operations

    def _create_operation(self, operation: Operation) -> DomainOperation:
        trades = []
        for op_trade in operation.trades:
            trade = self._create_trade(operation.id, operation.operation_type, op_trade)
            trades.append(trade)

        return DomainOperation(
            id=operation.id,
            account_id=self.account_id,
            parent_operation_id=operation.parent_operation_id,
            currency=operation.currency,
            payment=money_to_decimal(operation.payment),
            price=money_to_decimal(operation.price),
            state=operation.state,
            quantity=operation.quantity,
            quantity_rest=operation.quantity_rest,
            figi=operation.figi,
            instrument_type=operation.instrument_type,
            date=operation.date,
            type=operation.type,
            operation_type=operation.operation_type,
            asset_uid=operation.asset_uid,
            position_uid=operation.position_uid,
            instrument_uid=operation.instrument_uid,
            trades=trades,
            operation_json=dataclass_to_json(operation))

    @staticmethod
    def _create_trade(
            operation_id: uuid,
            operation_type: int,
            op_trade: OperationTrade) -> Trade:
        return Trade(
            operation_id=operation_id,
            operation_type=operation_type,
            trade_id=op_trade.trade_id,
            date_time=op_trade.date_time,
            quantity=op_trade.quantity,
            price=money_to_decimal(op_trade.price),
            trade_json=dataclass_to_json(op_trade))
