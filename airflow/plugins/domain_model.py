import decimal
import enum
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@enum.unique
class Interval(enum.Enum):
    """
    Числовые коды интервалов для свечей:
        - 1 - 1 минута
        - 10 - 10 минут
        - 60 - 1 час
        - 24 - 1 день
        - 7 - 1 неделя
        - 31 - 1 месяц
        - 4 - 1 квартал
    """
    min_1 = 1
    min_10 = 10
    hour_1 = 60
    day_1 = 24
    week_1 = 7
    month_1 = 31
    quarter_1 = 4

    def equal(self, interval: "Interval") -> bool:
        return self.value == interval.value


class Instrument:
    """
    Финансовый инструмент: Акция, Индекс, Облигация и т.п.
    """

    def __init__(
            self,
            symbol: str,
            isin: str,
            instrument_id: str,
            figi: str | None = None,
            engine="stock",
            market="shares",
            board="TQBR",
            lot_size: int = 10,
    ):
        self.symbol = symbol
        self.isin = isin
        self.instrument_id = instrument_id
        self.figi = figi
        self.engine = engine
        self.market = market
        self.board = board
        self.lot_size = lot_size

    def __str__(self):
        return f"symbol={self.symbol} isin={self.isin} engine={self.engine} market={self.market} board={self.board}"

    @staticmethod
    def get_instrument(symbol: str):
        """
        Возвращает объект инструмента по заданному символу
        """
        return Instrument._all_instruments()[symbol]

    @staticmethod
    def _all_instruments() -> dict:
        """
        Инструменты, торгуемые в системе
        Пример запроса к ИСС: https://iss.moex.com/iss/securities/SBER.xml
        """
        return {
            "SBER": Instrument(
                symbol="SBER",
                isin="RU0009029540",
                instrument_id="e6123145-9665-43e0-8413-cd61b8aa9b13",
                figi="BBG004730N88",
                engine="stock",
                market="shares",
                board="TQBR",
                lot_size=10),
            "SBERF": Instrument(
                symbol="SBERF",
                isin="",
                instrument_id="9e9c5921-43ac-47a8-b537-ccf32ebc6da3",
                figi="FUTSBERF0000",
                engine="futures",
                market="forts",
                board="SPBFUT",
                lot_size=10 # TODO: Check this
                ),
        }


class Candle:
    """
    Объект ценовой свечи
    """

    def __init__(self,
                 time_utc: datetime,
                 interval: int,
                 symbol: str,
                 open: decimal,
                 close: decimal,
                 high: decimal,
                 low: decimal,
                 value: decimal,
                 volume: decimal,
                 ):
        self.time_utc = time_utc
        self.interval = interval
        self.symbol = symbol
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.value = value
        self.volume = volume

    def __str__(self):
        return (f"Candle: time_utc={self.time_utc} symbol={self.symbol} interval={self.interval} "
                f"open={self.open} close={self.close} high={self.high} low={self.low} "
                f"value={self.value} volume={self.volume}")

    @staticmethod
    def from_tuple(candle: tuple):
        """
        Создает свечу из sql кортежа
        """
        if candle is None:
            return None

        return Candle(
            time_utc=candle[0],
            interval=candle[1],
            symbol=candle[2],
            open=candle[3],
            close=candle[4],
            high=candle[5],
            low=candle[6],
            value=candle[7],
            volume=candle[8],
        )


@dataclass
class Order:
    order_id: uuid
    account_id: str
    order_request_id: uuid
    direction: int
    executed_commission: decimal
    executed_order_price: decimal
    execution_report_status: int
    initial_commission: decimal
    instrument_uid: uuid
    lots_executed: int
    lots_requested: int
    message: str
    order_type: int
    server_time: datetime
    order_json: str

    @staticmethod
    def from_dict(order: dict[str, Any]) -> "Order":
        return Order(
            order_id=order["order_id"],
            account_id=order["account_id"],
            order_request_id=order["order_request_id"],
            direction=order["direction"],
            executed_commission=order["executed_commission"],
            executed_order_price=order["executed_order_price"],
            execution_report_status=order["execution_report_status"],
            initial_commission=order["initial_commission"],
            instrument_uid=order["instrument_uid"],
            lots_executed=order["lots_executed"],
            lots_requested=order["lots_requested"],
            message=order["message"],
            order_type=order["order_type"],
            server_time=order["server_time"],
            order_json=order["order_json"])


@dataclass
class Trade:
    operation_id: uuid
    operation_type: int
    trade_id: uuid
    date_time: datetime
    quantity: int
    price: decimal
    trade_json: str

    @staticmethod
    def from_dict(trade: dict[str, Any]) -> "Trade":
        return Trade(
            operation_id=trade["operation_id"],
            operation_type=trade["operation_type"],
            trade_id=trade["trade_id"],
            date_time=trade["date_time"],
            quantity=trade["quantity"],
            price=trade["price"],
            trade_json=trade["trade_json"])


@dataclass
class Operation:
    id: uuid
    account_id: str
    parent_operation_id: str | None
    currency: str
    payment: decimal
    price: decimal
    state: int
    quantity: int
    quantity_rest: int
    figi: str
    instrument_type: str
    date: datetime
    type: str
    operation_type: int
    asset_uid: str | None
    position_uid: str | None
    instrument_uid: uuid
    trades: list[Trade]
    operation_json: str

    @staticmethod
    def from_dict(operation: dict[str, Any]) -> "Operation":
        return Operation(
            id=operation["id"],
            account_id=operation["account_id"],
            parent_operation_id=operation["parent_operation_id"],
            currency=operation["currency"],
            payment=operation["payment"],
            price=operation["price"],
            state=operation["state"],
            quantity=operation["quantity"],
            quantity_rest=operation["quantity_rest"],
            figi=operation["figi"],
            instrument_type=operation["instrument_type"],
            date=operation["date"],
            type=operation["type"],
            operation_type=operation["operation_type"],
            asset_uid=operation["asset_uid"],
            position_uid=operation["position_uid"],
            instrument_uid=operation["instrument_uid"],
            trades=operation["trades"],
            operation_json=operation["operation_json"])


@dataclass
class Signal:
    order_request_id: uuid
    time_utc: datetime
    quantity: int
    symbol: str
    origin: str
    prediction: str

    @staticmethod
    def from_dict(signal: dict[str, Any]) -> "Signal":
        return Signal(
            order_request_id=signal['order_request_id'],
            time_utc=signal['time_utc'],
            quantity=signal['quantity'],
            symbol=signal['symbol'],
            origin=signal['origin'],
            prediction=signal['prediction'])


@dataclass
class PortfolioSnapshot:
    time_utc: datetime
    account_id: str
    shares: decimal
    bonds: decimal
    futures: decimal
    options: decimal
    currencies: decimal
    portfolio: decimal
    expected_yield: decimal


@dataclass
class PositionSnapshot:
    time_utc: datetime
    account_id: str
    instrument_uid: str
    instrument_str: str
    balance: decimal


