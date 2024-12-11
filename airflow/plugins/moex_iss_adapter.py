from datetime import datetime, timezone

import requests
import apimoex

from domain_model import Instrument, Interval
from time_utils import as_moscow_timezone, to_moscow_timezone, to_moscow_time_str


def moex_get_candles(
        instrument: Instrument,
        interval: Interval,
        start_date_utc: datetime | None = None,
        end_date_utc: datetime | None = None) -> list:
    """
    Получить свечи для заданного инструмента и интервала
    """
    start_date_mos, end_date_mos = convert_to_moex_dates(
        instrument,
        interval,
        start_date_utc,
        end_date_utc)

    print(f"Requesting moex {interval.name} candles {instrument.symbol} "
          f"from={start_date_mos} to={end_date_mos}")

    with requests.Session() as session:
        data = apimoex.get_market_candles(
            session,
            security=instrument.symbol,
            market=instrument.market,
            interval=interval.value,
            engine=instrument.engine,
            start=start_date_mos,
            end=end_date_mos)

        return data


def convert_to_moex_dates(
        instrument: Instrument,
        interval: Interval,
        start_date_utc: datetime | None = None,
        end_date_utc: datetime | None = None) -> tuple:

    known_start_date = datetime(2010, 1, 1, tzinfo=timezone.utc)
    known_end_date = datetime(2010, 1, 1, tzinfo=timezone.utc)

    if start_date_utc or end_date_utc is None:
        known_start_date, known_end_date = moex_get_candle_intervals(instrument=instrument, interval=interval)

    if start_date_utc is None:
        start_date_utc = known_start_date

    if end_date_utc is None:
        end_date_utc = known_end_date

    start_date_mos = to_moscow_timezone(start_date_utc)
    end_date_mos = to_moscow_timezone(end_date_utc)

    return to_moscow_time_str(start_date_mos), to_moscow_time_str(end_date_mos)


def moex_get_candle_intervals(
        instrument: Instrument,
        interval: Interval) -> tuple:
    """
    Получить диапазон доступных дат для свечей различного размера
    """
    with requests.Session() as session:
        intervals = apimoex.get_board_candle_borders(
            session,
            engine=instrument.engine,
            market=instrument.market,
            security=instrument.symbol,
            board=instrument.board)

        entry = next(item for item in intervals if item["interval"] == interval.value)
        start_date_mos = as_moscow_timezone(entry['begin'])
        end_date_mos = as_moscow_timezone(entry['end'])
        return start_date_mos.astimezone(timezone.utc), end_date_mos.astimezone(timezone.utc)

