from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

mos_tz = ZoneInfo("Europe/Moscow")


def as_moscow_timezone(dtime: str) -> datetime:
    """
    Получает время в виде строки без таймзоны
    и формирует объект datetime с московской таймзоной
    """
    return datetime.fromisoformat(f"{dtime}+03:00")


def to_moscow_timezone(dtime: datetime) -> datetime:
    """
    Конвертирует метку времени в таймзону Москвы
    """
    return dtime.astimezone(mos_tz)


def to_moscow_time_str(dtime: datetime) -> str:
    return dtime.strftime("%Y-%m-%d %H:%M:%S")


def localize_index(df: pd.DataFrame, col_name: str = "timestamp") -> None:
    df[col_name] = df.tz_convert(mos_tz).index.tz_localize(None)
    df.set_index(col_name, inplace=True)

