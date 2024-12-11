import abc
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from domain_model import Interval, Instrument


class DataAdapter(abc.ABC):
    def __init__(
            self,
            interval: Interval,
            instrument: Instrument | None = None,
    ):
        self.interval = interval
        self.instrument = instrument

    @abc.abstractmethod
    def get_candles(self,
                    start_time_utc: datetime,
                    end_time_utc: datetime,
                    ) -> pd.DataFrame:
        pass


class SyntheticDataAdapter(DataAdapter):

    def __init__(
            self,
            func: Callable[[np.array], np.array],
            interval: Interval,
            use_sampling: bool = True,
            instrument: Instrument | None = None
    ):
        super().__init__(interval, instrument)

        if use_sampling:
            self.sampling_freq = self._interval_to_sampling_frequency(self.interval)
        else:
            self.sampling_freq = self._interval_to_frequency(self.interval)

        self._func = func

    def get_candles(self,
                    start_time_utc: datetime,
                    end_time_utc: datetime,
                    ) -> pd.DataFrame:
        x_dates = pd.date_range(start_time_utc, end_time_utc, freq=self.sampling_freq)
        x_values = self._get_x_values(x_dates, 0.1)
        y = self._func(x_values)
        series = pd.Series(y, x_dates, name="seq")
        rule = self._interval_to_frequency(self.interval)
        df = series.resample(rule).ohlc()
        df["time_utc"] = df.index
        return df

    @staticmethod
    def _get_x_values(
            dates: pd.DatetimeIndex,
            step: float = 0.1) -> np.ndarray:
        df_len = len(dates)
        x_stop = df_len * step
        return np.arange(start=0, stop=x_stop, step=step)

    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    @staticmethod
    def _interval_to_frequency(interval: Interval) -> str:
        if interval.equal(Interval.day_1):
            return "1D"
        if interval.equal(Interval.quarter_1):
            return "1QS"
        if interval.equal(Interval.month_1):
            return "1MS"
        if interval.equal(Interval.week_1):
            return "1W"
        if interval.equal(Interval.hour_1):
            return "1h"
        if interval.equal(Interval.min_10):
            return "10min"
        if interval.equal(Interval.min_1):
            return "1min"

        raise ValueError(f"Interval {interval.name} not supported.")

    @staticmethod
    def _interval_to_sampling_frequency(interval: Interval) -> str:
        if interval.equal(Interval.day_1):
            return "1h"
        if interval.equal(Interval.quarter_1):
            return "1MS"
        if interval.equal(Interval.month_1):
            return "1W"
        if interval.equal(Interval.week_1):
            return "1D"
        if interval.equal(Interval.hour_1):
            return "10min"
        if interval.equal(Interval.min_10):
            return "1min"
        if interval.equal(Interval.min_1):
            return "1s"

        raise ValueError(f"Interval {interval.name} not supported.")
