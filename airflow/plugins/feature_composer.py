from datetime import datetime
import pandas as pd
import pandas_ta as ta

from domain_model import Interval
from synthetic_data_adapter import DataAdapter


class FeatureComposer:
    """
    Формирует датафрейм с признаками для обучения
    """

    def __init__(
            self,
            data_adapter: DataAdapter,
            fill_missing_values: bool = False,
    ):
        if data_adapter is None:
            raise ValueError('data_adapter is not set')

        self._data_adapter = data_adapter
        self._fill_missing_values = fill_missing_values
        self._columns_to_drop = [
            'interval', 'symbol', 'high', 'low', 'value', 'volume', 'is_complete']

    def compose(
            self,
            start_time_utc: datetime,
            end_time_utc: datetime) -> pd.DataFrame:

        candles = self._data_adapter.get_candles(start_time_utc, end_time_utc)

        candles.drop(columns=self._columns_to_drop, inplace=True, errors='ignore')
        candles.set_index('time_utc', inplace=True)
        candles.sort_index(inplace=True)

        candles = self._add_returns(candles)
        candles = self._add_returns_std(candles)
        candles = self._add_values(candles)
        candles = self._add_values_std(candles)

        candles["info_intraday_return"] = (candles["close"] - candles["open"]).round(6)
        candles.dropna(inplace=True)

        return candles

    @staticmethod
    def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
        df["feature_close_return"] = (df["close"] - df["close"].shift(1)).round(6)
        df["feature_ret_m1"] = df["feature_close_return"].shift(1)
        df["feature_ret_m2"] = df["feature_close_return"].shift(2)
        df["feature_ret_m3"] = df["feature_close_return"].shift(3)
        df["feature_ret_m4"] = df["feature_close_return"].shift(4)
        return df

    @staticmethod
    def _add_returns_std(df: pd.DataFrame) -> pd.DataFrame:
        df["feature_close_std"] = df["close"].rolling(2).std()
        df["feature_std_m1"] = df["feature_close_std"].shift(1)
        df["feature_std_m2"] = df["feature_close_std"].shift(2)
        df["feature_std_m3"] = df["feature_close_std"].shift(3)
        return df

    @staticmethod
    def _add_values(df: pd.DataFrame) -> pd.DataFrame:
        df["feature_value_return"] = df["value"]#.pct_change(1)
        df["feature_val_m1"] = df["feature_value_return"].shift(1)
        df["feature_val_m2"] = df["feature_value_return"].shift(2)
        df["feature_val_m3"] = df["feature_value_return"].shift(3)
        return df

    @staticmethod
    def _add_values_std(df: pd.DataFrame) -> pd.DataFrame:
        df["feature_value_std"] = df["value"].rolling(2).std()
        df["feature_val_std_m1"] = df["feature_value_std"].shift(1)
        df["feature_val_std_m2"] = df["feature_value_std"].shift(2)
        df["feature_val_std_m3"] = df["feature_value_std"].shift(3)
        return df

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
    def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет индикаторы технического анализа
        https://github.com/twopirllc/pandas-ta
        https://ta-lib.github.io/ta-lib-python/doc_index.html
        """
        # https://www.investopedia.com/articles/technical/100801.asp
        #df.ta.obv(cumulative=False, append=True, talib=True)
        #df.rename(columns={'OBV': 'feature_obv'}, inplace=True)

        # https://www.investopedia.com/terms/a/accumulationdistribution.asp
        #df.ta.ad(cumulative=False, append=True, talib=True)
        #df.rename(columns={'AD': 'feature_ad'}, inplace=True)

        # https://www.investopedia.com/terms/a/adx.asp
        df.ta.adx(cumulative=False, append=True, length=14)
        df.rename(columns=
                  {'ADX_14': 'feature_adx',
                   'DMP_14': 'feature_dmp',
                   'DMN_14': 'feature_dmn'}, inplace=True)

        # https://www.investopedia.com/terms/a/aroonoscillator.asp
        df.ta.aroon(cumulative=False, append=True, talib=True, length=14)
        df.rename(columns=
                  {'AROOND_14': 'feature_aroond',
                   'AROONU_14': 'feature_aroonu',
                   'AROONOSC_14': 'feature_aroonosc'}, inplace=True)

        # https://www.investopedia.com/terms/m/macd.asp
        df.ta.macd(cumulative=False, append=True, talib=True, fast=12, slow=26, signal=9)
        df.rename(columns=
                 {'MACD_12_26_9': 'feature_macd',
                   'MACDh_12_26_9': 'feature_macdh',
                   'MACDs_12_26_9': 'feature_macds'}, inplace=True)

        # https://www.investopedia.com/terms/r/rsi.asp
        df.ta.rsi(cumulative=False, append=True, talib=True, length=14)
        df.rename(columns={'RSI_14': 'feature_rsi'}, inplace=True)

        # https://www.investopedia.com/terms/s/stochasticoscillator.asp
        df.ta.stoch(cumulative=False, append=True, k=14, d=3, smooth_k=3)
        df.rename(columns=
                  {'STOCHk_14_3_3': 'feature_stochk',
                   'STOCHd_14_3_3': 'feature_stochd'}, inplace=True)

        return df
