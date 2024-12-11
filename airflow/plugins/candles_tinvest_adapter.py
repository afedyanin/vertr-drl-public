import pandas as pd
from datetime import datetime, timezone, timedelta

from domain_model import Instrument, Interval
from synthetic_data_adapter import DataAdapter
from tinvest_sandbox_adapter import TinvestSandboxAdapter


class CandlesTinvestAdapter(DataAdapter):
    def __init__(
            self,
            interval: Interval,
            instrument: Instrument,
            account_id: str | None = None):
        super().__init__(interval, instrument)
        self._tinvest_adapter = TinvestSandboxAdapter(account_id)

    def get_candles(self,
                    start_date_utc: datetime | None = None,
                    end_date_utc: datetime | None = None) -> pd.DataFrame:

        if end_date_utc is None:
            end_date_utc = datetime.now(timezone.utc)

        if start_date_utc is None:
            start_date_utc = end_date_utc - timedelta(days=2)

        # TODO: Convert Interval to T-Invest interval
        candles = self._tinvest_adapter.get_candles(
            instrument=self.instrument,
            start_date_utc=start_date_utc,
            end_date_utc=end_date_utc)

        candles = candles[candles['is_complete']]

        return candles

