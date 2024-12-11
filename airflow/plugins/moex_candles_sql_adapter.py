import psycopg
import pandas as pd
from datetime import datetime, timezone

from db_connection import DbConnection
from domain_model import Instrument, Interval
from moex_iss_adapter import moex_get_candles
from synthetic_data_adapter import DataAdapter
from time_utils import as_moscow_timezone


class CandlesSqlAdapter(DataAdapter):
    """
    Модуль SQL адаптера для работы со свечами.
    Выполняет CRUD SQL операции над таблицей со свечами.
    """
    def __init__(
            self,
            dbconnection: DbConnection,
            interval: Interval,
            instrument: Instrument
    ):
        super().__init__(interval, instrument)
        self._dbconnection = dbconnection
        self._candles_table = "moex_candles"

    def get_candles(self,
                    start_date_utc: datetime | None = None,
                    end_date_utc: datetime | None = None) -> pd.DataFrame:
        """
        Получить датафрейм со свечами на заданный интервал дат.
        Интервалы задаются в utc формате
        """
        if start_date_utc is None:
            start_date_utc = datetime(2010, 1, 1, tzinfo=timezone.utc)

        if end_date_utc is None:
            end_date_utc = datetime.now(timezone.utc)

        sql = (f"SELECT * FROM {self._candles_table} "
               f"WHERE time_utc >= '{start_date_utc}' AND time_utc <= '{end_date_utc}' "
               f"AND symbol = '{self.instrument.symbol}' "
               f"AND interval = '{self.interval.value}'"
               )
        df = pd.read_sql_query(sql, self._dbconnection.engine)
        return df

    def get_last_time_utc(self) -> datetime | None:
        """
        Получить последнюю известную дату/время свечи
        """
        res = self.get_last_candle()

        if res is None:
            return None

        return res[0]

    def get_first_time_utc(self) -> datetime | None:
        """
        Получить первую известную дату/время свечи
        """
        res = self.get_first_candle()

        if res is None:
            return None

        return res[0]

    def get_last_candle(self):
        """
        Получить последнюю известную свечу
        """
        with psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {self._candles_table} "
                            f"WHERE symbol = '{self.instrument.symbol}' "
                            f"AND interval = {self.interval.value} "
                            "ORDER BY time_utc DESC LIMIT 1;")
                res = cur.fetchone()
                return res

    def get_first_candle(self):
        """
        Получить первую известную свечу
        """
        with psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {self._candles_table} "
                            f"WHERE symbol = '{self.instrument.symbol}' "
                            f"AND interval = {self.interval.value} "
                            "ORDER BY time_utc LIMIT 1;")
                res = cur.fetchone()
                return res

    def clear_candles(self) -> None:
        """
        Очистить таблицу свечей от заданного символа и интервала
        """
        with psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._candles_table} "
                            f"WHERE symbol = '{self.instrument.symbol}' "
                            f"AND interval = {self.interval.value}")
            conn.commit()

    def import_candles(self,
                       start_date_utc: datetime | None = None,
                       end_date_utc: datetime | None = None) -> int:
        """
        Импортировать свечи заданного символа и интервала из МОЕХ в SQL
        """

        candles = moex_get_candles(
            instrument=self.instrument,
            interval=self.interval,
            start_date_utc=start_date_utc,
            end_date_utc=end_date_utc
        )

        candles_count = len(candles)
        if candles_count == 0:
            print("No candles found. Nothing to import.")
            return 0

        last_candle_time = candles[-1]['begin']
        last_candle_time_mos = as_moscow_timezone(last_candle_time)
        last_candle_time_utc = last_candle_time_mos.astimezone(timezone.utc)

        print(f"{candles_count} candles loaded from MOEX. "
              f"Last candle time: {last_candle_time_mos} (utc: {last_candle_time_utc})")

        inserted_rows = self._insert_candles(candles)
        return inserted_rows

    def _insert_candles(self, candles: list) -> int:
        """
        Вставить список свечей в БД
        """
        with (psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn):
            with conn.cursor() as cur:
                count = 0
                for record in candles:
                    cur.execute(
                        f"INSERT INTO {self._candles_table} ("
                        "time_utc, interval, symbol, open, close, high, low, value, volume)"
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT ON CONSTRAINT moex_candles_unique DO UPDATE "
                        "SET open = EXCLUDED.open, "
                        "close = EXCLUDED.close, "
                        "high = EXCLUDED.high, "
                        "low = EXCLUDED.low, "
                        "value = EXCLUDED.value, "
                        "volume = EXCLUDED.volume ",
                        (
                            as_moscow_timezone(record['begin']),
                            self.interval.value,
                            self.instrument.symbol,
                            record['open'],
                            record['close'],
                            record['high'],
                            record['low'],
                            record['value'],
                            record['volume']
                        ))
                    count = count + cur.rowcount
                conn.commit()

                return count
