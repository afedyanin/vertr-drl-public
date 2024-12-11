import uuid

import psycopg
import pandas as pd

from db_connection import DbConnection
from domain_model import Signal


class SignalsSqlAdapter:
    def __init__(self, dbconnection: DbConnection):
        self._dbconnection = dbconnection
        self._signals_table = "signals"

    def get_signals(self) -> pd.DataFrame:
        sql = f"SELECT * FROM {self._signals_table}"
        df = pd.read_sql_query(sql, self._dbconnection.engine)
        return df

    def get_last_signal(self):
        with psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {self._signals_table} "
                            f"ORDER BY time_utc DESC LIMIT 1")
                row = cur.fetchone()
                if row is None:
                    return None
                columns = list(cur.description)
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col.name] = row[i]
                return Signal.from_dict(row_dict)

    def delete_signal(self, order_request_id: uuid) -> None:
        with psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._signals_table} WHERE order_request_id = {order_request_id}")
                conn.commit()

    def insert_signal(self, signal: Signal) -> None:
        with psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self._signals_table} ("
                    "order_request_id, "
                    "time_utc, "
                    "quantity, "
                    "symbol, "
                    "origin, "
                    "prediction)"
                    "VALUES (%s, %s, %s, %s, %s, %s) ",
                    (signal.order_request_id,
                     signal.time_utc,
                     signal.quantity,
                     signal.symbol,
                     signal.origin,
                     signal.prediction))
                conn.commit()
