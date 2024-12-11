import uuid

import psycopg
import pandas as pd

from db_connection import DbConnection
from domain_model import Operation, Trade


class OperationsSqlAdapter:
    def __init__(self, dbconnection: DbConnection):
        self._dbconnection = dbconnection
        self._operations_table = "operations"
        self._trades_table = "trades"

    def get_operations(
            self,
            account_id: str,
            instrument_id: str,
            ) -> pd.DataFrame:
        sql = (f"SELECT * FROM {self._operations_table} "
               f"WHERE account_id = '{account_id}' "
               f"AND instrument_uid = '{instrument_id}'")
        df = pd.read_sql_query(sql, self._dbconnection.engine)
        return df

    def get_trades(self) -> pd.DataFrame:
        sql = f"SELECT * FROM {self._trades_table}"
        df = pd.read_sql_query(sql, self._dbconnection.engine)
        return df

    def delete_operations(self) -> None:
        with psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._operations_table} ")
                cur.execute(f"DELETE FROM {self._trades_table} ")
                conn.commit()

    def insert_operations(
            self,
            account_id: str,
            operations: list[Operation]) -> None:
        with (psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn):
            with conn.cursor() as cur:
                for operation in operations:
                    cur.execute(
                        f"INSERT INTO {self._operations_table} ("
                        "id, "
                        "account_id, "
                        "parent_operation_id, "
                        "currency, "
                        "payment, "
                        "price, "
                        "state, "
                        "quantity, "
                        "quantity_rest, "
                        "figi, "
                        "instrument_type, "
                        "date, "
                        "type, "
                        "operation_type, "
                        "asset_uid, "
                        "position_uid, "
                        "instrument_uid, "
                        "operation_json)"
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (id) DO UPDATE "
                        "SET payment = EXCLUDED.payment, "
                        "price = EXCLUDED.price, "
                        "state = EXCLUDED.state, "
                        "quantity = EXCLUDED.quantity, "
                        "quantity_rest = EXCLUDED.quantity_rest, "
                        "date = EXCLUDED.date, "
                        "asset_uid = EXCLUDED.asset_uid, "
                        "position_uid = EXCLUDED.position_uid, "
                        "instrument_uid = EXCLUDED.instrument_uid, "
                        "operation_json = EXCLUDED.operation_json ",
                        (operation.id,
                         account_id,
                         operation.parent_operation_id,
                         operation.currency,
                         operation.payment,
                         operation.price,
                         operation.state,
                         operation.quantity,
                         operation.quantity_rest,
                         operation.figi,
                         operation.instrument_type,
                         operation.date,
                         operation.type,
                         operation.operation_type,
                         operation.asset_uid,
                         operation.position_uid,
                         operation.instrument_uid,
                         operation.operation_json
                         ))
                    for trade in operation.trades:
                        cur.execute(
                            f"INSERT INTO {self._trades_table} ("
                            "operation_id, "
                            "operation_type, "
                            "trade_id, "
                            "date_time, "
                            "quantity, "
                            "price, "
                            "trade_json)"
                            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                            "ON CONFLICT (trade_id) DO NOTHING ",
                            (trade.operation_id,
                             trade.operation_type,
                             trade.trade_id,
                             trade.date_time,
                             trade.quantity,
                             trade.price,
                             trade.trade_json
                             ))
                conn.commit()
