import uuid

import psycopg
import pandas as pd

from db_connection import DbConnection
from domain_model import Order


class OrdersSqlAdapter:
    def __init__(self, dbconnection: DbConnection):
        self._dbconnection = dbconnection
        self._orders_table = "orders"

    def get_orders(self) -> pd.DataFrame:
        sql = f"SELECT * FROM {self._orders_table}"
        df = pd.read_sql_query(sql, self._dbconnection.engine)
        return df

    def get_order_by_request_id(
            self,
            order_request_id: uuid) -> Order | None:
        with psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {self._orders_table} "
                            f"WHERE order_request_id = '{order_request_id}' LIMIT 1")
                row = cur.fetchone()
                if row is None:
                    return None
                columns = list(cur.description)
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col.name] = row[i]
                return Order.from_dict(row_dict)

    def delete_order(self, order_id: uuid) -> None:
        with psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._orders_table} WHERE order_id = {order_id}")
                conn.commit()

    def insert_order(
            self,
            account_id: str,
            order: Order) -> None:
        with (psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn):
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self._orders_table} ("
                    "order_id, "
                    "account_id, "
                    "order_request_id, "
                    "direction, "
                    "executed_commission, "
                    "executed_order_price, "
                    "execution_report_status, "
                    "initial_commission, "
                    "instrument_uid, "
                    "lots_executed, "
                    "lots_requested, "
                    "message, "
                    "order_type, "
                    "server_time, "
                    "order_json)"
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ",
                    (order.order_id,
                     account_id,
                     order.order_request_id,
                     order.direction,
                     order.executed_commission,
                     order.executed_order_price,
                     order.execution_report_status,
                     order.initial_commission,
                     order.instrument_uid,
                     order.lots_executed,
                     order.lots_requested,
                     order.message,
                     order.order_type,
                     order.server_time,
                     order.order_json))
                conn.commit()
