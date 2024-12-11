import psycopg
import pandas as pd

from db_connection import DbConnection
from domain_model import PortfolioSnapshot


class PortfolioSnapshotSqlAdapter:
    def __init__(self, dbconnection: DbConnection):
        self._dbconnection = dbconnection
        self._snapshots_table = "portfolio_snapshots"

    def get_signals(self) -> pd.DataFrame:
        sql = f"SELECT * FROM {self._snapshots_table}"
        df = pd.read_sql_query(sql, self._dbconnection.engine)
        return df

    def insert_snapshot(self, portfolio_snapshot: PortfolioSnapshot) -> None:
        with psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self._snapshots_table} ("
                    "time_utc, "
                    "account_id, "
                    "shares, "
                    "bonds, "
                    "futures, "
                    "options, "
                    "currencies, "
                    "portfolio, "
                    "expected_yield)"
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ",
                    (portfolio_snapshot.time_utc,
                     portfolio_snapshot.account_id,
                     portfolio_snapshot.shares,
                     portfolio_snapshot.bonds,
                     portfolio_snapshot.futures,
                     portfolio_snapshot.options,
                     portfolio_snapshot.currencies,
                     portfolio_snapshot.portfolio,
                     portfolio_snapshot.expected_yield
                     ))
                conn.commit()
