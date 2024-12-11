import psycopg
import pandas as pd

from db_connection import DbConnection
from domain_model import PositionSnapshot


class PositionSnapshotSqlAdapter:
    def __init__(self, dbconnection: DbConnection):
        self._dbconnection = dbconnection
        self._snapshots_table = "position_snapshots"

    def get_signals(self) -> pd.DataFrame:
        sql = f"SELECT * FROM {self._snapshots_table}"
        df = pd.read_sql_query(sql, self._dbconnection.engine)
        return df

    def insert_snapshots(self, position_snapshots: list[PositionSnapshot]) -> None:
        with psycopg.connect(
                dbname=self._dbconnection.dbname,
                user=self._dbconnection.user,
                password=self._dbconnection.password,
                host=self._dbconnection.host,
                port=self._dbconnection.port) as conn:
            with conn.cursor() as cur:
                for snapshot in position_snapshots:
                    cur.execute(
                        f"INSERT INTO {self._snapshots_table} ("
                        "time_utc, "
                        "account_id, "
                        "instrument_uid, "
                        "instrument_str, "
                        "balance)"
                        "VALUES (%s, %s, %s, %s, %s) ",
                        (snapshot.time_utc,
                         snapshot.account_id,
                         snapshot.instrument_uid,
                         snapshot.instrument_str,
                         snapshot.balance
                         ))
                conn.commit()
