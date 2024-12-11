from datetime import datetime, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

from db_connection import DbConnection
from domain_model import Instrument, Interval
from tinvest_candles_sql_adapter import TinvestCandlesSqlAdapter
from tinvest_sandbox_adapter import TinvestSandboxAdapter, SB_TRADING_ACCOUNT_ID

with DAG(
    'tinvest_update_sber_candles',
    default_args={'retries': 2},
    description='Update history data from TINVEST for SBER',
    schedule_interval='1/15 6-23 * * 1-5',  # Every 15 minutes, starting at 1 minutes past the hour
    start_date=datetime(2024, 10, 30, 0, 0, tzinfo=timezone.utc),
    catchup=False,
    tags=['msu', 'tinvest'],
) as dag:

    def update_candles(**kwargs):
        account_id = kwargs['account_id']
        tinvest = TinvestSandboxAdapter(account_id)
        loader = TinvestCandlesSqlAdapter(
            dbconnection=kwargs['dbconnection'],
            instrument=kwargs['instrument'],
            interval=kwargs['interval'],
            tinvest_adapter=tinvest)

        last_known_time = loader.get_last_time_utc()

        if last_known_time is None:
            last_known_time = kwargs['start_date_utc']

        rows = loader.import_candles(start_date_utc=last_known_time)
        print(f'Updating candles for {kwargs["instrument"]} '
              f'interval={kwargs["interval"]} start_date_utc={last_known_time} ')
        print(f'{rows} rows loaded')
        return rows

    tinvest_update_sber_hour_1 = PythonOperator(
        task_id='tinvest_update_sber_hour_1',
        python_callable=update_candles,
        dag=dag,
        op_kwargs={
            'dbconnection': DbConnection.airflow_db_connection(),
            'instrument': Instrument.get_instrument("SBER"),
            'interval': Interval.hour_1,
            'start_date_utc': datetime(2022, 12, 1, 0, 0, tzinfo=timezone.utc),
            'account_id': SB_TRADING_ACCOUNT_ID,
            # 'account_id': SB_TESTING_ACCOUNT_ID,
        },
    )


    tinvest_update_sber_hour_1
