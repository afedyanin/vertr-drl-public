from datetime import datetime, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

from db_connection import DbConnection
from domain_model import Instrument, Interval
from moex_candles_sql_adapter import CandlesSqlAdapter

with DAG(
    'update_sber_candles',
    default_args={'retries': 2},
    description='Update history data from MOEX for SBER',
    schedule_interval='2/15 6-23 * * 1-5',  # Every 15 minutes, starting at 1 minutes past the hour
    start_date=datetime(2024, 10, 30, 0, 0, tzinfo=timezone.utc),
    catchup=False,
    tags=['msu', 'moex'],
) as dag:

    def update_candles(**kwargs):
        loader = CandlesSqlAdapter(
            dbconnection=kwargs['dbconnection'],
            instrument=kwargs['instrument'],
            interval=kwargs['interval'])

        last_known_time = loader.get_last_time_utc()

        if last_known_time is None:
            last_known_time = kwargs['start_date_utc']

        rows = loader.import_candles(start_date_utc=last_known_time)
        print(f'Updating candles for {kwargs["instrument"]} '
              f'interval={kwargs["interval"]} start_date_utc={last_known_time} ')
        print(f'{rows} rows loaded')
        return rows

    update_sber_day_1 = PythonOperator(
        task_id='update_sber_day_1',
        python_callable=update_candles,
        dag=dag,
        op_kwargs={
            'dbconnection': DbConnection.airflow_db_connection(),
            'instrument': Instrument.get_instrument("SBER"),
            'interval': Interval.day_1,
            'start_date_utc': None,
        },
    )

    update_sber_hour_1 = PythonOperator(
        task_id='update_sber_hour_1',
        python_callable=update_candles,
        dag=dag,
        op_kwargs={
            'dbconnection': DbConnection.airflow_db_connection(),
            'instrument': Instrument.get_instrument("SBER"),
            'interval': Interval.hour_1,
            'start_date_utc': None,
        },
    )

    start = DummyOperator(
        task_id='start',
        dag=dag,
    )

    start >> [
        update_sber_day_1,
        update_sber_hour_1
    ]
