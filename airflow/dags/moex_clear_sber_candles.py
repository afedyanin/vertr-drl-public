from datetime import datetime, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

from db_connection import DbConnection
from domain_model import Instrument, Interval
from moex_candles_sql_adapter import CandlesSqlAdapter

with DAG(
    'clear_sber_candles',
    default_args={'retries': 2},
    description='Clear SBER candles',
    schedule_interval=None,
    start_date=datetime(2024, 10, 27, tzinfo=timezone.utc),
    catchup=False,
    tags=['msu', 'moex'],
) as dag:

    def clear_data(**kwargs):
        loader = CandlesSqlAdapter(
            dbconnection=kwargs['dbconnection'],
            instrument=kwargs['instrument'],
            interval=kwargs['interval'])
        loader.clear_candles()
        print(f'Clearing all history data for {kwargs["instrument"]} interval={kwargs["interval"]} ')

    clear_sber_day_1 = PythonOperator(
        task_id='clear_sber_day_1',
        python_callable=clear_data,
        dag=dag,
        op_kwargs={
            'dbconnection': DbConnection.airflow_db_connection(),
            'instrument': Instrument.get_instrument("SBER"),
            'interval': Interval.day_1,
        },
    )

    clear_sber_hour_1 = PythonOperator(
        task_id='clear_sber_hour_1',
        python_callable=clear_data,
        dag=dag,
        op_kwargs={
            'dbconnection': DbConnection.airflow_db_connection(),
            'instrument': Instrument.get_instrument("SBER"),
            'interval': Interval.hour_1,
        },
    )

    start = DummyOperator(
        task_id='start',
        dag=dag,
    )

    start >> [
        clear_sber_day_1,
        clear_sber_hour_1,
    ]
