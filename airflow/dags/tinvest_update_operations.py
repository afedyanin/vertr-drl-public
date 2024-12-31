from datetime import datetime, timezone, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from db_connection import DbConnection
from domain_model import Instrument
from operations_sql_adapter import OperationsSqlAdapter
from tinvest_sandbox_adapter import SB_TRADING_ACCOUNT_ID, TinvestSandboxAdapter

# https://crontab.cronhub.io/
with DAG(
    'tinvest_update_operations',
    default_args={'retries': 2},
    description='Update history of operations and trades from T-invest account',
    schedule_interval='3/10 6-23 * * 5',  # Every 15 minutes, starting at 3 minutes past the hour
    start_date=datetime(2024, 11, 29, 18, 0, tzinfo=timezone.utc),
    catchup=False,
    tags=['msu', 'tinvest'],
    max_active_tasks=1
) as dag:

    def update_operations(**kwargs):
        db_connection = kwargs['db_connection']
        account_id = kwargs['account_id']
        date_from = kwargs['date_from']
        date_to = kwargs['date_to']
        instrument = kwargs['instrument']

        tinvest_adapter = TinvestSandboxAdapter(account_id)
        ops_adapter = OperationsSqlAdapter(db_connection)

        ops = tinvest_adapter.get_operations(
            date_from=date_from,
            date_to=date_to,
            instrument_figi=instrument.figi)

        ops_adapter.insert_operations(account_id, ops)
        print('Updating operations completed.')

    update_operations_3min_of10 = PythonOperator(
        task_id='update_operations_3min_of10',
        python_callable=update_operations,
        dag=dag,
        op_kwargs={
            'db_connection': DbConnection.airflow_db_connection(),
            'date_from': datetime.now() - timedelta(hours=1),
            'date_to': datetime.now(),
            'instrument': Instrument.get_instrument("SBER"),
            'account_id': SB_TRADING_ACCOUNT_ID,
            #'account_id': SB_TESTING_ACCOUNT_ID,
        },
    )

    update_operations_3min_of10
