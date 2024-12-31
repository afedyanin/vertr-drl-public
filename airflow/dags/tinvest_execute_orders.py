from datetime import datetime, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

from domain_model import Instrument
from db_connection import DbConnection
from tinvest_risk_engine import TinvestRiskEngine
from tinvest_sandbox_adapter import SB_TRADING_ACCOUNT_ID

# https://crontab.cronhub.io/
with DAG(
    'tinvest_execute_orders',
    default_args={'retries': 2},
    description='Get last signal from DB and post order to T-invest API',
    schedule_interval='2/10 6-23 * * 5',  # Every 15 minutes, starting at 1 minutes past the hour
    start_date=datetime(2024, 11, 29, 18, 0, tzinfo=timezone.utc),
    catchup=False,
    tags=['msu', 'tinvest'],
    max_active_tasks=1
) as dag:

    def post_order(**kwargs):
        db_connection = kwargs['db_connection']
        instrument = kwargs['instrument']
        account_id = kwargs['account_id']
        quantity_lots = kwargs['quantity_lots']
        risk_engine = TinvestRiskEngine(db_connection, instrument, account_id)
        risk_engine.post_order(quantity_lots)

    execute_orders_2min_of10 = PythonOperator(
        task_id='execute_orders_2min_of10',
        python_callable=post_order,
        dag=dag,
        op_kwargs={
            'db_connection': DbConnection.airflow_db_connection(),
            'instrument': Instrument.get_instrument("SBER"),
            'account_id': SB_TRADING_ACCOUNT_ID,
            'quantity_lots': 10,
        },
    )

    execute_orders_2min_of10
