from datetime import datetime, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

from portfolio_snapshot_sql_adapter import PortfolioSnapshotSqlAdapter
from position_snapshot_sql_adapter import PositionSnapshotSqlAdapter
from db_connection import DbConnection
from tinvest_sandbox_adapter import SB_TRADING_ACCOUNT_ID, TinvestSandboxAdapter, SB_TESTING_ACCOUNT_ID


# https://crontab.cronhub.io/
with DAG(
    'tinvest_make_snapshots',
    default_args={'retries': 2},
    description='Make portfolio and positions snapshots via T-invest API',
    schedule_interval='5/5 6-23 * * 1-5', # Every 5 minutes, starting at 4 minutes past the hour
    start_date=datetime(2024, 11, 30, 15, 0, tzinfo=timezone.utc),
    catchup=False,
    tags=['msu', 'tinvest'],
    max_active_tasks=1,
) as dag:

    def make_snapshots(**kwargs):
        db_connection = kwargs['db_connection']
        account_id = kwargs['account_id']

        tinvest_adapter = TinvestSandboxAdapter(account_id)

        portfolio_adapter = PortfolioSnapshotSqlAdapter(db_connection)
        portfolio = tinvest_adapter.get_portfolio_snapshot()
        portfolio_adapter.insert_snapshot(portfolio)
        print(f'New portfolio snapshot created.')

        positions_adapter = PositionSnapshotSqlAdapter(db_connection)
        positions = tinvest_adapter.get_positions_snapshot()
        positions_adapter.insert_snapshots(positions)
        print(f'New positions snapshot created.')

    make_snapshots_4min_of5 = PythonOperator(
        task_id='make_portfolio_snapshots_4min_of5',
        python_callable=make_snapshots,
        dag=dag,
        op_kwargs={
            'db_connection': DbConnection.airflow_db_connection(),
            'account_id': SB_TRADING_ACCOUNT_ID,

        },
    )

    make_snapshots_4min_of5
