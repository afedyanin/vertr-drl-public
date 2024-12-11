import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Type

from sb3_contrib import ARS, QRDQN, TQC, TRPO, RecurrentPPO
from stable_baselines3 import DQN, A2C, DDPG, PPO, SAC, TD3

from airflow import DAG
from airflow.operators.python import PythonOperator
from stable_baselines3.common.base_class import BaseAlgorithm

from candles_tinvest_adapter import CandlesTinvestAdapter
from gym_env_single_asset import register_single_asset_trading_env
from domain_model import Interval, Instrument, Signal
from signals_sql_adapter import SignalsSqlAdapter
from db_connection import DbConnection
from strategy_predictor import StrategyPredictor
from strategy_sb3 import StrategySb3

ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    # SB3 Contrib,
    "ars": ARS,
    "qrdqn": QRDQN,
    "tqc": TQC,
    "trpo": TRPO,
    "ppo_lstm": RecurrentPPO,
}


# https://crontab.cronhub.io/
with DAG(
    'tinvest_generate_signals',
    default_args={'retries': 2},
    description='Get last signal from DB and post order to T-invest API',
    schedule_interval='3/15 6-23 * * 1-5', # Every 15 minutes, starting at 1 minutes past the hour
    start_date=datetime(2024, 11, 30, 15, 0, tzinfo=timezone.utc),
    catchup=False,
    tags=['msu', 'tinvest'],
    max_active_tasks=1,
) as dag:

    def generate_signal(**kwargs):
        db_connection = kwargs['db_connection']
        algo_name = kwargs['algo_name']
        interval = kwargs['interval']
        instrument = kwargs['instrument']

        register_single_asset_trading_env(1)

        # TODO: optimize it
        prediction = get_prediction(algo_name, instrument, interval)
        prediction_time = prediction['time_utc']

        signals_adapter = SignalsSqlAdapter(db_connection)
        last_signal = signals_adapter.get_last_signal()

        if last_signal is not None and last_signal.time_utc >= prediction_time:
            print(f'Signal already exists: last_signal={last_signal.time_utc} >= prediction_time={prediction_time}')
            print(f'Skip generating signal.')
            return

        signal = Signal(
            order_request_id=uuid.uuid4(),
            time_utc=prediction_time,
            quantity=get_quantity(prediction),
            symbol=instrument.symbol,
            origin=prediction['origin'],
            prediction=json.dumps(
                prediction,
                ensure_ascii=False,
                indent=4,
                sort_keys=False,
                default=str)
        )

        signals_adapter.insert_signal(signal)
        print(f'New signal generated: order_request_id={signal.order_request_id}.')

    def get_quantity(prediction: dict) -> int:
        action = prediction['action']
        if action == 2: # Buy
            return 1
        if action == 1: # Sell
            return -1
        return 0 # Hold

    def get_prediction(
            algo_name: str,
            instrument: Instrument,
            interval: Interval,
    ):
        candles_adapter = CandlesTinvestAdapter(interval=interval, instrument=instrument)
        model_path = f"/opt/airflow/dags/models/{algo_name}_model"
        model = ALGOS[algo_name].load(model_path)
        strategy = StrategySb3(model)
        predictor = StrategyPredictor(data_adapter=candles_adapter, strategy=strategy)
        start_date_utc, end_date_utc = get_date_range()
        prediction = predictor.predict(start_date_utc, end_date_utc)
        origin = f'{strategy.__class__.__name__}:{model.__class__.__name__}'
        prediction.update({'origin': origin})
        return prediction

    def get_date_range() -> (datetime, datetime):
        end_date_utc = datetime.now(timezone.utc)
        start_date_utc = end_date_utc - timedelta(days=5)
        return start_date_utc, end_date_utc

    generate_signals_1min_of15 = PythonOperator(
        task_id='generate_signals_1min_of15',
        python_callable=generate_signal,
        dag=dag,
        op_kwargs={
            'db_connection': DbConnection.airflow_db_connection(),
            'algo_name': 'dqn',
            'instrument': Instrument.get_instrument("SBER"),
            'interval': Interval.hour_1,
        },
    )

    generate_signals_1min_of15
