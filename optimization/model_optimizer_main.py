import sys
from datetime import datetime, timezone, timedelta
import torch as th

from optimization.model_optimizer import ModelOptimizer

sys.path.append("../airflow/plugins")

from gym_env_factory import GymEnvFactory
from db_connection import DbConnection
from domain_model import Instrument, Interval
from gym_env_single_asset import register_single_asset_trading_env
from moex_candles_sql_adapter import CandlesSqlAdapter

if __name__ == '__main__':
    algo = sys.argv[1]
    if algo is None:
        print(f"DRL algorithm as first argument must be specified.")
        print(f"Supported algorithms: a2c, ddpg, dqn, ppo, sac, td3, ars, qrdqn, tqc, trpo, ppo_lstm.")
        exit()

    print(f"Registering gym env...")
    register_single_asset_trading_env(1)

    db_connection = DbConnection.local_db_connection()
    instrument = Instrument.get_instrument("SBER")
    interval = Interval.hour_1
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    sql_adapter = CandlesSqlAdapter(db_connection, interval, instrument)
    env_factory = GymEnvFactory(sql_adapter)
    study_name = f"{algo}_sber_hour1_v1"
    optuna_connection = DbConnection.local_db_optuna_connection().get_storage_url()

    train_episodes = 50
    train_episode_duration = 1000
    train_end_time_utc = datetime(2024, 10, 31, tzinfo=timezone.utc)
    train_start_time_utc = train_end_time_utc - timedelta(days=365*2)

    eval_episodes = 4
    eval_episode_duration = 100

    eval_end_time_utc = datetime(2024, 12, 5, tzinfo=timezone.utc)
    eval_start_time_utc = eval_end_time_utc - timedelta(days=30)

    basic_hyperparams = {
        "policy": "MlpPolicy",
    }

    optimizer = ModelOptimizer(
        env_factory=env_factory,
        algo=algo,
        train_start_time_utc=train_start_time_utc,
        train_end_time_utc=train_end_time_utc,
        eval_start_time_utc=eval_start_time_utc,
        eval_end_time_utc=eval_end_time_utc,
        train_episode_duration=train_episode_duration,
        eval_episode_duration=eval_episode_duration,
        storage=optuna_connection,
        study_name=study_name,
        default_hyperparams=basic_hyperparams,
        verbose=0,
        n_jobs=1,
        device=device,
        n_startup_episodes=2,
        n_trials=64,
        n_episodes=train_episodes,
        n_eval_episodes=eval_episodes,
        n_startup_trials=3)

    optimizer.optimize()
    print("Optimization finished.")
