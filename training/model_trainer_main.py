import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import torch as th

sys.path.append("../airflow/plugins")

from airflow.plugins.gym_env_factory import GymEnvFactory
from airflow.plugins.db_connection import DbConnection
from airflow.plugins.domain_model import Instrument, Interval
from airflow.plugins.gym_env_single_asset import register_single_asset_trading_env
from airflow.plugins.synthetic_data_adapter import SyntheticDataAdapter
from airflow.plugins.moex_candles_sql_adapter import CandlesSqlAdapter

from training.model_trainer import ModelTrainer

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

    trainer = ModelTrainer(
        env_factory=env_factory,
        algo=algo,
        verbose=0,
        device=device,
        seed=None,
        log_dir="logs",
    )

    train_end_time_utc = datetime(2024, 10, 31, tzinfo=timezone.utc)
    train_start_time_utc = train_end_time_utc - timedelta(days=365*2)

    basic_hyperparams = {
        "policy": "MlpPolicy",
    }

    print(f"Start training algorithm: {algo}")
    model = trainer.train(
        start_time_utc=train_start_time_utc,
        end_time_utc=train_end_time_utc,
        episode_duration=1000,
        episodes=100,
        hyperparams=basic_hyperparams,
        optimized=False)
    print("Training completed.")

    return_episode_rewards = False
    eval_episodes = 5

    eval_end_time_utc = datetime(2024, 12, 5, tzinfo=timezone.utc)
    eval_start_time_utc = eval_end_time_utc - timedelta(days=30)

    rewards, steps = trainer.evaluate(
        start_time_utc=eval_start_time_utc,
        end_time_utc=eval_end_time_utc,
        model=model,
        episode_duration=100,
        episodes=eval_episodes,
        return_episode_rewards=return_episode_rewards)

    if return_episode_rewards:
        print(f"{algo} evaluation: rewards={rewards} steps={steps}")
    else:
        print(f"{algo} reward evaluation by {eval_episodes} episodes: mean={rewards} std={steps}")

    model_name = f"{algo}_model"
    print(f"Saving model to '{model_name}.zip'")
    model.save(model_name)
