from datetime import datetime

import gymnasium as gym

from typing import Dict, Any

from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from feature_composer import FeatureComposer
from gym_env_single_asset import SingleAssetTrading
from synthetic_data_adapter import DataAdapter


class GymEnvFactory:
    def __init__(
            self,
            data_adapter: DataAdapter,
            env_id: str = "SingleAssetTrading-v1",
            seed: int | None = None,
    ):
        self._data_adapter = data_adapter
        self.env_name = EnvironmentName(env_id)
        self.seed = seed

    def create_vector_env(
            self,
            start_time_utc: datetime,
            end_time_utc: datetime,
            episode_duration: int | str = "max") -> (VecEnv, int):

        env_kwargs, episode_steps = self._compose_kwargs(
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
            episode_duration=episode_duration)

        env = make_vec_env(
            self.env_name.gym_id,
            n_envs=1,
            seed=self.seed,
            env_kwargs=env_kwargs,
            monitor_dir=None)

        return env, episode_steps

    def create_env(
            self,
            start_time_utc: datetime,
            end_time_utc: datetime,
            episode_duration: int | str = "max") -> (gym.Env, int):

        env_kwargs, episode_steps = self._compose_kwargs(
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
            episode_duration=episode_duration)

        spec = gym.spec(self.env_name.gym_id)
        env = spec.make(**env_kwargs)

        return env, episode_steps

    def _compose_kwargs(
            self,
            start_time_utc: datetime,
            end_time_utc: datetime,
            episode_duration: int | str = "max") -> (Dict[str, Any], int):

        composer = FeatureComposer(data_adapter=self._data_adapter)
        df = composer.compose(start_time_utc=start_time_utc, end_time_utc=end_time_utc)

        env_kwargs = {
            'df': df,
            'max_episode_duration': episode_duration,
            'render_mode': 'ansi'
        }

        episode_steps = SingleAssetTrading.get_time_steps(df, episode_duration)
        return env_kwargs, episode_steps
