"""
Single asset trading environment
Based on https://github.com/ClementPerroud/Gym-Trading-Env
"""
from random import Random

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from enum import IntEnum


class Actions(IntEnum):
    Hold = 0
    Sell = 1
    Buy = 2


def register_single_asset_trading_env(version=1):
    name = 'SingleAssetTrading'
    last_version = gym.envs.registration.find_highest_version(ns=None, name=name)
    env_id = gym.envs.registration.get_env_id(ns=None, name=name, version=version)
    if last_version is None:
        gym.envs.registration.register(
            id=env_id,
            entry_point='gym_env_single_asset:SingleAssetTrading',
            disable_env_checker=True
        )
        spec = gym.spec(env_id)
        print(f"Environment registered. {spec}")


class SingleAssetTrading(gym.Env):
    metadata = {'render_modes': ["ansi"]}

    def __init__(self,
                 df: pd.DataFrame,
                 max_episode_duration="max",
                 name="SingleAssetTrading",
                 render_mode="ansi",
                 ):

        self.name = name
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.max_episode_duration = max_episode_duration
        self._df = df.copy()
        self._df_length = len(self._df)
        self._features_columns = [col for col in self._df.columns if "feature" in col]
        self._obs_array = np.array(self._df[self._features_columns], dtype=np.float32)
        self.time_steps = self.get_time_steps(self._df, self.max_episode_duration)
        self._default_threshold = self.get_threshold(self._df)

        # spaces
        self.action_space = spaces.Discrete(len(Actions))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=[len(self._features_columns)],
            dtype=np.float32,
        )

        # step info declared in ctor
        self._idx = 0
        self._step = 0
        self._step_reward = 0
        self._total_reward = 0
        self._truncated = False
        self._rng = Random()
        self._action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._idx = 0
        self._step = 0
        self._step_reward = 0
        self._total_reward = 0
        self._truncated = False
        self._rng = Random(seed)
        self._action = None

        if isinstance(self.max_episode_duration, int):
            self._idx = self._rng.randint(
                self._idx,
                self._df_length - self.max_episode_duration
            )

        return self._get_obs(), self._get_info()

    def step(self, action):

        if self._truncated:
            raise ValueError("Truncated environment.")

        self._idx += 1
        self._step += 1
        self._action = action
        self._step_reward = self._get_reward()
        self._total_reward += self._step_reward
        self._truncated = self._is_truncated()

        return self._get_obs(), self._step_reward, False, self._truncated, self._get_info()

    def _is_truncated(self) -> bool:
        """
        Контроль границ таймфрейма
        """
        if self._idx >= self._df_length - 1:
            return True

        if (isinstance(self.max_episode_duration, int)
                and self._step >= self.max_episode_duration - 1):
            return True

        return False

    def _get_reward(self):
        """
        Вознаграждение на текущем шаге по выбранному действию
        """
        direction = self._get_direction(self._action)
        intraday_return = self._df.iloc[self._idx]["info_intraday_return"]
        expected_direction = self.get_expected_direction(
            intraday_return=intraday_return,
            threshold=self._default_threshold)

        if direction == 0:
            return 0
        if direction == expected_direction:
            return 1
        return -1

    def _get_obs(self):
        return self._obs_array[self._idx]

    def _get_info(self):
        row = self._df.iloc[self._idx]
        return dict(
            idx=self._idx,
            step=self._step,
            total_reward=self._total_reward,
            time_utc=self._df.index[self._idx],
            open_price=row["open"],
            close_price=row["close"],
            intraday_return=row["info_intraday_return"],
        )

    def render(self):
        if self.render_mode == "ansi":
            return self._render_frame()

    def _render_frame(self):
        return self._get_info()

    def close(self):
        pass

    @staticmethod
    def get_threshold(
            df: pd.DataFrame
    ) -> float:
        return df["info_intraday_return"].std() * 0.001

    @staticmethod
    def get_expected_direction(
            intraday_return: float,
            threshold: float = 0) -> int:
        action = SingleAssetTrading.get_expected_action(intraday_return, threshold)
        direction = SingleAssetTrading._get_direction(action)
        return direction

    @staticmethod
    def get_expected_action(
            intraday_return: float,
            threshold: float = 0) -> Actions:
        if threshold > 0 and abs(intraday_return) <= threshold:
            return Actions.Hold
        if intraday_return > 0:
            return Actions.Buy
        elif intraday_return < 0:
            return Actions.Sell
        return Actions.Hold

    @staticmethod
    def _get_direction(action: Actions) -> int:
        if action == Actions.Buy:
            return 1
        if action == Actions.Sell:
            return -1
        if action == Actions.Hold:
            return 0
        raise ValueError(f"Invalid action: {action}")

    @staticmethod
    def get_time_steps(df: pd.DataFrame, max_episode_duration="max", ) -> int:
        if isinstance(max_episode_duration, int):
            if max_episode_duration >= len(df):
                raise ValueError("max_episode_duration must be less than dataframe length.")
            return max_episode_duration
        else:
            return len(df)
