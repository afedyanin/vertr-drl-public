from datetime import datetime

import pandas as pd

from feature_composer import FeatureComposer
from gym_env_factory import GymEnvFactory
from synthetic_data_adapter import DataAdapter

from strategy_basic import Strategy
from gym_env_single_asset import Actions, SingleAssetTrading


class StrategyPredictor:
    def __init__(
            self,
            data_adapter: DataAdapter,
            strategy: Strategy):
        self.data_adapter = data_adapter
        self.strategy = strategy

    def predict(self,
                start_time_utc: datetime,
                end_time_utc: datetime) -> dict:

        env_factory = GymEnvFactory(self.data_adapter)
        env, total_steps = env_factory.create_env(start_time_utc, end_time_utc)
        observation, info = env.reset()

        is_done = False
        while not is_done:
            action = self.strategy.get_action(observation=observation, info=info)
            observation, reward, terminated, truncated, info = env.step(action)
            is_done = terminated or truncated

        # выход из цикла вернул данные по последней свече
        # используем их для реального предсказания
        action = self.strategy.get_action(observation=observation, info=info)
        info.update({'action': action})
        return info

    def evaluate(
            self,
            start_time_utc: datetime,
            end_time_utc: datetime) -> pd.DataFrame:

        env_factory = GymEnvFactory(self.data_adapter)
        env, total_steps = env_factory.create_env(start_time_utc, end_time_utc)
        feature_composer = FeatureComposer(self.data_adapter, fill_missing_values=False)
        candles = feature_composer.compose(start_time_utc, end_time_utc)
        threshold = SingleAssetTrading.get_threshold(candles)

        time_index = []
        predicted_actions = []
        expected_actions = []
        rewards = []

        observation, info = env.reset()
        action = self.strategy.get_action(observation=observation, info=info)
        #print(f'env.reset: time: {info['time_utc']}, action: {action}')

        time_index.append(info["time_utc"])
        predicted_actions.append(action) # тут действие для следующего шага
        expected_actions.append(Actions.Hold)
        rewards.append(0)

        is_done = False
        count = 1

        while not is_done:
            action = self.strategy.get_action(observation=observation, info=info)
            #print(f'Before env.step. По данным из шага: {info['time_utc']}, предсказание: {action}')
            observation, reward, terminated, truncated, info = env.step(action)
            #print(f'After env.step. Предсказание {action} запишется в шаг {info['time_utc']}.')

            time_index.append(info["time_utc"])
            predicted_actions.append(action)
            expected_action = SingleAssetTrading.get_expected_action(info["intraday_return"], threshold)
            expected_actions.append(expected_action)
            rewards.append(reward)

            is_done = terminated or truncated
            count += 1

        #action = self.strategy.get_action(observation=observation, info=info)
        #print(f'Last env.step. По данным из шага: {info['time_utc']}, предсказание: {action}')

        df_pred = pd.DataFrame(
            index=time_index,
            data={
                "action_pred": predicted_actions,
                "action_actual": expected_actions,
                "reward": rewards})

        df = candles.join(df_pred, how='left', sort=True)[[
            "open",
            "close",
            "info_intraday_return",
            "action_pred",
            "action_actual",
            "reward"]]

        return df

