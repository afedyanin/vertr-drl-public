from typing import Any, Dict

from stable_baselines3.common.base_class import BaseAlgorithm

from gym_env_single_asset import Actions
from strategy_basic import Strategy


class StrategySb3(Strategy):
    def __init__(self, model: BaseAlgorithm):
        super().__init__()
        self._model = model

    def get_action(
            self,
            observation: Any | None = None,
            info: dict[str, Any] | None = None) -> Actions:
        action, _ = self._model.predict(observation=observation, deterministic=True)
        return action.item()
