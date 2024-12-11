"""
Оптимизатор гиперпараметров для DRL алгоритмов SB3 и SB3-contrib.
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/exp_manager.py
"""
import sys
import warnings
from datetime import datetime
from pprint import pprint
from typing import Any, Type, List
from typing import Dict
import torch as th
import optuna

from rl_zoo3.callbacks import TrialEvalCallback
from rl_zoo3.utils import get_callback_list
from sb3_contrib import ARS, QRDQN, TQC, TRPO, RecurrentPPO
from stable_baselines3 import HerReplayBuffer, A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from optimization.pruner_factory import PrunerFactory
from optimization.sampler_factory import SamplerFactory

sys.path.append("../airflow/plugins")

from optimization import hyperparams as hp
from gym_env_factory import GymEnvFactory

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


class ModelOptimizer:
    def __init__(
            self,
            env_factory: GymEnvFactory,
            algo: str,
            train_start_time_utc: datetime,
            train_end_time_utc: datetime,
            eval_start_time_utc: datetime,
            eval_end_time_utc: datetime,
            train_episode_duration: int | str = "max",
            eval_episode_duration: int | str = "max",
            storage: str | None = None,
            study_name: str | None = None,
            default_hyperparams: Dict[str, Any] | None = None,
            verbose: int = 1,
            n_jobs: int = 4,
            sampler: str = "tpe",
            pruner: str = "halving",
            seed: int | None = None,
            device: th.device | str = "auto",
            n_startup_episodes: int = 10,
            n_trials: int = 64,
            n_episodes: int = 100,
            n_eval_episodes: int = 3,
            n_startup_trials: int = 2
    ):
        self.algo = algo
        self.train_start_time_utc = train_start_time_utc
        self.train_end_time_utc = train_end_time_utc
        self.eval_start_time_utc = eval_start_time_utc
        self.eval_end_time_utc = eval_end_time_utc
        self.train_episode_duration = train_episode_duration
        self.eval_episode_duration = eval_episode_duration
        self._hyperparams = default_hyperparams
        self.storage = storage
        self.study_name = study_name
        self.verbose = verbose
        self.sampler = sampler
        self.pruner = pruner
        self.n_jobs = n_jobs
        self.device = device
        self.env_factory = env_factory
        self.seed = seed
        self.n_startup_episodes = n_startup_episodes
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.n_episodes = n_episodes
        self.n_eval_episodes = n_eval_episodes

        # Callbacks
        self.specified_callbacks: List = []

    def objective(self, trial: optuna.Trial) -> float:
        kwargs = self._hyperparams.copy()

        additional_args = {
            "using_her_replay_buffer": kwargs.get("replay_buffer_class") == HerReplayBuffer,
            "her_kwargs": kwargs.get("replay_buffer_kwargs", {}),
        }

        sampled_hyperparams = hp.HYPERPARAMS_SAMPLER[self.algo](
            trial=trial,
            n_actions=0,
            n_envs=1,
            additional_args=additional_args)

        kwargs.update(sampled_hyperparams)

        train_env, train_episode_steps = self.env_factory.create_vector_env(
            self.train_start_time_utc,
            self.train_end_time_utc,
            self.train_episode_duration)

        eval_env, eval_episode_steps = self.env_factory.create_vector_env(
            self.eval_start_time_utc,
            self.eval_end_time_utc,
            self.eval_episode_duration)

        trial_verbosity = 0
        if self.verbose >= 2:
            trial_verbosity = self.verbose

        model = ALGOS[self.algo](
            env=train_env,
            tensorboard_log=None,
            seed=None,
            verbose=trial_verbosity,
            device=self.device,
            **kwargs,
        )

        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            eval_freq=train_episode_steps, ## через какое количество шагов обучения вызываем евал
            n_eval_episodes=self.n_eval_episodes, ## используется для оценки медианного реварда
            deterministic=True,
        )

        learn_kwargs = {}
        time_steps = train_episode_steps * self.n_episodes

        try:
            model.learn(time_steps, callback=eval_callback, **learn_kwargs)
            assert model.env is not None
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            assert model.env is not None
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned() from e
        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    def optimize(self) -> None:
        if self.verbose > 0:
            print("Optimizing hyperparameters")

        if self.storage is not None and self.study_name is None:
            warnings.warn(
                f"You passed a remote storage: {self.storage} but no `--study-name`."
                "The study name will be generated by Optuna, make sure to re-use the same study name "
                "when you want to do distributed hyperparameter optimization."
            )

        sampler = SamplerFactory.create(
            n_startup_trials=self.n_startup_trials,
            seed=self.seed,
            sampler_method=self.sampler)

        pruner = PrunerFactory.create(
            n_startup_trials=self.n_startup_trials,
            n_evaluations=self.n_eval_episodes,
            pruner_method=self.pruner)

        if self.verbose > 0:
            print(f"Sampler: {self.sampler} - Pruner: {self.pruner}")

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True,
            direction="maximize",
        )

        try:
            study.optimize(self.objective, n_jobs=self.n_jobs, n_trials=self.n_trials)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
