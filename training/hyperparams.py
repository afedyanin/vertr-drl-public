"""
Набор гиперпараметров DLR алгоритмов
"""

from typing import Any, Dict

from rl_zoo3 import linear_schedule
from torch import nn


def ppo_params() -> Dict[str, Any]:
    return {
        "batch_size": 32,
        "n_steps": 16,
        "gamma": 0.9999,
        "learning_rate": 0.0005602358707524044,
        "ent_coef": 0.08405937469314628,
        "clip_range": 0.3,
        "n_epochs": 1,
        "gae_lambda": 0.9,
        "max_grad_norm": 1,
        "vf_coef": 0.9228222128896565,
        "policy_kwargs": dict(
            net_arch=[64],
            activation_fn=nn.Tanh,
            ortho_init=False,
        ),
    }


def ppo_lstm_params() -> Dict[str, Any]:
    return {
        "n_steps": 512,
        "batch_size": 8,
        "gamma": 0.999,
        "learning_rate": 0.07108302214619301,
        "ent_coef": 1.961839490282344e-08,
        "clip_range": 0.1,
        "n_epochs": 1,
        "gae_lambda": 0.98,
        "max_grad_norm": 0.9,
        "vf_coef": 0.5204561298067211,
        "policy_kwargs": dict(
            net_arch=[256, 256],
            activation_fn=nn.ReLU,
            ortho_init=False,
            enable_critic_lstm=True,
            lstm_hidden_size=16,
        ),
    }


def trpo_params() -> Dict[str, Any]:
    return {
        "batch_size": 16,
        "n_steps": 32,
        "gamma": 0.9999,
        "learning_rate": 0.0003750702770476366,
        "n_critic_updates": 5,
        "cg_max_steps": 25,
        "target_kl": 0.03,
        "gae_lambda": 1.0,
        "policy_kwargs": dict(
            net_arch=[256, 256],
            activation_fn=nn.Tanh,
            ortho_init=False,
        ),
    }


def a2c_params() -> Dict[str, Any]:
    return {
        "gamma": 0.995,
        "normalize_advantage": False,
        "max_grad_norm": 0.9,
        "use_rms_prop": True,
        "gae_lambda": 0.8,
        "n_steps": 1024,
        "learning_rate": 0.000193150020468329,
        "ent_coef": 1.7065880108173808e-07,
        "vf_coef": 0.8556005755855345,
        "policy_kwargs": dict(
            net_arch=[64],
            activation_fn=nn.Tanh,
            ortho_init=False,
            ##lr_schedule=linear_schedule,
        ),
    }

def dqn_params() -> Dict[str, Any]:
    return {
        "gamma": 0.995,
        "learning_rate": 0.0003416020887251245,
        "batch_size": 16,
        "buffer_size": 10000,
        "exploration_final_eps": 0.08886619376104432,
        "exploration_fraction": 0.05849817136891532,
        "target_update_interval": 15000,
        "learning_starts": 1000,
        "train_freq": 1,
        "policy_kwargs": dict(
            net_arch=[64],
        ),
    }


def qrdqn_params() -> Dict[str, Any]:
    return {
        "gamma": 0.95,
        "learning_rate": 0.0001722920955939197,
        "batch_size": 32,
        "buffer_size": 100000,
        "exploration_final_eps": 0.060543395017399214,
        "exploration_fraction": 0.13002018039159535,
        "target_update_interval": 10000,
        "learning_starts": 0,
        "train_freq": 8,
        #"subsample_steps": 4,
        "policy_kwargs": dict(
            net_arch=[64],
            n_quantiles=175,
        ),
    }


def ars_params() -> Dict[str, Any]:
    return {
        "n_delta": 8,
        "learning_rate": 4.2817369808620575e-05,
        "delta_std": 0.025,
        ##"top_frac_size": 0.9,
        "zero_policy": False,
    }


HYPERPARAMS_OPTIMIZED = {
    "a2c": a2c_params,
    "ars": ars_params,
    "ddpg": None,
    "dqn": dqn_params,
    "qrdqn": qrdqn_params,
    "sac": None,
    "tqc": None,
    "ppo": ppo_params,
    "ppo_lstm": ppo_lstm_params,
    "td3": None,
    "trpo": trpo_params,
}