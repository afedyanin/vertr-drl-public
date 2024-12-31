"""
Набор гиперпараметров DLR алгоритмов
"""

from typing import Any, Dict

from rl_zoo3 import linear_schedule
from torch import nn

def dqn_params() -> Dict[str, Any]:
    return {
        "gamma": 0.9999,
        "learning_rate": 0.005,
        "batch_size": 512,
        "buffer_size": int(1e6),
        "target_update_interval": 20000,
        "learning_starts": 0,
        "train_freq": 1,
        "policy_kwargs": dict(
            net_arch=[128, 128],
            activation_fn=nn.ReLU,
        ),
    }

def trpo_params() -> Dict[str, Any]:
    return {
        "batch_size": 256,
        "n_steps": 256,
        "gamma": 0.9999,
        "learning_rate": 0.0005,
        "n_critic_updates": 20,
        "cg_max_steps": 10,
        "target_kl": 0.02,
        "gae_lambda": 0.8,
        "policy_kwargs": dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.ReLU,
            ortho_init=False,
        ),
    }

def ppo_params() -> Dict[str, Any]:
    return {
        "ent_coef": 0.00001,
        "vf_coef": 0.8,
        "max_grad_norm": 2,
        "clip_range": 0.8,
        "n_steps": 512,
        "n_epochs": 3,
        "gae_lambda": 0.7,
        "gamma": 0.9999,
        "learning_rate": 0.01,
        "batch_size": 512,
        "policy_kwargs": dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.LeakyReLU,
        ),
    }

def a2c_params() -> Dict[str, Any]:
    return {
        "max_grad_norm": 0.7,
        "use_rms_prop": True,
        "gae_lambda": 1.0,
        "n_steps": 2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "gamma": 0.9999,
        "normalize_advantage": False,
        "learning_rate": 0.0007,
        "policy_kwargs": dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.ReLU,
        ),
    }

def qrdqn_params() -> Dict[str, Any]:
    return {
        "gamma": 0.9999,
        "learning_rate": 0.005,
        "batch_size": 512,
        "buffer_size": int(1e6),
        "target_update_interval": 20000,
        "learning_starts": 0,
        "train_freq": 1,
        "policy_kwargs": dict(
            net_arch=[128, 128],
            activation_fn=nn.ReLU,
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
