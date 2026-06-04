# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from dataclasses import is_dataclass
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass", "validate_config"]


def _load_object(target: str) -> Any:
    module_name, _, attr_name = target.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid _target_ path: {target!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _convert_node(node: Any) -> Any:
    if isinstance(node, DictConfig):
        return omega_conf_to_dataclass(node)
    if isinstance(node, ListConfig):
        return [_convert_node(item) for item in node]
    if isinstance(node, list):
        return [_convert_node(item) for item in node]
    if isinstance(node, tuple):
        return tuple(_convert_node(item) for item in node)
    if isinstance(node, dict):
        if "_target_" in node:
            target_type = _load_object(node["_target_"])
            kwargs = {key: _convert_node(value) for key, value in node.items() if key != "_target_"}
            return target_type(**kwargs)
        return {key: _convert_node(value) for key, value in node.items()}
    return node


def omega_conf_to_dataclass(config: Any, dataclass_type: type | None = None) -> Any:
    """Convert an OmegaConf node into the configured dataclass instance.

    The repo's Hydra configs encode dataclass targets via `_target_`. This helper
    recursively instantiates those targets while preserving plain dict/list values.
    """

    if config is None:
        if dataclass_type is None:
            return None
        return dataclass_type()

    if isinstance(config, DictConfig):
        data = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, ListConfig):
        data = OmegaConf.to_container(config, resolve=True)
    else:
        data = config

    if dataclass_type is not None:
        if isinstance(data, dict) and "_target_" in data:
            target_type = _load_object(data["_target_"])
            if not issubclass(target_type, dataclass_type):
                raise TypeError(f"{target_type} is not a subclass of {dataclass_type}")
            dataclass_type = target_type

        if isinstance(data, dataclass_type):
            return data

        if isinstance(data, dict):
            kwargs = {key: _convert_node(value) for key, value in data.items() if key != "_target_"}
            return dataclass_type(**kwargs)

        if data == {}:
            return dataclass_type()

        raise TypeError(f"Cannot convert {type(data)} to {dataclass_type}")

    if isinstance(data, dict) and "_target_" in data:
        target_type = _load_object(data["_target_"])
        kwargs = {key: _convert_node(value) for key, value in data.items() if key != "_target_"}
        return target_type(**kwargs)

    if is_dataclass(data):
        return data

    return _convert_node(data)


def validate_config(config: Any, use_reference_policy: bool, use_critic: bool) -> None:
    """Run runtime validation hooks for the PPO config."""

    n_gpus = int(config.trainer.nnodes) * int(config.trainer.n_gpus_per_node)
    train_batch_size = int(config.data.train_batch_size)

    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    model_config = OmegaConf.to_container(config.actor_rollout_ref.model, resolve=True)
    actor_config.validate(n_gpus=n_gpus, train_batch_size=train_batch_size, model_config=model_config)

    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus=n_gpus, train_batch_size=train_batch_size)

    requires_reference = bool(actor_config.use_kl_loss or config.algorithm.use_kl_in_reward)
    if requires_reference and not use_reference_policy:
        raise ValueError(
            "Reference policy is required when `actor.use_kl_loss` or `algorithm.use_kl_in_reward` is enabled."
        )
