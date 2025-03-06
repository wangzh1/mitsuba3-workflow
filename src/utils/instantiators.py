from typing import Any, Dict, List

import hydra
import wandb
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def instantiate_loggers(logger_cfg: DictConfig, object_dict: Dict[str, Any]) -> None:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig):
            lg_conf = dict(lg_conf)
            wandb.init(**lg_conf, config=object_dict)
            log.info(f"Instantiating wandb logger for project <{lg_conf.project}>, name <{lg_conf.name}>")
        else:
            raise RuntimeError("Cannot init wandb logger!")
    return 
