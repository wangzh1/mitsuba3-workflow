from omegaconf import DictConfig
import rootutils
import hydra
from typing import Any, Dict, List, Optional, Tuple


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.trainer_base import MitsubaTrainer
from src import utils

log = utils.get_pylogger(__name__)


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: MitsubaTrainer = hydra.utils.instantiate(cfg.trainer)
    
    object_dict = {
        "cfg": cfg,
        "trainer": trainer,
    }

    utils.instantiate_loggers(cfg.get("logger"), object_dict=object_dict)


    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit()

    # train_metrics = trainer.callback_metrics

    # if cfg.get("test"):
    #     log.info("Starting testing!")
    #     ckpt_path = trainer.checkpoint_callback.best_model_path
    #     if ckpt_path == "":
    #         log.warning(
    #             "Best ckpt not found! Using current weights for testing...")
    #         ckpt_path = None
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #     log.info(f"Best ckpt path: {ckpt_path}")

    # test_metrics = trainer.callback_metrics

    # merge train and test metrics
    # metric_dict = {**train_metrics, **test_metrics}

    # return metric_dict, object_dict
    return 


@hydra.main(version_base="1.3", config_path="./configs", config_name="fit_image.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    utils.extras(cfg)
    train(cfg)
    return

if __name__ == "__main__":
    main()
