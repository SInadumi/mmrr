import logging
import warnings
from collections.abc import Mapping
from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
import transformers.utils.logging as hf_logging
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from omegaconf import DictConfig, ListConfig, OmegaConf

from mmrr.callbacks import CohesionWriter
from mmrr.datamodule.multitask_datamodule import MTDataModule
from mmrr.datasets import CohesionDataset
from mmrr.modules import CohesionModule
from mmrr.utils.util import current_datetime_string
from utils import save_prediction, save_results

hf_logging.set_verbosity(hf_logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r"It is recommended to use .+ when logging on epoch level in distributed setting to accumulate the metric"
    r" across devices",
    category=PossibleUserWarning,
)
logging.getLogger("torch").setLevel(logging.WARNING)
OmegaConf.register_new_resolver(
    "now", current_datetime_string, replace=True, use_cache=True
)
OmegaConf.register_new_resolver("len", len, replace=True, use_cache=True)


@hydra.main(config_path="../configs/test", config_name="cohesion", version_base=None)
def main(eval_cfg: DictConfig):
    if isinstance(eval_cfg.devices, str):
        eval_cfg.devices = (
            list(map(int, eval_cfg.devices.split(",")))
            if "," in eval_cfg.devices
            else int(eval_cfg.devices)
        )
    if isinstance(eval_cfg.max_batches_per_device, str):
        eval_cfg.max_batches_per_device = int(eval_cfg.max_batches_per_device)
    if isinstance(eval_cfg.num_workers, str):
        eval_cfg.num_workers = int(eval_cfg.num_workers)

    # Load saved model and config
    model = CohesionModule.load_from_checkpoint(
        checkpoint_path=hydra.utils.to_absolute_path(eval_cfg.checkpoint)
    )
    if eval_cfg.compile is True:
        model = torch.compile(model)

    train_cfg: DictConfig = model.hparams
    OmegaConf.set_struct(train_cfg, False)  # enable to add new key-value pairs
    cfg = OmegaConf.merge(train_cfg, eval_cfg)
    assert isinstance(cfg, DictConfig)

    prediction_writer = CohesionWriter(
        flip_writer_reader_according_to_type_id=cfg.flip_reader_writer,
        analysis_target_threshold=cfg.analysis_target_threshold,
    )

    num_devices: int = (
        len(cfg.devices) if isinstance(cfg.devices, (list, ListConfig)) else cfg.devices
    )
    cfg.effective_batch_size = cfg.max_batches_per_device * num_devices
    cfg.datamodule.batch_size = cfg.max_batches_per_device

    # Instantiate lightning trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[prediction_writer],
        logger=False,
        devices=cfg.devices,
    )

    # Instantiate lightning datamodule
    datamodule = MTDataModule(cfg=cfg.datamodule)
    datamodule.setup(stage=TrainerFn.TESTING)

    datasets: dict[str, CohesionDataset]
    if cfg.eval_set == "valid":
        datasets = datamodule.val_datasets
        dataloaders = datamodule.val_dataloader()
    elif cfg.eval_set == "test":
        datasets = datamodule.test_datasets
        dataloaders = datamodule.test_dataloader()
    else:
        raise ValueError(f"datasets for eval set {cfg.eval_set} not found")

    # Run evaluation
    raw_results: list[Mapping[str, float]] = trainer.test(
        model=model, dataloaders=dataloaders
    )
    save_results(raw_results, Path(cfg.eval_dir))

    # Run prediction
    pred_dir = Path(cfg.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    for corpus, dataloader in dataloaders.items():
        prediction_writer.knp_destination = pred_dir / f"knp_{corpus}"
        prediction_writer.json_destination = pred_dir / f"json_{corpus}"
        trainer.predict(model=model, dataloaders=dataloader)
    save_prediction(datasets, pred_dir)


if __name__ == "__main__":
    main()
