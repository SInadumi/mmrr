import logging
from pathlib import Path
from typing import TextIO, Union

import hydra
import lightning.pytorch as pl
import torch
import transformers.utils.logging as hf_logging
from callbacks import MMRefWriter
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from cl_mmref.datamodule.multitask_datamodule import MTDataModule
from cl_mmref.modules import MMRefModule
from cl_mmref.utils.util import current_datetime_string

hf_logging.set_verbosity(hf_logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

OmegaConf.register_new_resolver(
    "now", current_datetime_string, replace=True, use_cache=True
)
OmegaConf.register_new_resolver("len", len, replace=True, use_cache=True)


class Analyzer:
    def __init__(self, cfg: DictConfig) -> None:
        # Load saved model and config
        self.device = self._prepare_device(device_name="auto")
        self.model = MMRefModule.load_from_checkpoint(
            checkpoint_path=hydra.utils.to_absolute_path(cfg.checkpoint),
            map_location=self.device,
        )
        cfg_train: DictConfig = self.model.hparams
        OmegaConf.set_struct(cfg_train, False)  # enable to add new key-value pairs
        self.cfg = OmegaConf.merge(cfg_train, cfg)
        assert isinstance(self.cfg, DictConfig)

        callbacks: list[Callback] = list(
            map(hydra.utils.instantiate, self.cfg.get("callbacks", {}).values())
        )
        self.prediction_writer = MMRefWriter()

        # Instantiate lightning trainer
        self.trainer: pl.Trainer = hydra.utils.instantiate(
            self.cfg.trainer,
            callbacks=[*callbacks, self.prediction_writer],
            logger=False,
            devices=self.cfg.devices,
        )

    @staticmethod
    def _prepare_device(device_name: str) -> torch.device:
        n_gpu = torch.cuda.device_count()
        if device_name == "auto":
            if n_gpu > 0:
                device_name = "gpu"
            else:
                device_name = "cpu"
        if device_name == "gpu" and n_gpu == 0:
            logger.warning(
                "There's no GPU available on this machine. Using CPU instead."
            )
            return torch.device("cpu")
        else:
            return torch.device("cuda:0" if device_name == "gpu" else "cpu")

    def gen_dataloader(
        self, input_dir: Path, object_root: str, object_name: str
    ) -> DataLoader:
        # Instantiate lightning datamodule
        datamodule_cfg = self.cfg.datamodule
        OmegaConf.set_struct(
            datamodule_cfg, False
        )  # HACK: enable to add new key-value pairs
        if "predict" not in datamodule_cfg:
            import copy

            datamodule_cfg.predict = copy.deepcopy(datamodule_cfg.test.jcre3)
            datamodule_cfg.predict.include_nonidentical = True
        datamodule_cfg.predict.data_path = str(input_dir)
        datamodule_cfg.predict.object_file_root = object_root
        datamodule_cfg.predict.object_file_name = object_name
        datamodule_cfg.num_workers = int(self.cfg.num_workers)
        datamodule_cfg.batch_size = int(self.cfg.max_batches_per_device)
        datamodule = MTDataModule(cfg=datamodule_cfg)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule.predict_dataloader()

    def analyze(
        self,
        dataloader: DataLoader,
        prediction_destination: Union[Path, TextIO, None] = None,
        json_destination: Union[Path, TextIO, None] = None,
    ) -> None:
        self.prediction_writer.prediction_destination = prediction_destination
        self.prediction_writer.json_destination = json_destination
        self.trainer.predict(model=self.model, dataloaders=dataloader)


@hydra.main(config_path="../configs/predict", config_name="mmref", version_base=None)
def main(cfg: DictConfig):
    if isinstance(cfg.devices, str):
        cfg.devices = (
            list(map(int, cfg.devices.split(",")))
            if "," in cfg.devices
            else int(cfg.devices)
        )
    if isinstance(cfg.max_batches_per_device, str):
        cfg.max_batches_per_device = int(cfg.max_batches_per_device)
    if isinstance(cfg.num_workers, str):
        cfg.num_workers = int(cfg.num_workers)

    analyzer = Analyzer(cfg)

    assert cfg.input_dir is not None
    assert cfg.export_dir is not None

    source = Path(cfg.input_dir)
    destination = Path(cfg.export_dir)
    dataloader = analyzer.gen_dataloader(
        source, cfg.object_file_root, cfg.object_file_name
    )
    analyzer.analyze(dataloader, prediction_destination=destination)


if __name__ == "__main__":
    main()
