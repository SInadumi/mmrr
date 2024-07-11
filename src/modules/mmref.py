import torch
from omegaconf import DictConfig

from metrics.mmref import MMRefMetric
from modules.base import BaseModule

dummy_model = torch.nn.Sequential(
    torch.nn.Linear(28 * 28, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3)
)


# NOTE: Dummy Module
class MMRefModule(BaseModule[MMRefMetric]):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams, MMRefMetric)
        self.model = dummy_model

    def forward(self, inputs, target):
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx) -> None:
        return None
