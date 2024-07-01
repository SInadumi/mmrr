
import lightning.pytorch as pl
import torch

encoder = torch.nn.Sequential(torch.nn.Linear(28 * 28, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3))
decoder = torch.nn.Sequential(torch.nn.Linear(3, 64), torch.nn.ReLU(), torch.nn.Linear(64, 28 * 28))

class DummyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = encoder

    def forward(self, inputs, target):
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx) -> None:
        return None

