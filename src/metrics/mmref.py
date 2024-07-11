from .base import BaseModuleMetric


# NOTE: Dummy Metric
class MMRefMetric(BaseModuleMetric):
    def __init__(self) -> None:
        super().__init__()

    def compute(self) -> None:
        pass
