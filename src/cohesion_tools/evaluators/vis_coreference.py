import pandas as pd

from utils.annotation import SentenceAnnotation
from utils.prediction import SentencePrediction


class VisCoreferenceResolutionEvaluator:
    def __init__(self) -> None:
        pass

    def run(
        self,
        predicted_annotation: SentencePrediction,
        gold_annotation: SentenceAnnotation,
    ) -> pd.DataFrame:
        metrics = pd.DataFrame()

        return metrics
