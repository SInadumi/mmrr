from typing import Collection

import pandas as pd

from utils.annotation import SentenceAnnotation
from utils.prediction import SentencePrediction


class VisPASAnalysisEvaluator:
    def __init__(
        self,
        cases: Collection[str],
    ) -> None:
        self.cases: list[str] = list(cases)

    def run(
        self,
        predicted_annotation: SentencePrediction,
        gold_annotation: SentenceAnnotation,
    ) -> pd.DataFrame:
        metrics = pd.DataFrame()

        return metrics
