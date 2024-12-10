import io
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, TextIO, Union

import lightning.pytorch as pl
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.utils.data import Dataset
from typing_extensions import override

from mmrr.datamodule.example import MMRefExample
from mmrr.datasets import MMRefDataset
from mmrr.utils.annotation import SentenceAnnotation
from mmrr.utils.prediction import MMRefPrediction, SentencePrediction
from mmrr.writer.mmref import ProbabilityJsonWriter

logger = logging.getLogger(__name__)


class MMRefWriter(BasePredictionWriter):
    def __init__(
        self,
        prediction_destination: Union[Path, TextIO, None] = None,
        json_destination: Union[Path, TextIO, None] = None,
        clipping_threshold: float = 0.0,
    ) -> None:
        super().__init__(write_interval="epoch")

        self.prediction_destination: Union[Path, TextIO, None] = prediction_destination
        self.json_destination: Union[Path, TextIO, None] = json_destination
        self.clipping_threshold = clipping_threshold
        for dest in (self.prediction_destination, self.json_destination):
            if dest is None:
                continue
            assert isinstance(
                dest, (Path, TextIO)
            ), f"destination must be either Path or TextIO, but got {type(dest)}"
            if isinstance(dest, Path):
                dest.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        pass

    @override
    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]] = None,
    ) -> None:
        dataset: Dataset = trainer.predict_dataloaders.dataset  # type: ignore
        assert isinstance(dataset, MMRefDataset)
        json_writer = ProbabilityJsonWriter(dataset, clipping_threshold=self.clipping_threshold)
        for pred in predictions:
            batch_example_ids = pred["example_ids"]  # (b)
            batch_relation_logits = pred["relation_logits"]  # (b, rel, seq, seq)
            assert len(batch_relation_logits) == len(batch_example_ids)
            for example_id, relation_logits in zip(
                batch_example_ids, batch_relation_logits
            ):
                example: MMRefExample = dataset.examples[example_id.item()]
                sentences: list[SentenceAnnotation] = [
                    dataset.sid2vis_sentence[sid] for sid in example.sentence_indices
                ]
                # (phrase, rel, candidate)
                relation_prediction: np.ndarray = dataset.dump_relation_prediction(
                    relation_logits.cpu().numpy(), example
                )
                # descending order
                predicted_candidates: np.ndarray = np.argsort(
                    -relation_prediction, axis=2
                )
                predicted_probabilities: np.ndarray = (-1) * np.sort(
                    -relation_prediction, axis=2
                )
                assert predicted_candidates.size == predicted_probabilities.size

                sentence_prediction = json_writer.write_sentence_predictions(
                    example,
                    sentences,
                    predicted_candidates.tolist(),
                    predicted_probabilities.tolist(),
                )

                self.write_prediction(
                    example.doc_id, example.image_id, sentence_prediction
                )

    def write_prediction(
        self, doc_id: str, image_id: str, prediction: list[SentencePrediction]
    ) -> None:
        mmref_prediction = MMRefPrediction(
            doc_id=doc_id,
            image_id=image_id,
            phrases=[phrase for sp in prediction for phrase in sp.phrases],
        )
        if isinstance(self.prediction_destination, Path):
            self.prediction_destination.mkdir(parents=True, exist_ok=True)
            self.prediction_destination.joinpath(f"{image_id}.json").write_text(
                mmref_prediction.to_json(indent=2, ensure_ascii=False)
            )
        elif isinstance(self.prediction_destination, io.TextIOBase):
            self.prediction_destination.write(
                mmref_prediction.to_json(indent=2, ensure_ascii=False)
            )
