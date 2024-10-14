from rhoknp import Sentence

from cl_mmref.cohesion_tools.task import Task
from cl_mmref.datamodule.example import MMRefExample
from cl_mmref.datasets.mmref_dataset import MMRefDataset
from cl_mmref.utils.annotation import PhraseAnnotation, SentenceAnnotation
from cl_mmref.utils.prediction import (
    BoundingBoxPrediction,
    ObjectFeature,
    PhrasePrediction,
    RelationPrediction,
    SentencePrediction,
)
from cl_mmref.utils.util import Rectangle


class SentenceJsonWriter:
    def __init__(self, dataset: MMRefDataset) -> None:
        self.rel_types: list[str] = dataset.rel_types
        self.tasks: list[Task] = dataset.tasks
        self.task_to_rels: dict[Task, list[str]] = dataset.task_to_rels

    def write_sentence_predictions(
        self,
        example: MMRefExample,
        sentences: list[SentenceAnnotation],
        candidate_selection_prediction: list[
            list[list[int]]
        ],  # (phrase, rel, candidate)
        is_analysis_target: list[list[bool]],  # (phrase, task)
    ) -> list[SentencePrediction]:
        assert len(candidate_selection_prediction) == len(is_analysis_target)
        phrase_predictions: list[PhrasePrediction] = self.write_phrase_predictions(
            example,
            [
                sentence.sid
                for sentence in sentences
                for _ in range(len(sentence.phrases))
            ],
            [phrase for sentence in sentences for phrase in sentence.phrases],
            candidate_selection_prediction,
            is_analysis_target,
        )
        sentence_predictions: list[SentencePrediction] = []
        for sentence in sentences:
            phrases_tmp: list[PhrasePrediction] = []
            for phrase_prediction in phrase_predictions:
                if sentence.sid != phrase_prediction.sid:
                    continue
                phrases_tmp.append(phrase_prediction)

            sentence_predictions.append(
                SentencePrediction(
                    text=sentence.text, sid=sentence.sid, phrases=phrases_tmp
                )
            )
        return sentence_predictions

    def write_phrase_predictions(
        self,
        example: MMRefExample,
        sids: list[str],
        phrases: list[PhraseAnnotation],
        candidate_selection_prediction: list[
            list[list[int]]
        ],  # (phrase, rel, candidate)
        is_analysis_target: list[list[bool]],  # (phrase, task)
    ) -> list[PhrasePrediction]:
        assert (
            len(phrases)
            == len(candidate_selection_prediction)
            == len(is_analysis_target)
        )

        phrase_predictions: list[PhrasePrediction] = []
        for sid, phrase, selected_candidates, is_targets in zip(
            sids,
            phrases,
            candidate_selection_prediction,
            is_analysis_target,
        ):
            rel_type_to_candidate = dict(zip(self.rel_types, selected_candidates))

            for task, is_target in zip(self.tasks, is_targets):
                relation_predictions: list[RelationPrediction] = []
                # mmref_base_phrase = example.phrases[task][idx]
                # NOTE: mmref_base_phrase.is_target is gold annotation
                if is_target is True:
                    for rel_type in self.task_to_rels[task]:
                        candidate_predictions: list[ObjectFeature] = [
                            example.candidates[idx]
                            for idx in rel_type_to_candidate[rel_type]
                        ]
                        bbox_predictions: list[BoundingBoxPrediction] = []
                        for pred in candidate_predictions:
                            bbox_predictions.append(
                                BoundingBoxPrediction(
                                    image_id=pred.image_id,
                                    class_id=pred.class_id,
                                    confidence=pred.confidence,
                                    rect=pred.rect,
                                )
                            )

                        relation_predictions.append(
                            RelationPrediction(
                                type=rel_type, bounding_box=bbox_predictions
                            )
                        )

                phrase_predictions.append(
                    PhrasePrediction(
                        sid=sid,
                        task=task,
                        text=phrase.text,
                        relations=relation_predictions,
                    )
                )

        return phrase_predictions
