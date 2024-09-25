from collections import defaultdict

from rhoknp import Document

from cohesion_tools.task import Task
from datamodule.example import MMRefExample
from datasets.mmref_dataset import MMRefDataset
from utils.annotation import PhraseAnnotation, SentenceAnnotation
from utils.prediction import (
    BoundingBoxPrediction,
    ObjectFeature,
    PhrasePrediction,
    RelationPrediction,
    SentencePrediction,
)
from utils.sub_document import extract_target_sentences
from utils.util import Rectangle


class SentenceJsonWriter:
    def __init__(self, dataset: MMRefDataset) -> None:
        self.rel_types: list[str] = dataset.rel_types
        self.tasks: list[Task] = dataset.tasks
        self.task_to_rels: dict[Task, list[str]] = dataset.task_to_rels

    def write_sentence_annotations(
        self, document: Document, phrase_annotations: list[PhraseAnnotation]
    ) -> list[SentenceAnnotation]:
        sid_to_phrases: dict[str, list[PhraseAnnotation]] = defaultdict(list)
        assert len(document.base_phrases) == len(phrase_annotations)
        for base_phrase, phrase_annotation in zip(
            document.base_phrases, phrase_annotations
        ):
            sid_to_phrases[base_phrase.sentence.sid].append(phrase_annotation)

        sentence_annotations: list[SentenceAnnotation] = []
        for sentence in extract_target_sentences(document):
            sentence_annotations.append(
                SentenceAnnotation(
                    sid=sentence.sid,
                    text=sentence.text,
                    phrases=sid_to_phrases[sentence.sid],
                )
            )

        return sentence_annotations

    def write_sentence_predictions(
        self,
        example: MMRefExample,
        document: Document,
        candidate_selection_prediction: list[
            list[list[int]]
        ],  # (phrase, rel, candidate)
        is_analysis_target: list[list[bool]],  # (phrase, task)
    ) -> list[SentencePrediction]:
        assert (
            len(document.base_phrases)
            == len(candidate_selection_prediction)
            == len(is_analysis_target)
        )
        phrase_predictions: list[PhrasePrediction] = self.write_phrase_predictions(
            example, document, candidate_selection_prediction, is_analysis_target
        )
        sentence_predictions: list[SentencePrediction] = []
        for sentence in extract_target_sentences(document):
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
        document: Document,
        candidate_selection_prediction: list[
            list[list[int]]
        ],  # (phrase, rel, candidate)
        is_analysis_target: list[list[bool]],  # (phrase, task)
    ) -> list[PhrasePrediction]:
        assert (
            len(document.base_phrases)
            == len(candidate_selection_prediction)
            == len(is_analysis_target)
        )

        phrase_predictions: list[PhrasePrediction] = []
        for idx, (base_phrase, selected_candidates, is_targets) in enumerate(
            zip(
                document.base_phrases,
                candidate_selection_prediction,
                is_analysis_target,
            )
        ):
            rel_type_to_candidate = dict(zip(self.rel_types, selected_candidates))

            for task, is_target in zip(self.tasks, is_targets):
                relation_predictions: list[RelationPrediction] = []
                # mmref_base_phrase = example.phrases[task][idx]
                # NOTE: mmref_base_phrase.is_target is gold annotation
                if is_target is True:
                    for rel_type in self.task_to_rels[task]:
                        candidate_predictions: list[ObjectFeature] = [
                            example.all_candidates[idx]
                            for idx in rel_type_to_candidate[rel_type]
                        ]
                        bbox_predictions: list[BoundingBoxPrediction] = []
                        for pred in candidate_predictions:
                            bbox = pred.bbox.tolist()
                            bbox_predictions.append(
                                BoundingBoxPrediction(
                                    image_id=pred.image_id,
                                    class_id=pred.class_id.item(),
                                    rect=Rectangle(
                                        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]
                                    ),
                                    confidence=pred.score.item(),
                                )
                            )

                        relation_predictions.append(
                            RelationPrediction(
                                type=rel_type, bounding_box=bbox_predictions
                            )
                        )

                phrase_predictions.append(
                    PhrasePrediction(
                        sid=base_phrase.sentence.sid,
                        task=task,
                        text=base_phrase.text,
                        relations=relation_predictions,
                    )
                )

        return phrase_predictions
