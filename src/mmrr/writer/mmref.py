from mmrr.datamodule.example import MMRefExample
from mmrr.datasets.mmref_dataset import MMRefDataset
from mmrr.tools.task import Task
from mmrr.utils.annotation import PhraseAnnotation, SentenceAnnotation
from mmrr.utils.prediction import (
    BoundingBoxPrediction,
    ObjectFeature,
    PhrasePrediction,
    RelationPrediction,
    SentencePrediction,
)


class ProbabilityJsonWriter:
    def __init__(self, dataset: MMRefDataset) -> None:
        self.rel_types: list[str] = dataset.rel_types
        self.tasks: list[Task] = dataset.tasks
        self.task_to_rels: dict[Task, list[str]] = dataset.task_to_rels

    def write_sentence_predictions(
        self,
        example: MMRefExample,
        sentences: list[SentenceAnnotation],
        predicted_candidates: list[list[list[int]]],  # (phrase, rel, candidate)
        predicted_probabilities: list[list[list[float]]],  # (phrase, rel, candidate)
    ) -> list[SentencePrediction]:
        phrase_predictions: list[PhrasePrediction] = self.write_phrase_predictions(
            example,
            [
                sentence.sid
                for sentence in sentences
                for _ in range(len(sentence.phrases))
            ],
            [phrase for sentence in sentences for phrase in sentence.phrases],
            predicted_candidates,
            predicted_probabilities,
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
        predicted_candidates: list[list[list[int]]],  # (phrase, rel, candidate)
        predicted_probabilities: list[list[list[float]]],  # (phrase, rel, candidate)
    ) -> list[PhrasePrediction]:
        assert len(phrases) == len(predicted_candidates) == len(predicted_probabilities)

        phrase_predictions: list[PhrasePrediction] = []
        for example_idx, (sid, phrase, candidates, probabilities) in enumerate(
            zip(
                sids,
                phrases,
                predicted_candidates,
                predicted_probabilities,
            )
        ):
            rel_type_to_candidate = dict(zip(self.rel_types, candidates))
            rel_type_to_probability = dict(zip(self.rel_types, probabilities))

            relation_predictions: list[RelationPrediction] = []
            for task in self.tasks:
                mmref_base_phrase = example.phrases[task][example_idx]
                # NOTE: `mmref_base_phrase.is_target` is gold base_phrase
                if mmref_base_phrase.is_target is True:
                    for rel_type in self.task_to_rels[task]:
                        # NOTE: `mmref_base_phrase.rel2tags[rel_type]` has gold phrase to object relationships
                        if mmref_base_phrase.rel2tags.get(rel_type) is None:
                            continue
                        candidate_predictions: list[ObjectFeature] = (
                            self.update_object_features(
                                example.candidates,
                                rel_type_to_candidate[rel_type],
                                rel_type_to_probability[rel_type],
                            )
                        )
                        bbox_predictions: list[BoundingBoxPrediction] = []
                        for pred in candidate_predictions:
                            bbox_predictions.append(
                                BoundingBoxPrediction(
                                    image_id=f"{pred.image_id:03}",
                                    class_id=pred.class_id,
                                    confidence=pred.confidence,
                                    rect=pred.rect,
                                )
                            )

                        relation_predictions.append(
                            RelationPrediction(
                                type=rel_type, bounding_boxes=bbox_predictions
                            )
                        )

            phrase_predictions.append(
                PhrasePrediction(
                    sid=sid,
                    text=phrase.text,
                    relations=relation_predictions,
                )
            )

        return phrase_predictions

    @staticmethod
    def update_object_features(
        object_annotations: list[ObjectFeature],
        candidates: list[int],
        probabilities: list[float],
    ) -> list[ObjectFeature]:
        object_predictions: list[ObjectFeature] = []
        for candidate_idx, prob in zip(candidates, probabilities):
            prediction: ObjectFeature = object_annotations[candidate_idx]
            prediction.confidence = prob
            # HACK: clipping
            if prob > 0.01:
                object_predictions.append(prediction)
        return object_predictions
