import json
import logging
import math
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Literal

from rhoknp import Document
from src.utils.annotation import (
    DatasetInfo,
    ImageTextAnnotation,
    SentenceAnnotation,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ImageTextAugmenter:
    SPAN_TYPE = Literal[
        "past-current", "prev-current", "current", "prev-next", "current-next"
    ]

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        _id2cat = json.load(open("./data/categories.json", "r", encoding="utf-8"))
        self.cat2id = {v: i for i, v in enumerate(_id2cat)}

    @staticmethod
    def to_idx_from_sid(sid: str) -> int:
        # '<did>-1-<idx>' -> idx
        return int(sid.split("-")[-1])

    @staticmethod
    def remove_unseen_objects(annotation: ImageTextAnnotation) -> ImageTextAnnotation:
        for image in annotation.images:
            # "region"は学習・評価の対象外
            image.boundingBoxes = [
                bbox for bbox in image.boundingBoxes if bbox.className != "region"
            ]
        for utterance in annotation.utterances:
            for phrase in utterance.phrases:
                _relation = []
                for relation in phrase.relations:
                    # unseen object を除外
                    if relation.boundingBoxes is None:
                        continue
                    # "region"は学習・評価の対象外
                    relation.boundingBoxes = [
                        bbox
                        for bbox in relation.boundingBoxes
                        if bbox.className != "region"
                    ]
                    if len(relation.boundingBoxes) > 1:
                        _relation.append(relation)

                phrase.relations = _relation

        return annotation

    def split_utterances_to_sentences(
        self, annotation: ImageTextAnnotation
    ) -> ImageTextAnnotation:
        """visual_annotation/*.jsonの"utterances"エントリとtextual_annotation/*.knpの文分割を揃える処理"""
        scenario_id = annotation.scenarioId
        document = Document.from_knp(
            (
                self.dataset_dir / "textual_annotations" / f"{scenario_id}.knp"
            ).read_text()
        )
        dataset_info = DatasetInfo.from_json(
            (self.dataset_dir / "recording" / scenario_id / "info.json").read_text()
        )

        # collect sids corresponding to utterances
        sid_mapper: list[int] = [u_info.sids for u_info in dataset_info.utterances]
        assert len(sid_mapper) == len(annotation.utterances)

        # split utterances field
        sentences = []  # [{"sid": xx, "phrases": xx}, ...]
        for idx, utterance in enumerate(annotation.utterances):
            sids = sid_mapper[idx]
            sentences.extend(
                [
                    SentenceAnnotation(text="", phrases=utterance.phrases, sid=sid)
                    for sid in sids
                ]
            )
        assert len(sentences) == len(document.sentences)

        # format visual phrases by base_phrases
        for sentence in document.sentences:
            s_idx = self.to_idx_from_sid(sentence.sid)  # a index of sid
            _s = sentences[s_idx]
            _s.text = sentence.text
            if len(sentence.base_phrases) != len(_s.phrases):
                # update visual phrase annotation
                doc_phrase = [b.text for b in sentence.base_phrases]
                vis_phrase = [u.text for u in _s.phrases]
                st_idx = vis_phrase.index(doc_phrase[0])
                end_idx = st_idx + len(doc_phrase)
                _s.phrases = _s.phrases[st_idx:end_idx]
        annotation.utterances = sentences

        return annotation

    def add_class_id(self, annotation: ImageTextAnnotation) -> ImageTextAnnotation:
        iid2cid = {}

        # add object class id to bounding box annotations
        for image in annotation.images:
            for bbox in image.boundingBoxes:
                iid = bbox.instanceId
                cid = self.cat2id[bbox.className]
                bbox.classId = cid
                iid2cid[iid] = cid

        # add object class id to phrase to object relation annotations
        for utterance in annotation.utterances:
            for phrase in utterance.phrases:
                for relation in phrase.relations:
                    iid = relation.instanceId
                    relation.classId = iid2cid[iid]
        return annotation

    def add_bboxes_to_phrase_annotations(
        self, annotation: dict, image_span: SPAN_TYPE
    ) -> dict:
        scenario_id = annotation.scenarioId
        image_id_to_annotation = {image.imageId: image for image in annotation.images}

        dataset_info = DatasetInfo.from_json(
            (self.dataset_dir / "recording" / scenario_id / "info.json").read_text()
        )

        assert len(dataset_info.utterances) == len(annotation.utterances)
        a_utterances = annotation.utterances
        d_utterances = dataset_info.utterances
        all_image_ids = [image.id for image in dataset_info.images]
        ignore_cnt = 0
        for idx, (a_utt, d_utt) in enumerate(zip(a_utterances, d_utterances)):
            sidx = math.ceil(d_utt.start / 1000)
            eidx = math.ceil(d_utt.end / 1000)

            if idx >= 1:
                prev_utterance = d_utterances[idx - 1]
                prev_eidx = math.ceil(prev_utterance.end / 1000)
            else:
                prev_eidx = 0

            if idx + 1 < len(d_utterances):
                next_utterance = d_utterances[idx + 1]
                next_sdix = math.ceil(next_utterance.start / 1000)
            else:
                next_sdix = len(d_utterances)

            if image_span == "past-current":
                image_ids = all_image_ids[:eidx]
            elif image_span == "prev-current":
                image_ids = all_image_ids[prev_eidx:eidx]
            elif image_span == "current":
                image_ids = all_image_ids[sidx:eidx]
            elif image_span == "prev-next":
                image_ids = all_image_ids[prev_eidx:next_sdix]
            elif image_span == "current-next":
                image_ids = all_image_ids[sidx:next_sdix]
            else:
                raise ValueError(f"Unknown image span: {image_span}")

            instance_id_to_bboxes = defaultdict(lambda: [])
            for iid in image_ids:
                for bbox in image_id_to_annotation[iid].boundingBoxes:
                    instance_id_to_bboxes[bbox.instanceId].append(bbox)

            for phrase in a_utt.phrases:
                if len(phrase.relations) == 0:
                    continue
                for rel in phrase.relations:
                    if rel.instanceId in instance_id_to_bboxes:
                        rel.boundingBoxes = instance_id_to_bboxes[rel.instanceId]
                    else:
                        ignore_cnt += 1
                        logger.info(
                            f"{rel.instanceId} (in {scenario_id}:{iid}) is ignored."
                        )
        logger.info(f"{scenario_id}:{ignore_cnt} instances is ignored.")
        return self.remove_unseen_objects(annotation)


def main():
    parser = ArgumentParser()
    parser.add_argument("INPUT", type=str, help="path to input visual annotation dir")
    parser.add_argument("OUTPUT", type=str, help="path to output dir")
    parser.add_argument("--id", type=str, help="path to id")
    parser.add_argument(
        "--image-span", type=str, default="current", help="phrase-grounding range"
    )

    args = parser.parse_args()

    visual_dir = Path(args.INPUT) / "visual_annotations"
    output_root = Path(args.OUTPUT)

    vis_id2split = {}
    for id_file in Path(args.id).glob("*.id"):
        if id_file.stem not in {"train", "dev", "valid", "test"}:
            continue
        split = "valid" if id_file.stem == "dev" else id_file.stem
        output_root.joinpath(split).mkdir(parents=True, exist_ok=True)
        for vis_id in id_file.read_text().splitlines():
            vis_id2split[vis_id] = split

    # split visual annotations
    visual_paths = visual_dir.glob("*.json")
    augmenter = ImageTextAugmenter(Path(args.INPUT))
    for source in visual_paths:
        vis_id = source.stem
        image_text_annotation = ImageTextAnnotation.from_json(Path(source).read_text())
        image_text_annotation = augmenter.add_bboxes_to_phrase_annotations(
            image_text_annotation, args.image_span
        )
        image_text_annotation = augmenter.split_utterances_to_sentences(
            image_text_annotation
        )
        image_text_annotation = augmenter.add_class_id(image_text_annotation)
        target = output_root / vis_id2split[vis_id] / f"{vis_id}.json"
        target.write_text(image_text_annotation.to_json(ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
