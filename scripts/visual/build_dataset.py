import copy
import json
import logging
import math
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from rhoknp import Document

from mmrr.tools.constants import CASES, EXOPHORA_REFERENT_TYPES
from mmrr.tools.extractors.pas import PasExtractor
from mmrr.utils.annotation import (
    DatasetInfo,
    ImageTextAnnotation,
    Phrase2ObjectRelation,
    SentenceAnnotation,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ImageTextAugmenter:
    def __init__(self, dataset_dir: Path, dataset_name: str) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        _id2cat = json.load(open("./data/categories.json", "r", encoding="utf-8"))
        self.cat2id = {v: i for i, v in enumerate(_id2cat)}
        self.pas_extractor = PasExtractor(
            CASES,
            EXOPHORA_REFERENT_TYPES,
            verbal_predicate=True,
            nominal_predicate=True,
        )

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
                    if len(relation.boundingBoxes) > 0:
                        _relation.append(relation)

                phrase.relations = _relation

        return annotation

    def split_utterances_to_sentences(
        self, annotation: ImageTextAnnotation
    ) -> ImageTextAnnotation:
        """image_text_annotation/*.jsonの"utterances"エントリとknp/*.knpの文分割を揃える処理"""
        scenario_id = annotation.scenarioId
        document = Document.from_knp(
            (self.dataset_dir / "knp" / f"{scenario_id}.knp").read_text()
        )
        dataset_info = DatasetInfo.from_json(
            (self.dataset_dir / "recording" / scenario_id / "info.json").read_text()
        )

        # collect sids corresponding to utterances
        sid_mapper: list[list[int]] = [
            u_info.sids for u_info in dataset_info.utterances
        ]
        assert len(sid_mapper) == len(annotation.utterances)

        # split utterances field
        vis_sentences = []  # [{"sid": xx, "phrases": xx}, ...]
        for idx, utterance in enumerate(annotation.utterances):
            sids = sid_mapper[idx]
            vis_sentences.extend(
                [
                    SentenceAnnotation(text="", phrases=utterance.phrases, sid=sid)
                    for sid in sids
                ]
            )
        assert len(vis_sentences) == len(document.sentences)

        # format visual phrases by base_phrases
        for sentence in document.sentences:
            s_idx = self.to_idx_from_sid(sentence.sid)  # a index of sid
            vis_sentence = vis_sentences[s_idx]
            vis_sentence.text = sentence.text
            if len(sentence.base_phrases) != len(vis_sentence.phrases):
                # update visual phrase annotation
                doc_phrase = [b.text for b in sentence.base_phrases]
                vis_phrase = [u.text for u in vis_sentence.phrases]
                st_idx = vis_phrase.index(doc_phrase[0])
                end_idx = st_idx + len(doc_phrase)
                vis_sentence.phrases = vis_sentence.phrases[st_idx:end_idx]
        annotation.utterances = vis_sentences

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

    def split_annotations_per_frame(
        self,
        annotation: ImageTextAnnotation,
        num_utterances: int,
        num_overlapping: int,
    ) -> list[ImageTextAnnotation]:
        ret = []
        split_stride = num_utterances - num_overlapping
        assert split_stride > 0

        scenario_id = annotation.scenarioId
        dataset_info = DatasetInfo.from_json(
            (self.dataset_dir / "recording" / scenario_id / "info.json").read_text()
        )
        iid2bboxes = {
            image.imageId: [bbox for bbox in image.boundingBoxes]
            for image in annotation.images
        }
        for start_idx in range(0, len(dataset_info.utterances), split_stride):
            end_idx = min(start_idx + num_utterances, len(dataset_info.utterances))
            assert start_idx <= end_idx
            utterances = [utt for utt in dataset_info.utterances[start_idx:end_idx]]
            sids = [sid for utt in utterances for sid in utt.sids]
            iids = list(set(iid for utt in utterances for iid in utt.image_ids))
            for iid in iids:
                # collect sentence annotations based on utterance annotations
                _utterances = [
                    copy.deepcopy(sent)
                    for sent in annotation.utterances
                    if sent.sid in sids
                ]
                # update boundingBoxes
                # `utterances` -> `phrases` -> `relations` -> `boundingBoxes`
                _instances = {bbox.instanceId: bbox for bbox in iid2bboxes[iid]}
                tot_bboxes = 0
                for utterance in _utterances:
                    for phrase in utterance.phrases:
                        for rel in phrase.relations:
                            _set_instances = set(
                                bbox.instanceId for bbox in rel.boundingBoxes
                            )
                            _bboxes = []
                            for _instanceId in _set_instances:
                                if (_bbox := _instances.get(_instanceId)) is not None:
                                    _bboxes.append(_bbox)
                            rel.boundingBoxes = _bboxes
                            tot_bboxes += len(_bboxes)
                if tot_bboxes == 0:
                    continue

                if self.dataset_name == "jcre3":
                    ret.append(
                        ImageTextAnnotation(
                            scenarioId=scenario_id,
                            images=[annotation.images[int(iid) - 1]],  # 0-origin
                            utterances=_utterances,
                        )
                    )
                elif self.dataset_name == "f30k_ent_jp":
                    ret.append(
                        ImageTextAnnotation(
                            scenarioId=scenario_id,
                            images=annotation.images,
                            utterances=_utterances,
                        )
                    )
                else:
                    raise ValueError(
                        f"`dataset_name` has an invalid argument {self.dataset_name}"
                    )
        return ret

    def add_bboxes_to_phrase_annotations(
        self, annotation: ImageTextAnnotation
    ) -> ImageTextAnnotation:
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
        for a_utt, d_utt in zip(a_utterances, d_utterances):
            start_idx = math.ceil(d_utt.start / 1000)
            end_idx = math.ceil(d_utt.end / 1000)
            assert start_idx <= end_idx
            image_ids = all_image_ids[start_idx:end_idx]

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
        logger.info(f"{scenario_id}:{ignore_cnt} instances are ignored.")
        return self.remove_unseen_objects(annotation)

    def add_silver_cases_to_phrase_annotations(
        self, annotation: ImageTextAnnotation
    ) -> ImageTextAnnotation:
        scenario_id = annotation.scenarioId
        document = Document.from_knp(
            (self.dataset_dir / "knp" / f"{scenario_id}.knp").read_text()
        )

        doc_phrases = document.base_phrases
        vis_phrases = [ph for utt in annotation.utterances for ph in utt.phrases]
        assert len(doc_phrases) == len(vis_phrases)
        _logs: dict[str, int] = defaultdict(lambda: 0)

        for doc_phrase, source_vis_phrase in zip(doc_phrases, vis_phrases):
            all_rels = self.pas_extractor.extract_rels(doc_phrase)
            if len(all_rels) == 0:
                continue
            for rel, tags in all_rels.items():
                for tag in tags:
                    global_idx = tag.base_phrase.global_index
                    target_vis_phrase = vis_phrases[global_idx]
                    for vis_rel in target_vis_phrase.relations:
                        source_vis_phrase.relations.append(
                            Phrase2ObjectRelation(
                                type=rel,
                                instanceId=vis_rel.instanceId,
                                boundingBoxes=vis_rel.boundingBoxes,
                            )
                        )
                    _logs[rel] += 1

        for _rel, _cnt in _logs.items():
            logger.info(f"{scenario_id}/{_rel}: {_cnt} relations are augmented.")
        return annotation


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "ROOT_DIR",
        type=str,
        help="path to input/output annotation dir (current project)",
    )
    parser.add_argument("--dataset-name", type=str, choices=["jcre3", "f30k_ent_jp"])
    parser.add_argument("--id", type=str, help="path to id")
    parser.add_argument(
        "--num-utterances-per-sample",
        type=int,
        default=3,
        help="number of utterances per a sample",
    )
    parser.add_argument(
        "--num-overlapping-utterances",
        type=int,
        default=0,
        help="number of overlapping utterances",
    )

    args = parser.parse_args()

    visual_dir = Path(args.ROOT_DIR) / "image_text_annotation"
    output_root = Path(args.ROOT_DIR)

    vis_id2split = {}
    for id_file in Path(args.id).glob("*.id"):
        if id_file.stem not in {"train", "valid", "val", "test"}:
            continue
        split = "valid" if id_file.stem == "val" else id_file.stem
        output_root.joinpath(split).mkdir(parents=True, exist_ok=True)
        for scenario_id in id_file.read_text().splitlines():
            vis_id2split[scenario_id] = split

    # split visual annotations
    visual_paths = visual_dir.glob("*.json")
    augmenter = ImageTextAugmenter(Path(args.ROOT_DIR), args.dataset_name)
    for source in visual_paths:
        scenario_id = source.stem
        raw_annot = json.load(open(source, "r", encoding="utf-8"))  # for faster loading
        image_text_annotation = ImageTextAnnotation(**raw_annot)
        image_text_annotation = augmenter.add_bboxes_to_phrase_annotations(
            image_text_annotation
        )
        if args.dataset_name == "jcre3":
            image_text_annotation = augmenter.split_utterances_to_sentences(
                image_text_annotation
            )
            image_text_annotation = augmenter.add_class_id(image_text_annotation)
            annotations = augmenter.split_annotations_per_frame(
                image_text_annotation,
                args.num_utterances_per_sample,
                args.num_overlapping_utterances,
            )

        if args.dataset_name == "f30k_ent_jp":
            image_text_annotation = augmenter.add_silver_cases_to_phrase_annotations(
                image_text_annotation
            )
            annotations = [image_text_annotation]

        for idx, annotation in enumerate(annotations):
            assert len(annotation.images) == 1
            iid = annotation.images[0].imageId
            output_file_name = f"{scenario_id}-{iid}-{idx}"
            target = (
                output_root / vis_id2split[scenario_id] / f"{output_file_name}.json"
            )
            target.write_text(annotation.to_json(ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
