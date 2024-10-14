import copy
import json
import logging
import math
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from rhoknp import Document

from cl_mmref.utils.annotation import (
    DatasetInfo,
    ImageTextAnnotation,
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
        sid_mapper: list[list[int]] = [
            u_info.sids for u_info in dataset_info.utterances
        ]
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

        for start_idx in range(0, len(dataset_info.utterances), split_stride):
            end_idx = min(start_idx + num_utterances, len(dataset_info.utterances))
            assert start_idx <= end_idx
            utterances = [utt for utt in dataset_info.utterances[start_idx:end_idx]]
            sids = [sid for utt in utterances for sid in utt.sids]
            iids = list(set(iid for utt in utterances for iid in utt.image_ids))
            for iid in iids:
                _utterances = [
                    copy.deepcopy(sent)
                    for sent in annotation.utterances
                    if sent.sid in sids
                ]
                tot_bboxes = 0
                for utterance in _utterances:
                    for phrase in utterance.phrases:
                        for rel in phrase.relations:
                            _bboxes = [
                                bbox
                                for bbox in rel.boundingBoxes
                                if bbox.imageId == iid
                            ]
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


def main():
    parser = ArgumentParser()
    parser.add_argument("INPUT", type=str, help="path to input visual annotation dir")
    parser.add_argument("OUTPUT", type=str, help="path to output dir")
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

    visual_dir = Path(args.INPUT) / "visual_annotations"
    output_root = Path(args.OUTPUT)

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
    augmenter = ImageTextAugmenter(Path(args.INPUT), args.dataset_name)
    for source in visual_paths:
        scenario_id = source.stem
        raw_annot = json.load(open(source, "r", encoding="utf-8")) # for faster loading
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
        for idx, annotation in enumerate(annotations):
            assert len(annotation.images) == 1
            iid = annotation.images[0].imageId
            output_file_name = f"{scenario_id}-{iid}"
            if args.dataset_name == "f30k_ent_jp":
                output_file_name = f"{idx}-{iid}"
            target = (
                output_root / vis_id2split[scenario_id] / f"{output_file_name}.json"
            )
            target.write_text(annotation.to_json(ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
