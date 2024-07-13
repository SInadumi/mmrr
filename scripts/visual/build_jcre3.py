import json
import os
from argparse import ArgumentParser
from pathlib import Path

from rhoknp import Document, RegexSenter

exclude_vis_ids = ["20220302-56130295-0"]


class ImageTextAugmenter:
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir

        self.senter = RegexSenter()

        _id2cat = json.load(open("./data/categories.json", "r", encoding="utf-8"))
        self.cat2id = {v: i for i, v in enumerate(_id2cat)}

    @staticmethod
    def to_idx_from_sid(sid: str) -> int:
        # '<did>-1-<idx>' -> idx
        return int(sid.split("-")[-1])

    def split_utterances_to_sentences(self, annotation: dict) -> dict:
        """visual_annotation/*.jsonの"utterances"エントリとtextual_annotation/*.knpの文分割を揃える処理

        cf.) https://rhoknp.readthedocs.io/en/stable/_modules/rhoknp/processors/senter.html
        """
        scenario_id = annotation["scenarioId"]
        document = Document.from_knp(
            (
                self.dataset_dir / "textual_annotations" / f"{scenario_id}.knp"
            ).read_text()
        )

        sentences = []

        # split utterances field
        for utterance in annotation["utterances"]:
            doc_sents = self.senter.apply_to_document(utterance["text"])
            sentences.extend(
                [
                    {"text": s.text, "phrases": utterance["phrases"]}
                    for s in doc_sents.sentences
                ]
            )
        assert len(sentences) == len(document.sentences)

        # format "ret" phrase entries by base_phrases
        for sentence in document.sentences:
            s_idx = self.to_idx_from_sid(sentence.sid)  # a index of sid
            sentences[s_idx]
            _s = sentences[s_idx]
            _s["sid"] = sentence.sid
            if len(sentence.base_phrases) != len(_s["phrases"]):
                doc_phrase = [b.text for b in sentence.base_phrases]
                vis_phrase = [u["text"] for u in _s["phrases"]]
                st_idx = vis_phrase.index(doc_phrase[0])
                end_idx = st_idx + len(doc_phrase)
                _s["phrases"] = _s["phrases"][st_idx:end_idx]

        annotation["utterances"] = sentences

        return annotation

    def add_class_id(self, annotation: dict) -> dict:
        scenario_id = annotation["scenarioId"]
        iid2cid = {}

        # add object class id to bounding box annotations
        for image in annotation["images"]:
            for bbox in image["boundingBoxes"]:
                iid = bbox["instanceId"]
                cid = self.cat2id[bbox["className"]]
                bbox["classId"] = cid
                iid2cid[iid] = cid

        # add object class id to phrase to object relation annotations
        for utterance in annotation["utterances"]:
            for phrase in utterance["phrases"]:
                for relation in phrase["relations"]:
                    iid = relation["instanceId"]
                    #  HACK: visual/*.json の "images"中にclassNameが存在しないinstanceIdを回避する例外処理
                    try:
                        relation["classId"] = iid2cid[iid]
                    except Exception as e:
                        relation["classId"] = -1  # HACK: dummy category ID
                        print(f"{scenario_id}: {e.__class__.__name__}: {e}")
        return annotation


def main():
    parser = ArgumentParser()
    parser.add_argument("INPUT", type=str, help="path to input visual annotation dir")
    parser.add_argument("OUTPUT", type=str, help="path to output dir")
    parser.add_argument("--id", type=str, help="path to id")

    args = parser.parse_args()

    visual_dir = Path(args.INPUT) / "visual_annotations"
    object_dir = Path(args.INPUT) / "region_features" / "regionclip_pretrained-cc_rn50"
    output_root = Path(args.OUTPUT)

    vis_id2split = {}
    for id_file in Path(args.id).glob("*.id"):
        if id_file.stem not in {"train", "dev", "valid", "test"}:
            continue
        split = "valid" if id_file.stem == "dev" else id_file.stem
        output_root.joinpath(split).mkdir(parents=True, exist_ok=True)
        for vis_id in id_file.read_text().splitlines():
            vis_id2split[vis_id] = split

    # split object features
    object_paths = object_dir.glob("*.pth")
    for source in object_paths:
        obj_id = source.stem
        target = output_root / vis_id2split[obj_id] / f"{obj_id}.pth"
        os.system(f"cp {source} {target}")

    # split visual annotations
    visual_paths = visual_dir.glob("*.json")
    augmenter = ImageTextAugmenter(Path(args.INPUT))
    for source in visual_paths:
        vis_id = source.stem
        if vis_id in exclude_vis_ids:
            # FIXME: 20220302-56130295-0.knp の文分割におけるアノテーションミス
            # cf.) https://github.com/riken-grp/J-CRe3/blob/ca6f5e86a4939f60158ea2999ffab6bea6924527/textual_annotations/20220302-56130295-0.knp#L161-L198
            # "あそうそう。おもちゃを ..." が区切られていない
            continue
        image_text_annotation = json.load(open(source, "r", encoding="utf-8"))
        image_text_annotation = augmenter.split_utterances_to_sentences(
            image_text_annotation
        )
        image_text_annotation = augmenter.add_class_id(image_text_annotation)
        target = output_root / vis_id2split[vis_id] / f"{vis_id}.json"
        json.dump(
            image_text_annotation, open(target, "w"), indent=1, ensure_ascii=False
        )


if __name__ == "__main__":
    main()
