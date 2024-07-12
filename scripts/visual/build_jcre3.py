import json
import os
from argparse import ArgumentParser
from pathlib import Path

from rhoknp import Document, RegexSenter

senter = RegexSenter()
exclude_vis_ids = ["20220302-56130295-0"]


def to_idx_from_sid(sid: str) -> int:
    # '<did>-1-<idx>' -> idx
    return int(sid.split("-")[-1])


def split_utterances_to_sentences(dataset_dir: Path, source_path: Path) -> dict:
    """visual_annotation/*.jsonの"utterances"エントリとtextual_annotation/*.knpの文分割を揃える処理

    c.f.) https://rhoknp.readthedocs.io/en/stable/_modules/rhoknp/processors/senter.html
    """
    annotation = json.load(open(source_path, "r", encoding="utf-8"))
    scenario_id = source_path.stem
    document = Document.from_knp(
        (dataset_dir / "textual_annotations" / f"{scenario_id}.knp").read_text()
    )

    repl_utterances = []

    # split utterances field
    for utterance in annotation["utterances"]:
        _sentences = senter.apply_to_document(utterance["text"])
        repl_utterances.extend(
            [{"text": s.text, "phrases": utterance["phrases"]} for s in _sentences.sentences]
        )
    assert len(repl_utterances) == len(document.sentences)

    # format "ret" phrase entries by base_phrases
    for sentence in document.sentences:
        s_idx = to_idx_from_sid(sentence.sid) # a index of sid
        _utterance = repl_utterances[s_idx]
        if len(sentence.base_phrases) != len(_utterance["phrases"]):
            doc_phrase = [b.text for b in sentence.base_phrases]
            vis_phrase = [u["text"] for u in _utterance["phrases"]]
            st_idx = vis_phrase.index(doc_phrase[0])
            end_idx = st_idx + len(doc_phrase)
            _utterance["phrases"] = _utterance["phrases"][st_idx:end_idx]

    annotation["utterances"] = repl_utterances

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
    for source in visual_paths:
        vis_id = source.stem
        if vis_id in exclude_vis_ids:
            # FIXME: 20220302-56130295-0.knp の文分割におけるアノテーションミス
            # c.f.) https://github.com/riken-grp/J-CRe3/blob/ca6f5e86a4939f60158ea2999ffab6bea6924527/textual_annotations/20220302-56130295-0.knp#L161-L198
            # "あそうそう。おもちゃを ..."が区切られていない
            continue
        image_text_annotation = split_utterances_to_sentences(Path(args.INPUT), source)
        target = output_root / vis_id2split[vis_id] / f"{vis_id}.json"
        json.dump(image_text_annotation,
                  open(target, "w"),
                  indent=1,
                  ensure_ascii=False
                )



if __name__ == "__main__":
    main()
