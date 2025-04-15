from argparse import ArgumentParser
from pathlib import Path

from rhoknp import Document, Sentence

from mmrr.utils.annotation import DatasetInfo


def count_tags(document: Document, dataset_info: DatasetInfo, utterance_window_size: int = 1, tag: str = "=") -> int:
    tot_tag = 0
    sid2sentence: dict[str, Sentence] = {sentence.sid: sentence for sentence in document.sentences}
    for idx in range(len(dataset_info.utterances)):
        utterances_in_window = dataset_info.utterances[max(0, idx + 1 - utterance_window_size): idx + 1]
        doc_utterance = Document.from_sentences(
            [sid2sentence[sid] for utterance in utterances_in_window for sid in utterance.sids]
        )
        for base_phrase in doc_utterance.base_phrases:
            for rel_tag in base_phrase.rel_tags:
                if tag in rel_tag.type:
                    tot_tag += 1
    return tot_tag

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "INPUT",
        type=Path,
        help="path to input annotation dir",
    )
    parser.add_argument("--window-size", type=int, default=1, help="utterances window size")
    parser.add_argument("--tag", type=str, default="=")
    args = parser.parse_args()

    doc_indices = []
    id_dir = args.INPUT / "id"
    for id_file in id_dir.glob("*.id"):
        if id_file.stem not in {"val", "valid", "test"}:
            continue
        for doc_id in id_file.read_text().splitlines():
            doc_indices.append(doc_id)

    tot_tag = 0
    for doc_id in doc_indices:
        document = Document.from_knp(
                (args.INPUT / "knp" / f"{doc_id}.knp").read_text()
            )
        dataset_info = DatasetInfo.from_json(
                (args.INPUT / "recording" / doc_id / "info.json").read_text()
            )
        tot_tag += count_tags(document, dataset_info, args.window_size, args.tag)
    print(f"window-size: {args.window_size}, tag: {args.tag}, total: {tot_tag}")



if __name__ == "__main__":
    main()
