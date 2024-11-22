# Copyright (c) 2024, Nobuhiro Ueda
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from flickr30k_entities import Annotation, Document
from rhoknp import KNP, Jumanpp
from rhoknp import Document as KNPDocument
from rhoknp import Sentence as KNPSentence
from tqdm import tqdm

from mmrr.utils.annotation import (
    BoundingBox,
    DatasetInfo,
    ImageAnnotation,
    ImageInfo,
    ImageTextAnnotation,
    Phrase2ObjectRelation,
    PhraseAnnotation,
    Rectangle,
    SentenceAnnotation,
    UtteranceInfo,
)

jumanpp = Jumanpp()
knp = KNP(options=["-tab"])
# kwja = KWJA(options=["--tasks", "word", "--model-size", "large", "--input-format", "jumanpp"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flickr-id-file", type=Path, help="Path to Flickr30k ID file."
    )
    parser.add_argument(
        "--flickr-image-dir", type=str, help="Path to flickr image directory."
    )
    parser.add_argument(
        "--flickr-annotations-dir",
        type=str,
        help="Path to flickr Annotations directory.",
    )
    parser.add_argument(
        "--flickr-documents-dir", type=str, help="Path to flickr Sentences directory."
    )
    parser.add_argument("--output-dir", "-o", type=str, help="Path to output dir")
    args = parser.parse_args()

    flickr_ids = args.flickr_id_file.read_text().splitlines()
    flickr_image_dir = Path(args.flickr_image_dir)
    flickr_annotations_dir = Path(args.flickr_annotations_dir)
    flickr_documents_dir = Path(args.flickr_documents_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    dataset_dir = output_dir / "recording"
    dataset_dir.mkdir(exist_ok=True)
    knp_dir = output_dir / "knp"
    knp_dir.mkdir(exist_ok=True)
    annotation_dir = output_dir / "image_text_annotation"
    annotation_dir.mkdir(exist_ok=True)
    id_dir = output_dir / "id"
    id_dir.mkdir(exist_ok=True)

    scenario_ids: list[str] = []
    for flickr_image_id in tqdm(flickr_ids):
        flickr_annotation_file = flickr_annotations_dir / f"{flickr_image_id}.xml"
        flickr_sentences_file = flickr_documents_dir / f"{flickr_image_id}.txt"
        flickr_annotation = Annotation.from_xml(
            ET.parse(flickr_annotation_file).getroot()
        )
        flickr_sentences = flickr_sentences_file.read_text().splitlines()
        flickr_document = Document.from_string(flickr_image_id, flickr_sentences)

        scenario_ids.append(
            convert_flickr(
                f"{int(flickr_image_id):010d}",
                flickr_annotation,
                flickr_document,
                flickr_image_dir,
                dataset_dir,
                annotation_dir,
                knp_dir,
            )
        )
    id_dir.joinpath(f"{args.flickr_id_file.stem}.id").write_text(
        "\n".join(scenario_ids) + "\n"
    )


def convert_flickr(
    flickr_image_id: str,
    flickr_annotation: Annotation,
    flickr_document: Document,
    flickr_image_dir: Path,
    dataset_dir: Path,
    annotation_dir: Path,
    knp_dir: Path,
) -> str:
    instance_id_to_class_name = {}
    for flickr_sentence in flickr_document.sentences:
        for phrase in flickr_sentence.phrases:
            instance_id_to_class_name[str(phrase.phrase_id)] = phrase.phrase_type
    instance_id_to_bounding_box = {
        obj.name: (
            BoundingBox(
                imageId=flickr_image_id,
                instanceId=obj.name,
                rect=Rectangle(
                    x1=obj.bndbox.xmin,
                    y1=obj.bndbox.ymin,
                    x2=obj.bndbox.xmax,
                    y2=obj.bndbox.ymax,
                ),
                className=instance_id_to_class_name.get(
                    obj.name, ""
                ),  # Some objects are not referred in the sentence
            )
            if obj.bndbox is not None
            else None
        )
        for obj in flickr_annotation.objects
    }

    scenario_id = f"{flickr_image_id}"
    image_dir: Path = dataset_dir / scenario_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    utterances = []
    for idx, flickr_sentence in enumerate(flickr_document.sentences):
        utterances.append(
            UtteranceInfo(
                text=flickr_sentence.text,
                sids=[f"{scenario_id}-{idx:02}"],
                start=0,
                end=1,
                duration=1,
                speaker="",
                image_ids=[flickr_image_id],
            )
        )
    dataset_info = DatasetInfo(
        scenario_id=scenario_id,
        utterances=utterances,
        images=[
            ImageInfo(
                id=flickr_image_id,
                path=f"images/{flickr_image_id}.jpg",
                time=0,
            )
        ],
    )
    dataset_dir.joinpath(scenario_id, "info.json").write_text(
        dataset_info.to_json(ensure_ascii=False, indent=2)
    )

    phrase_to_morpheme_global_indices = []
    global_morphemes = []
    knp_sentences: list[KNPSentence] = []
    cursor = 0

    for idx, flickr_sentence in enumerate(flickr_document.sentences):
        phrase_to_morpheme_indices = {}
        morphemes = []
        for phrase in flickr_sentence.phrases:
            chunk = flickr_sentence.text[cursor : phrase.span[0]]
            if chunk:
                morphemes += jumanpp.apply_to_sentence(chunk, timeout=3000).morphemes
            sent = jumanpp.apply_to_sentence(phrase.text)
            phrase_to_morpheme_indices[phrase] = list(
                range(
                    len(morphemes),
                    len(morphemes) + len(sent.morphemes),
                )
            )
            morphemes += sent.morphemes
            cursor = phrase.span[1]
        if flickr_sentence.text[cursor:]:
            morphemes += jumanpp.apply_to_sentence(
                flickr_sentence.text[cursor:]
            ).morphemes
        knp_sentence = KNPSentence()
        knp_sentence.morphemes = morphemes
        knp_sentence.sent_id = f"{scenario_id}-{idx:02}"
        knp_sentence.doc_id = scenario_id
        knp_sentences.append(knp_sentence)
        global_morphemes.extend(morphemes)
        phrase_to_morpheme_global_indices.append(phrase_to_morpheme_indices)

    instance_ids: list[str] = []
    for idx, flickr_sentence in enumerate(flickr_document.sentences):
        for phrase in flickr_sentence.phrases:
            instance_id = str(phrase.phrase_id)
            if instance_id not in instance_ids:
                instance_ids.append(instance_id)

    knp_document = KNPDocument.from_sentences(knp_sentences)
    knp_document = knp.apply_to_document(knp_document)
    for morpheme in knp_document.morphemes:
        morpheme.semantics.clear()
        morpheme.semantics.nil = True
        morpheme.features.clear()
    for knp_phrase in knp_document.phrases:
        knp_phrase.features.clear()
    knp_dir.joinpath(f"{scenario_id}.knp").write_text(knp_document.to_knp())

    image_text_utterances: list[SentenceAnnotation] = []
    for idx, knp_sentence in enumerate(knp_document.sentences):
        phrases: list[PhraseAnnotation] = [
            PhraseAnnotation(text=base_phrase.text, relations=[])
            for base_phrase in knp_sentence.base_phrases
        ]
        # タグを構成する形態素を含む基本句のうち最も後ろにある基本句をタグとして採用
        for (
            phrase,
            morpheme_global_indices,
        ) in phrase_to_morpheme_global_indices[idx].items():
            last_morpheme = knp_sentence.morphemes[morpheme_global_indices[-1]]
            phrase_annotation = phrases[last_morpheme.base_phrase.index]
            phrase_annotation.relations.append(
                Phrase2ObjectRelation(type="=", instanceId=str(phrase.phrase_id))
            )
        image_text_utterances.append(
            SentenceAnnotation(
                sid=knp_sentence.sid,
                text=knp_sentence.text,
                phrases=phrases,
            )
        )

    image_text_annotation = ImageTextAnnotation(
        scenarioId=scenario_id,
        utterances=image_text_utterances,
        images=[
            ImageAnnotation(
                imageId=flickr_image_id,
                boundingBoxes=[
                    instance_id_to_bounding_box[instance_id]  # type: ignore
                    for instance_id in instance_ids
                    if instance_id_to_bounding_box.get(instance_id) is not None
                ],
            )
        ],
    )
    annotation_dir.joinpath(f"{scenario_id}.json").write_text(
        image_text_annotation.to_json(ensure_ascii=False, indent=2)
    )
    shutil.copy(
        flickr_image_dir / flickr_annotation.filename,
        image_dir / f"{flickr_image_id}.jpg",
    )

    return scenario_id


if __name__ == "__main__":
    main()
