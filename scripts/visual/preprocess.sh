#!/usr/bin/env bash

set -euCo pipefail

readonly OUT_DIR="${OUT_DIR:-"./data"}"
readonly DATASET_ROOT="${DATASET_ROOT}"

usage() {
    cat << _EOT_
Usage:
    OUT_DIR=data/dataset $0

Options:
    OUT_DIR     path to output directory
_EOT_
    exit 1
}

if [[ $# -gt 0 ]]; then
    usage
fi

if [[ -z "${OUT_DIR}" ]]; then
    echo "missing required variable -- OUT_DIR" >&2
    usage
fi

mkdir -p "${OUT_DIR}"/{f30k_ent_jp,jcre3}
ln -s "${DATASET_ROOT}/J-CRe3/id" "${OUT_DIR}/jcre3/id"
ln -s "${DATASET_ROOT}/J-CRe3/recording" "${OUT_DIR}/jcre3/recording"
ln -s "${DATASET_ROOT}/J-CRe3/visual_annotations" "${OUT_DIR}/jcre3/image_text_annotation"
ln -s "${DATASET_ROOT}/J-CRe3/textual_annotations" "${OUT_DIR}/jcre3/knp"

for split in "train" "val" "test"
do
    echo "Converting Flickr30k-Ent-Ja ${split} Annotations ..."
    poetry run python ./scripts/visual/convert_flickr_to_annotation.py \
        --flickr-id-file "${DATASET_ROOT}/flickr30k_entities/${split}.txt" \
        --flickr-image-dir "${DATASET_ROOT}/Flickr30kEnt-JP/flickr30k_images" \
        --flickr-annotations-dir "${DATASET_ROOT}/flickr30k_entities/Annotations" \
        --flickr-documents-dir "${DATASET_ROOT}/Flickr30kEnt-JP/Sentences_jp_v2" \
        --output-dir "${OUT_DIR}/f30k_ent_jp"
done
