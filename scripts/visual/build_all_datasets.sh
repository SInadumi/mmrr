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

mkdir -p "${DATASET_ROOT}" "${OUT_DIR}"/{f30k_ent_jp,jcre3}

echo "Processing J-CRe3 ..."
poetry run python ./scripts/visual/build_dataset.py \
    "${DATASET_ROOT}/J-CRe3" "${OUT_DIR}/jcre3" \
    --dataset-name jcre3 --id "${DATASET_ROOT}/J-CRe3/id" \
    --num-utterances-per-sample 3 --num-overlapping-utterances 2

echo "Processing Flickr30k-Ent-Ja ..."
poetry run python ./scripts/visual/build_dataset.py \
    "${OUT_DIR}/f30k_ent_jp" "${OUT_DIR}/f30k_ent_jp" \
    --dataset-name f30k_ent_jp --id "${OUT_DIR}/f30k_ent_jp/id" \
    --num-utterances-per-sample 1 --num-overlapping-utterances 0

echo "done!"
