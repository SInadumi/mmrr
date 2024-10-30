#!/usr/bin/env bash

set -euCo pipefail

readonly ROOT_DIR="${ROOT_DIR:-"./data"}"
readonly DETECTION_CONFIG="${DETECTION_CONFIG:-""}"

usage() {
    cat << _EOT_
Usage:
    ROOT_DIR=data/dataset $0
    DETECTION_CONFIG= Detic, RegionCLIP ... $1
Options:
    ROOT_DIR     path to input/output directory
    DETECTION_CONFIG    file name of the detection results (xxx.h5)
_EOT_
    exit 1
}

if [[ $# -gt 0 ]]; then
    usage
fi

if [[ -z "${ROOT_DIR}" ]]; then
    echo "missing required variable -- ROOT_DIR" >&2
    usage
fi

if [ -z "${DETECTION_CONFIG}" ]; then
    echo "missing required variable -- DETECTION_CONFIG" >&2
    usage
fi

echo "Calculate intersection over union (IoU) in J-CRe3 ..."
poetry run python -u ./scripts/visual/calc_iou_mapper.py \
    "${ROOT_DIR}/jcre3" --dataset-name jcre3 \
	--object-file-name $DETECTION_CONFIG

echo "Calculate intersection over union (IoU) in Flickr30k-Ent-Ja ..."
poetry run python -u ./scripts/visual/calc_iou_mapper.py \
    "${ROOT_DIR}/f30k_ent_jp" --dataset-name f30k_ent_jp \
    --object-file-name $DETECTION_CONFIG

echo "done!"
