#!/usr/bin/env bash

set -euCo pipefail

readonly ROOT_DIR="${ROOT_DIR:-"./data"}"

usage() {
    cat << _EOT_
Usage:
    ROOT_DIR=data/dataset $0

Options:
    ROOT_DIR     path to input/output directory
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

for split in "train" "valid" "test"
do
    echo "Calculate intersection over union (IoU) in J-CRe3 ${split} split ..."
    poetry run python ./scripts/visual/calc_iou_mapper.py \
        "${ROOT_DIR}/jcre3" --dataset-name jcre3 \
		--object-file-name CLIP_fast_rcnn_R_50_C4_zsinf_w_GT \
		--split ${split}

	echo "Calculate intersection over union (IoU) in Flickr30k-Ent-Ja ${split} split ..."
    poetry run python ./scripts/visual/calc_iou_mapper.py \
        "${ROOT_DIR}/f30k_ent_jp" --dataset-name f30k_ent_jp \
		--object-file-name CLIP_fast_rcnn_R_50_C4_zsinf \
		--split ${split}
done

echo "done!"
