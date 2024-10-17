#!/usr/bin/env bash

set -euCo pipefail

readonly JOBS="${JOBS:-"4"}"
readonly OUT_DIR="${OUT_DIR:-"./data"}"
readonly DATASET_ROOT="${DATASET_ROOT}"

usage() {
  cat << _EOT_
Usage:
  OUT_DIR=data/dataset [JOBS=4] $0

Options:
  OUT_DIR      path to output directory
  JOBS         number of jobs (default=1)
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

mkdir -p "${DATASET_ROOT}" "${OUT_DIR}"/{kwdlc,fuman,wac}

# echo "Processing KyotoCorpus ..."
# poetry run python ./scripts/build_dataset.py "${DATASET_ROOT}"/KyotoCorpus/knp "${OUT_DIR}/kc" \　# cohesion analysisでは未指定
#   --id "${DATASET_ROOT}/KyotoCorpus/id/full" \
#   -j "${JOBS}"

# echo "Processing KWDLC ..."
# poetry run python ./scripts/textual/build_dataset.py "${DATASET_ROOT}/KWDLC/knp" "${OUT_DIR}/kwdlc" \
#     --id "${DATASET_ROOT}/KWDLC/id/split_for_pas" \
#     -j "${JOBS}" \
#     --doc-id-format kwdlc

# echo "Processing AnnotatedFKCCorpus ..."
# poetry run python ./scripts/textual/build_dataset.py "${DATASET_ROOT}/AnnotatedFKCCorpus/knp" "${OUT_DIR}/fuman" \
#     --id "${DATASET_ROOT}/AnnotatedFKCCorpus/id/split_for_pas" \
#     -j "${JOBS}"

# echo "Processing WikipediaAnnotatedCorpus ..."
# poetry run python ./scripts/textual/build_dataset.py "${DATASET_ROOT}/WikipediaAnnotatedCorpus/knp" "${OUT_DIR}/wac" \
#     --id "${DATASET_ROOT}/WikipediaAnnotatedCorpus/id" \
#     -j "${JOBS}" --doc-id-format wac

# echo "Processing J-CRe3 ..."
# poetry run python ./scripts/textual/build_dataset.py "${OUT_DIR}/jcre3/knp" "${OUT_DIR}/jcre3" \
#     --id "${DATASET_ROOT}/J-CRe3/id" \
#     -j "${JOBS}"

echo "Processing Flickr30k-Ent-Ja ..."
poetry run python ./scripts/textual/build_dataset.py "${OUT_DIR}/f30k_ent_jp/knp" "${OUT_DIR}/f30k_ent_jp" \
    --id "${OUT_DIR}/f30k_ent_jp/id" \
    -j "${JOBS}"

echo "done!"
