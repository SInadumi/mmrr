### Dataset Preparation
- Dataset download
```
export DATASET_ROOT=/path/to/input/dir
git clone --depth 1 git@github.com:ku-nlp/KyotoCorpus.git "${DATASET_ROOT}/KyotoCorpus" # cohesion analysisでは未指定
git clone --depth 1 git@github.com:ku-nlp/KWDLC.git "${DATASET_ROOT}/KWDLC"
git clone --depth 1 git@github.com:ku-nlp/AnnotatedFKCCorpus.git "${DATASET_ROOT}/AnnotatedFKCCorpus"
git clone --depth 1 git@github.com:ku-nlp/WikipediaAnnotatedCorpus.git "${DATASET_ROOT}/WikipediaAnnotatedCorpus"
git clone --depth 1 git@github.com:riken-grp/J-CRe3.git "${DATASET_ROOT}/J-CRe3"
```
- Extract region features
See [SInadumi/RegionCLIP](https://github.com/SInadumi/RegionCLIP).

### Construct & Split annotations

```
# JOBS: the number of jobs (default=1)
# OUT_DIR: output dir (default="./data")

# textual annotations
[OUT_DIR=/path/to/output/dir] [JOBS=4] bash ./scripts/textual/build_all_datasets.sh

# visual_annotations
[OUT_DIR=/path/to/output/dir] bash ./scripts/visual/build_all_datasets.sh
```

### Acknowledgement
- [Copyright (c) 2020, Nobuhiro Ueda](https://github.com/nobu-g/cohesion-analysis)
- [Copyright (c) 2024, Nobuhiro Ueda](https://github.com/riken-grp/multimodal-reference)
