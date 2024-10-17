# Dataset Preparation
## Download annotations
```
export DATASET_ROOT=/path/to/input/dir
git clone --depth 1 git@github.com:ku-nlp/KWDLC.git "${DATASET_ROOT}/KWDLC"
git clone --depth 1 git@github.com:ku-nlp/AnnotatedFKCCorpus.git "${DATASET_ROOT}/AnnotatedFKCCorpus"
git clone --depth 1 git@github.com:ku-nlp/WikipediaAnnotatedCorpus.git "${DATASET_ROOT}/WikipediaAnnotatedCorpus"
git clone --depth 1 git@github.com:riken-grp/J-CRe3.git "${DATASET_ROOT}/J-CRe3"
git clone --depth 1 git@github.com:nlab-mpg/Flickr30kEnt-JP.git "${DATASET_ROOT}/Flickr30kEnt-JP"
git clone --depth 1 git@github.com:BryanPlummer/flickr30k_entities.git "${DATASET_ROOT}/flickr30k_entities"
```

## Construct annotations
```
# JOBS: the number of jobs (default=1)
# OUT_DIR: path to output dir (default="./data")

# construct f30k_ent_jp annotations
[OUT_DIR=/path/to/output/dir] bash ./scripts/visual/preprocess.sh

# build annotations
[OUT_DIR=...] [JOBS=1] bash ./scripts/textual/build_all_datasets.sh
[OUT_DIR=...] bash ./scripts/visual/build_all_datasets.sh
```

## Extract region features
See [SInadumi/RegionCLIP](https://github.com/SInadumi/RegionCLIP).
```
# calulate IoU between gold and predicted bounding boxes
[OUT_DIR=...] bash ./scripts/visual/postprocess.sh
```

### Acknowledgement
- [Copyright (c) 2020, Nobuhiro Ueda](https://github.com/nobu-g/cohesion-analysis)
- [Copyright (c) 2024, Nobuhiro Ueda](https://github.com/riken-grp/multimodal-reference)
