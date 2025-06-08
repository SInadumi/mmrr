# A Framework for Textual and Multimodal Reference Resolution
[![arXiv](https://img.shields.io/badge/arXiv-2505.11726-b31b1b.svg)](https://arxiv.org/abs/2505.11726)

This repository contains experiments code for textual reference resolution (i.e. cohesion analysis) and multimodal reference resolution, introduced by the following paper:

**Disambiguating Reference in Visually Grounded Dialogues through Joint Modeling of Textual and Multimodal Semantic Structures**, Shun Inadumi, Nobuhiro Ueda, and Koichiro Yoshino, In ACL2025.


## Requirements
- Python: >= 3.9, < 3.11
- Python dependencies: See [pyproject.toml](/pyproject.toml)
- [Juman++](https://github.com/ku-nlp/jumanpp): 2.0.0-rc3+
- [KNP](https://github.com/ku-nlp/knp): 5.0+
- [SInadumi/Detic](https://github.com/SInadumi/Detic)
- [SInadumi/RegionCLIP](https://github.com/SInadumi/RegionCLIP) (optional)
- [SInadumi/multimodal-reference](https://github.com/SInadumi/multimodal-reference) (optional)

## Setup Environment
1. Create a virtual environment and install dependencies
```bash
# Create a virtual environment and install
pyenv global [python >=3.9,<3.11]
poetry env use path/to/python
poetry install

# Log in wandb (optional)
wandb login

# Setup pre-commit (optional)
# pipx install pre-commit
pre-commit install
```

## Dataset Preparation
### Download annotations
```bash
export DATASET_ROOT=/path/to/input/dir
git clone --depth 1 git@github.com:ku-nlp/KWDLC.git "${DATASET_ROOT}/KWDLC"
git clone --depth 1 git@github.com:ku-nlp/AnnotatedFKCCorpus.git "${DATASET_ROOT}/AnnotatedFKCCorpus"
git clone --depth 1 git@github.com:ku-nlp/WikipediaAnnotatedCorpus.git "${DATASET_ROOT}/WikipediaAnnotatedCorpus"
git clone --depth 1 git@github.com:riken-grp/J-CRe3.git "${DATASET_ROOT}/J-CRe3"
git clone --depth 1 git@github.com:nlab-mpg/Flickr30kEnt-JP.git "${DATASET_ROOT}/Flickr30kEnt-JP"
git clone --depth 1 git@github.com:BryanPlummer/flickr30k_entities.git "${DATASET_ROOT}/flickr30k_entities"
```

<!-- > [!TIP]
> If the full J-CRe3 dataset including videos and audio is not required, you may instead download the subset used in this study, which contains only the image folder. This subset can be downloaded from [here](). -->

### Construct annotations
```bash
# JOBS: the number of jobs (default=1)
# OUT_DIR: path to output dir (default="./data")

# construct f30k_ent_jp annotations
[OUT_DIR=/path/to/output/dir] bash ./scripts/visual/preprocess.sh

# build annotations
[OUT_DIR=...] [JOBS=1] bash ./scripts/textual/build_all_datasets.sh
[OUT_DIR=...] bash ./scripts/visual/build_all_datasets.sh
```

### Extract region features
First, clone the detection code from [SInadumi/Detic](https://github.com/SInadumi/Detic) and run the following script to detect objects.
```bash
poetry install
bash extract_region_features_jcre3.sh
```
Then, run the following script to post-process the detection results:
```bash
# ROOT_DIR: path to input/output dir (default="./data")
# DETECTION_CONFIG: file name of the detection results (e.g. Detic, RegionCLIP, ...). This is required argument.

# calulate IoU between gold and predicted bounding boxes
[ROOT_DIR=...] DETECTION_CONFIG=... bash ./scripts/visual/postprocess.sh
```

<!-- > [!TIP]
>  -->

## Training
```bash
# cohesion analysis
poetry run python scripts/train.py -cn cohesion devices=[0,1] max_batches_per_device=4 effective_batch_size=16
# multi-modal reference resolution
poetry run python scripts/train.py -cn mmref object_file_name=`file_name` devices=[0,1] max_batches_per_device=4 effective_batch_size=16
```
These are commonly used options to train cohesion model:
- `-cn`: Config name
- `devices`: GPUs to use (default: `0`)
- `max_batches_per_device`: Maximum number of batches to process per device (default: `4`)
- `model_name_or_path`: Path to a pre-trained model or model identifier from the [Huggingface Hub](https://huggingface.co/models) (default: `ku-nlp/deberta-v2-base-japanese`)

These are additional necessary options for train mmref models:
- `object_file_name`: File name of the detection results (`object_file_name`.h5) and IoU results (`object_file_name`_iou_mapper.h5)
- `source_checkpoint`: Path to the checkpoint of the trained cohesion model

For more options, see [cohesion configs](./configs/cohesion.yaml) and [mmref configs](./configs/mmref.yaml).
## Testing
### Cohesion Analysis
```bash
poetry run python scripts/test_cohesion.py checkpoint=/path/to/trained/checkpoint eval_set=test devices=[0,1]
```
### Multimodal Reference Resolution for J-CRe3
Clone [SInadumi/multimodal-reference](https://github.com/SInadumi/multimodal-reference) and see `Run Prediction` and `Run Evaluation`.

## Debugging
This is an example of the F30k-ent-jp preparation.
```bash
poetry run python -m pdb ./scripts/visual/convert_flickr_to_annotation.py \
  --flickr-id-file="${DATASET_ROOT}/flickr30k_entities/train.txt" \
  --flickr-image-dir="${DATASET_ROOT}/Flickr30kEnt-JP/flickr30k_images" \
  --flickr-annotations-dir="${DATASET_ROOT}/flickr30k_entities/Annotations" \
  --flickr-documents-dir="${DATASET_ROOT}/Flickr30kEnt-JP/Sentences_jp_v2" \
  "--output-dir=./data/f30k_ent_jp"
```
These are examples of the workflow from J-CRe3 preparation to training.
```bash
# dataset preparation
poetry run python -m pdb ./scripts/textual/build_textual.py ./data/jcre3/knp ./data/jcre3 \
  --id="${DATASET_ROOT}/J-CRe3/id" -j=1
poetry run python -m pdb ./scripts/visual/build_visual.py ./data/jcre3 \
  --dataset-name=jcre3 --id="${DATASET_ROOT}/J-CRe3/id"

# The following code examples are based on the assumption that object detection has already been performed.
poetry run python -m pdb ./scripts/visual/calc_iou_mapper.py ./data/jcre3 \
  --dataset-name=jcre3 --object-file-name=`file_name`

# training
poetry run python -m pdb -cn cohesion_debug devices=[0]
poetry run python -m pdb -cn mmref_debug object_file_name=`file_name` devices=[0]
```


## Environment Variables
- `CACHE_DIR`: A directory where processed annotations are cached. Default values are; `/tmp/{os.environ["USER"]}/cohesion_cache` for cohesion analysis and `/tmp/{os.environ["USER"]}/mmref_cache` for mmref.
- `OVERWRITE_CACHE`: If set, the data loader does not load cache even if it exists.
- `DISABLE_CACHE`: If set, the data loader does not load or save cache.

## Acknowledgement
The code is based on a [Japanese cohesion analyzer](https://github.com/nobu-g/cohesion-analysis) licensed under [MIT License](https://github.com/nobu-g/cohesion-analysis/blob/main/LICENSE).

## Citation
TBA
