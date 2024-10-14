import sys
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np

from cl_mmref.utils.annotation import (
    BoundingBox,
    ImageTextAnnotation,
)
from cl_mmref.utils.util import Rectangle, box_iou

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def scenario_id_to_iid(scenario_id: str) -> str:
    return scenario_id.split("-")[-1]


def calc_bbox_iou(paths: list[Path], input_fp: h5py.File, output_fp: h5py.File) -> None:
    for source in paths:
        raw_annot = json.load(open(source, "r", encoding="utf-8"))   # for faster loading
        image_text_annotation = ImageTextAnnotation(**raw_annot)
        assert len(image_text_annotation.images) == 1
        scenario_id = image_text_annotation.scenarioId
        image_id = image_text_annotation.images[0].imageId
        gold_bboxes: list[BoundingBox] = image_text_annotation.images[0].boundingBoxes
        predict_bboxes: list[np.ndarray] = list(
            input_fp[f"{scenario_id}/{image_id}/boxes"]
        )
        for gold_bbox in gold_bboxes:
            iou_values = []
            for idx, pred_bbox in enumerate(predict_bboxes):
                _pbb = Rectangle(
                    x1=pred_bbox[0], y1=pred_bbox[1], x2=pred_bbox[2], y2=pred_bbox[3]
                )
                iou_values.append(box_iou(gold_bbox.rect, _pbb))
            try:
                output_fp.create_dataset(
                    f"{scenario_id}/{image_id}/{gold_bbox.instanceId}",
                    data=iou_values,
                )
            except ValueError as e:
                logger.warning(
                    f"{type(e).__name__}: {e}, Skipping {scenario_id}/{image_id}/{gold_bbox.instanceId}"
                )
    return


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "ROOT_DIR",
        type=str,
        help="path to input/output annotation dir (current project)",
    )
    parser.add_argument("--dataset-name", type=str, choices=["jcre3", "f30k_ent_jp"])
    parser.add_argument(
        "--object-file-name", type=str, help="name of a hdf5 input file"
    )
    args = parser.parse_args()

    if args.dataset_name == "jcre3":
        dataset_dirs = [Path(args.ROOT_DIR) / split for split in ["train", "valid", "test"]]
        visual_paths = [fp for _dir in dataset_dirs for fp in _dir.glob("*.json")]
    elif args.dataset_name == "f30k_ent_jp":
        visual_paths = [fp for fp in (Path(args.ROOT_DIR) / "visual_annotations").glob("*.json")]
    else:
        raise NotImplementedError

    input_fp = h5py.File(Path(args.ROOT_DIR) / f"{args.object_file_name}.h5", mode="r")
    output_fp = h5py.File(
        Path(args.ROOT_DIR) / f"{args.object_file_name}_iou_mapper.h5", mode="w"
    )
    try:
        calc_bbox_iou(visual_paths, input_fp, output_fp)
    except Exception as e:
        logger.error(f"{type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        input_fp.close()
        output_fp.close()


if __name__ == "__main__":
    main()
