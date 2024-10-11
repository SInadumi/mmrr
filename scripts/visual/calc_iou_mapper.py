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
    parser.add_argument("--split", type=str, choices=["train", "valid", "test"])
    args = parser.parse_args()

    dataset_dir = Path(args.ROOT_DIR) / args.split
    object_fp = h5py.File(Path(args.ROOT_DIR) / f"{args.object_file_name}.h5", mode="r")
    output_fp = h5py.File(
        Path(args.ROOT_DIR) / f"{args.object_file_name}_iou_mapper.h5", mode="w"
    )

    visual_paths = dataset_dir.glob("*.json")
    for source in visual_paths:
        image_text_annotation = ImageTextAnnotation.from_json(Path(source).read_text())
        assert len(image_text_annotation.images) == 1
        scenario_id = image_text_annotation.scenarioId
        image_id = image_text_annotation.images[0].imageId
        gold_bboxes: list[BoundingBox] = image_text_annotation.images[0].boundingBoxes
        predict_bboxes: list[np.ndarray] = list(
            object_fp[f"{scenario_id}/{image_id}/boxes"]
        )
        for gold_bbox in gold_bboxes:
            for idx, pred_bbox in enumerate(predict_bboxes):
                _pbb = Rectangle(
                    x1=pred_bbox[0], y1=pred_bbox[1], x2=pred_bbox[2], y2=pred_bbox[3]
                )
                try:
                    output_fp.create_dataset(
                        f"{scenario_id}/{image_id}/{gold_bbox.instanceId}/{idx}",
                        data=box_iou(gold_bbox.rect, _pbb),
                    )
                except Exception as e:
                    logger.warning(
                        f"{type(e).__name__}: {e}, {scenario_id}/{image_id}/{gold_bbox.instanceId}/{idx}"
                    )
    object_fp.close()
    output_fp.close()


if __name__ == "__main__":
    main()
