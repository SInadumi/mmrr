import os
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser()
    parser.add_argument("INPUT", type=str, help="path to input visual annotation dir")
    parser.add_argument("OUTPUT", type=str, help="path to output dir")
    parser.add_argument("--id", type=str, help="path to id")

    args = parser.parse_args()

    visual_dir = Path(args.INPUT) / "visual_annotations"
    object_dir = Path(args.INPUT) / "region_features" / "regionclip_pretrained-cc_rn50"

    visual_paths = visual_dir.glob("*.json")
    object_paths = object_dir.glob("*.pth")

    output_root = Path(args.OUTPUT)
    vis_id2split = {}
    for id_file in Path(args.id).glob("*.id"):
        if id_file.stem not in {"train", "dev", "valid", "test"}:
            continue
        split = "valid" if id_file.stem == "dev" else id_file.stem
        output_root.joinpath(split).mkdir(parents=True, exist_ok=True)
        for vis_id in id_file.read_text().splitlines():
            vis_id2split[vis_id] = split

    for source in visual_paths:
        vis_id = source.stem
        target = output_root / vis_id2split[vis_id] / f"{vis_id}.json"
        print(f"cp {source} {target}")
        os.system(f"cp {source} {target}")

    for source in object_paths:
        obj_id = source.stem
        target = output_root / vis_id2split[obj_id] / f"{source.stem}.pth"
        print(f"cp {source} {target}")
        os.system(f"cp {source} {target}")


if __name__ == "__main__":
    main()
