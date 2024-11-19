from collections import defaultdict
from collections.abc import Mapping
from functools import reduce
from operator import add
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from rhoknp import Document

from cl_mmref.datasets.cohesion_dataset import CohesionDataset
from cl_mmref.tools.evaluators.cohesion import CohesionEvaluator, CohesionScore


@torch.no_grad()
def initialize_parameters(
    target: pl.LightningDataModule, source: pl.LightningDataModule
) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.copy_(sp)


@rank_zero_only
def save_results(results: list[Mapping[str, float]], save_dir: Path) -> None:
    test_results: dict[str, dict[str, float]] = defaultdict(dict)
    for k, v in [item for result in results for item in result.items()]:
        met, corpus = k.split("/")
        if met in test_results[corpus]:
            assert v == test_results[corpus][met]
        else:
            test_results[corpus][met] = v

    save_dir.mkdir(exist_ok=True, parents=True)
    for corpus, result in test_results.items():
        with save_dir.joinpath(f"{corpus}.csv").open(mode="wt") as f:
            f.write(",".join(result.keys()) + "\n")
            f.write(",".join(f"{v:.6}" for v in result.values()) + "\n")


@rank_zero_only
def save_prediction(datasets: dict[str, CohesionDataset], pred_dir: Path) -> None:
    all_results = []
    for corpus, dataset in datasets.items():
        predicted_documents: list[Document] = []
        for path in pred_dir.joinpath(f"knp_{corpus}").glob("*.knp"):
            predicted_documents.append(Document.from_knp(path.read_text()))
        evaluator = CohesionEvaluator(
            tasks=dataset.tasks,
            exophora_referent_types=[e.type for e in dataset.exophora_referents],
            pas_cases=dataset.cases,
            bridging_rel_types=dataset.bar_rels,
        )
        evaluator.coreference_evaluator.is_target_mention = (
            lambda mention: mention.features.get("体言") is True
        )
        score_result: CohesionScore = evaluator.run(
            gold_documents=dataset.orig_documents,
            predicted_documents=predicted_documents,
        )
        score_result.export_csv(pred_dir / f"{corpus}.csv")
        score_result.export_txt(pred_dir / f"{corpus}.txt")
        all_results.append(score_result)
    score_result = reduce(add, all_results)
    score_result.export_csv(pred_dir / "all.csv")
    score_result.export_txt(pred_dir / "all.txt")
