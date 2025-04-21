import itertools
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from rhoknp import Document, Morpheme
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.file_utils import PaddingStrategy

OmegaConf.register_new_resolver("len", len, replace=True, use_cache=True)
POS_CLASSES = ['名詞', '指示詞', '接続詞', '接尾辞', '形容詞', '感動詞', '助詞', '副詞', '接頭辞', '動詞', '特殊']

def save_encoder(config_name: str, target_checkpoint: str, model_path: Path):
    with hydra.initialize(config_path="../../configs"):
        cfg = hydra.compose(config_name=config_name, overrides=[f"target_checkpoint={target_checkpoint}"])
    assert "load_from_checkpoint" in cfg.module.target_module
    target_module = hydra.utils.call(
        cfg.module.target_module.load_from_checkpoint, _recursive_=False
    )
    torch.save(target_module.encoder.state_dict(), model_path)

def run_ica(mat_X: np.ndarray) -> np.ndarray:
    # NOTE: https://github.com/shimo-lab/Universal-Geometry-with-ICA/blob/main/geometry/src/ica.py
    # bar_mat = np.mean(mat_X)
    # mat_c = mat_X - bar_mat     # centering
    # pca = PCA(whiten=False)
    # mat_pca1 = pca.fit_transform(mat_c)
    # mat_pca2 = mat_pca1 / np.std(mat_pca1, axis=0)  # whitening

    # # ica = FastICA(n_components=2, random_state=0, whiten=False, max_iter=10000, tol=1e-10)
    # mat_ica = ica.fit_transform(mat_pca2)

    ica = FastICA(n_components=2, random_state=0)
    mat_ica = ica.fit_transform(mat_X)
    return mat_ica

def run_tsne(mat_X: np.ndarray, perplexity: float=15.0) -> np.ndarray:
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    embeddings = tsne.fit_transform(mat_X)
    return embeddings

def make_embeddings(sentence_level_morphemes: list[list[Morpheme]], model_card: str, model_path: Optional[Path]=None) -> list[list[np.ndarray]]:
    config = AutoConfig.from_pretrained(model_card)
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    if model_path is not None:
        text_encoder = AutoModel.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
    else:
        text_encoder = AutoModel.from_pretrained(model_card, config=config)
    text_encoder.eval()

    sentence_level_embeddings = []
    for morphemes in sentence_level_morphemes:
        encoding = tokenizer(
            " ".join([morpheme.text for morpheme in morphemes]),
            is_split_into_words=False,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            return_tensors="pt"
        )
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze(0))

        embeddings = []
        with torch.no_grad():
            encoded = text_encoder(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                token_type_ids=encoding["token_type_ids"],
            ).last_hidden_state.squeeze(0) # (seq, hid)
            for token, token_level_embedding in zip(tokens, encoded):
                if "▁" in token:
                    # NOTE: 形態素の先頭トークンを基本句の代表として選択
                    embeddings.append(token_level_embedding.numpy())
        sentence_level_embeddings.append(embeddings)
    return sentence_level_embeddings


def visualize_points(output_path: Path, embedding: np.ndarray, morphemes: list[Morpheme]) -> None:
    import plotly
    fig = plotly.graph_objects.Figure()
    assert output_path.suffix == ".pdf"

    for pos in list(POS_CLASSES):
        plot_indices = []
        labels = []
        for idx, morpheme in enumerate(morphemes):
            if pos == morpheme.pos:
                plot_indices.append(idx)
                labels.append(morpheme.text)
        fig.add_trace(
            plotly.graph_objects.Scatter(
                x=embedding[plot_indices, 0],
                y=embedding[plot_indices, 1],
                mode="markers",
                name=f"{pos}",
            )
        )
    plotly.io.kaleido.scope.mathjax = None
    fig.write_image(output_path)

def visualize_tokens(output_path: Path, embedding: np.ndarray, morphemes: list[Morpheme]) -> None:
    import holoviews as hv
    hv.extension("plotly")
    assert output_path.suffix == ".html"

    points = hv.Points(embedding)
    labels = hv.Labels(
        {("x", "y"): embedding, "text": [morpheme.text for morpheme in morphemes]},
        ["x", "y"], "text"
    )
    fig = (points * labels).opts(
        hv.opts.Labels(xoffset=0.05, yoffset=0.05, size=14, padding=0.2, width=1500, height=1000),
        hv.opts.Points(color='black', marker='x', size=3),
    )
    hv.save(fig, output_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config-name", type=str, default="cohesion")
    parser.add_argument("--model-card", type=str, default="ku-nlp/deberta-v2-large-japanese")
    parser.add_argument("--target-checkpoint", type=str, help="Path to trained module checkpoint")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/jcre3"), help="path to annotation dir")
    parser.add_argument("--id-files", type=Path, nargs="+", default=[Path("data/jcre3/id/test.id")], help="Paths to scenario id file")
    parser.add_argument("--output-path", type=Path, default="./data/analysis/debug.pdf")
    args = parser.parse_args()

    scenario_ids: list[str] = list(itertools.chain.from_iterable(path.read_text().splitlines() for path in args.id_files))
    sentence_level_morphemes: list[list[Morpheme]] = []
    for scenario_id in scenario_ids:
        document = Document.from_knp((args.dataset_dir/ "knp" / f"{scenario_id}.knp").read_text())
        for sentence in document.sentences:
            morphemes = [
                phrase.morphemes[0] for phrase in sentence.base_phrases
            ] # NOTE: 基本句の先頭の形態素
            sentence_level_morphemes.append(morphemes)

    if args.target_checkpoint is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "encoder.pth"
            save_encoder(config_name=args.config_name, target_checkpoint=args.target_checkpoint, model_path=model_path)
            sentence_level_embeddings = make_embeddings(sentence_level_morphemes=sentence_level_morphemes, model_card=args.model_card, model_path=model_path)
    else:
        sentence_level_embeddings = make_embeddings(sentence_level_morphemes=sentence_level_morphemes, model_card=args.model_card)
    morphemes = list(itertools.chain.from_iterable(sentence_level_morphemes))
    embeddings = run_tsne(np.stack(list(itertools.chain.from_iterable(sentence_level_embeddings))))

    output_dir = args.output_path.parent
    output_dir.mkdir(exist_ok=True)
    visualize_tokens(args.output_path, embeddings, list(itertools.chain.from_iterable(sentence_level_morphemes)))


if __name__ == "__main__":
    main()
