import copy
from typing import Union

from rhoknp import BasePhrase, Document
from rhoknp.cohesion import (
    Argument,
    EndophoraArgument,
    ExophoraArgument,
    ExophoraReferent,
)

from mmrr.tools.extractors import (
    BridgingExtractor,
    CoreferenceExtractor,
    PasExtractor,
)
from mmrr.tools.extractors.base import BaseExtractor
from mmrr.tools.task import Task
from mmrr.utils.dataset import CohesionBasePhrase

from .base import BaseExample


class KyotoExample(BaseExample):
    def __init__(self) -> None:
        super().__init__()
        self.phrases: dict[Task, list[CohesionBasePhrase]] = {}
        self.sid_to_type_id: dict[str, int] = {}

    def load(
        self,
        document: Document,
        tasks: list[Task],
        task_to_extractor: dict[Task, BaseExtractor],
        task_to_rels: dict[Task, list[str]],
        sid_to_type_id: dict[str, int],
        flip_writer_reader_according_to_type_id: bool,
    ):
        self.set_knp_params(document)
        self.sid_to_type_id = sid_to_type_id
        for task in tasks:
            extractor: BaseExtractor = task_to_extractor[task]
            self.phrases[task] = self._wrap_base_phrases(
                document.base_phrases,
                extractor,
                task_to_rels[task],
                flip_writer_reader_according_to_type_id,
            )

    def _wrap_base_phrases(
        self,
        base_phrases: list[BasePhrase],
        extractor: BaseExtractor,
        rel_types: list[str],
        flip_writer_reader_according_to_type_id: bool,
    ) -> list[CohesionBasePhrase]:
        cohesion_base_phrases = [
            CohesionBasePhrase(
                base_phrase.head.global_index,
                [morpheme.global_index for morpheme in base_phrase.morphemes],
                [morpheme.text for morpheme in base_phrase.morphemes],
                is_target=extractor.is_target(base_phrase),
                referent_candidates=[],
            )
            for base_phrase in base_phrases
        ]

        for base_phrase, cohesion_base_phrase in zip(
            base_phrases, cohesion_base_phrases
        ):
            all_rels = extractor.extract_rels(base_phrase)
            if isinstance(extractor, (PasExtractor, BridgingExtractor)):
                assert isinstance(all_rels, dict)
                rel2tags = {
                    rel_type: _get_argument_tags(all_rels[rel_type])
                    for rel_type in rel_types
                }
            elif isinstance(extractor, CoreferenceExtractor):
                assert rel_types == ["="]
                assert isinstance(all_rels, list)
                rel2tags = {"=": _get_referent_tags(all_rels)}
            else:
                raise AssertionError

            # flip reader-writer tags for jcre3
            if (
                flip_writer_reader_according_to_type_id is True
                and self.sid_to_type_id.get(base_phrase.sentence.sid) == 1
            ):
                flip_map = {"[著者]": "[読者]", "[読者]": "[著者]"}
                rel2tags = {
                    rel_type: [flip_map.get(s, s) for s in tags]
                    for rel_type, tags in rel2tags.items()
                }

            # set parameters
            cohesion_base_phrase.rel2tags = rel2tags
            referent_candidates = extractor.get_candidates(
                base_phrase, base_phrase.document.base_phrases
            )
            cohesion_base_phrase.referent_candidates = [
                cohesion_base_phrases[cand.global_index] for cand in referent_candidates
            ]
        return cohesion_base_phrases


def _get_argument_tags(arguments: list[Argument]) -> list[str]:
    """Get argument tags.

    Note:
        endophora argument: string of base phrase global index
        exophora argument: exophora referent
        no argument: [NULL]
    """
    argument_tags: list[str] = []
    for argument in arguments:
        if isinstance(argument, EndophoraArgument):
            argument_tag = str(argument.base_phrase.global_index)
        else:
            assert isinstance(argument, ExophoraArgument)
            exophora_referent = copy.copy(argument.exophora_referent)
            exophora_referent.index = None  # 不特定:人１ -> 不特定:人
            argument_tag = f"[{exophora_referent.text}]"  # 不特定:人 -> [不特定:人]
        argument_tags.append(argument_tag)
    return argument_tags or ["[NULL]"]


def _get_referent_tags(
    referents: list[Union[BasePhrase, ExophoraReferent]],
) -> list[str]:
    """Get referent tags.

    Note:
        endophora referent: string of base phrase global index
        exophora referent: exophora referent text wrapped by []
        no referent: [NA]
    """
    mention_tags: list[str] = []
    for referent in referents:
        if isinstance(referent, BasePhrase):
            mention_tag = str(referent.global_index)
        else:
            assert isinstance(referent, ExophoraReferent)
            referent.index = None  # 不特定:人１ -> 不特定:人
            mention_tag = f"[{referent.text}]"  # 著者 -> [著者]
        mention_tags.append(mention_tag)
    return mention_tags or ["[NA]"]
