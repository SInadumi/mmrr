from enum import Enum


class Task(Enum):
    BRIDGING_REFERENCE_RESOLUTION = "bridging"
    COREFERENCE_RESOLUTION = "coreference"
    PAS_ANALYSIS = "pas"
    PHRASE_GROUNDING = "phrase_grounding"
    VIS_PAS_ANALYSIS = "vis_pas"
