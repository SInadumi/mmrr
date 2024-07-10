from enum import Enum


class Task(Enum):
    BRIDGING_REFERENCE_RESOLUTION = "bridging"
    COREFERENCE_RESOLUTION = "coreference"
    PAS_ANALYSIS = "pas"
    VIS_PAS_ANALYSIS = "vis_pas"
    VIS_COREFERENCE_RESOLUTION = "vis_coreference"
