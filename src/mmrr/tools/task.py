from enum import Enum


class Task(Enum):
    BRIDGING_REFERENCE_RESOLUTION = "bridging"
    COREFERENCE_RESOLUTION = "coreference"
    PAS_ANALYSIS = "pas"
    MM_PAS_ANALYSIS = "mm_pas"
    MM_COREFERENCE_RESOLUTION = "mm_coreference"
