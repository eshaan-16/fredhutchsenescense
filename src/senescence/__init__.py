"""Monte Carlo threshold analysis for senescence biomarkers.

Dey Lab, Fred Hutchinson Cancer Center.

The question this package exists to answer: when a biomarker intensity is
turned into a binary senescent/not-senescent call, how much of the resulting
number is the biology and how much is the threshold?
"""

from .panel import Panel, Marker, load_panel
from .thresholds import ThresholdRule, call_positive
from .rules import PanelRule, enumerate_rules

__all__ = [
    "Panel",
    "Marker",
    "load_panel",
    "ThresholdRule",
    "call_positive",
    "PanelRule",
    "enumerate_rules",
]

__version__ = "0.1.0"
