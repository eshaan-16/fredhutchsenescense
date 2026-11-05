"""Cutoff rules for calling a single marker positive.

Every rule is calibrated on the NEGATIVE control distribution only. That is the
deliberate choice: a threshold derived from the positive control bakes the
expected effect into the measurement, and then the positive/negative separation
you report is partly an artifact of how you drew the line.

Three families, all expressed as a single scalar `k` so Monte Carlo can sweep
them on a common axis:

  sd(k)          control mean + k * control SD          k ~ [1, 3]
  percentile(k)  k-th percentile of the control         k ~ [90, 99]
  mad(k)         control median + k * 1.4826 * MAD      k ~ [1, 3]

`mad` is the robust counterpart of `sd`. Immunofluorescence negative controls
routinely carry a handful of very bright cells (debris, mitotic figures,
antibody aggregates), and those few points inflate the SD enough to pull a
mean+2SD cutoff well above where it belongs. When `sd` and `mad` disagree in
the sweep, that disagreement is itself a finding worth reporting.

A "down" marker (LMNB1, Ki67) is called positive-for-senescence in its LOWER
tail, so the threshold is mirrored to the opposite side of the control
distribution and the comparison operator flips.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Literal

import numpy as np

from .panel import Marker

RuleFamily = Literal["sd", "percentile", "mad"]

_MAD_TO_SD = 1.4826  # makes MAD a consistent estimator of SD under normality


@dataclasses.dataclass(frozen=True)
class ThresholdRule:
    """A cutoff rule: a family plus its scalar parameter."""

    family: RuleFamily
    k: float

    def __str__(self) -> str:
        return f"{self.family}({self.k:g})"

    def cutoff(self, control: np.ndarray, direction: str = "up") -> float:
        """Compute the numeric cutoff from a negative-control sample."""
        x = np.asarray(control, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            raise ValueError(
                f"need at least 2 finite control values to set a {self} "
                f"cutoff, got {x.size}"
            )
        lower = direction == "down"

        if self.family == "sd":
            mu, sd = x.mean(), x.std(ddof=1)
            return mu - self.k * sd if lower else mu + self.k * sd

        if self.family == "mad":
            med = np.median(x)
            mad = np.median(np.abs(x - med)) * _MAD_TO_SD
            return med - self.k * mad if lower else med + self.k * mad

        if self.family == "percentile":
            if not 0 < self.k < 100:
                raise ValueError(
                    f"percentile k must be in (0, 100), got {self.k}"
                )
            q = 100.0 - self.k if lower else self.k
            return float(np.percentile(x, q))

        raise ValueError(f"unknown rule family {self.family!r}")


def call_positive(
    values: np.ndarray,
    cutoff: float,
    direction: str = "up",
) -> np.ndarray:
    """Boolean senescence-positive call for one marker.

    Non-finite values return False rather than propagating NaN, so a missing
    channel in one cell does not silently drop that cell from the denominator.
    """
    v = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        hit = v <= cutoff if direction == "down" else v >= cutoff
    return np.where(np.isfinite(v), hit, False)


def marker_calls(
    values: np.ndarray,
    control: np.ndarray,
    rule: ThresholdRule,
    marker: Marker,
) -> tuple[np.ndarray, float]:
    """Convenience: cutoff from control, then calls on values.

    Returns (calls, cutoff) so the cutoff can be logged alongside the result.
    An analysis that reports percent-senescent without reporting the cutoff
    that produced it is not reproducible.
    """
    cut = rule.cutoff(control, marker.direction)
    return call_positive(values, cut, marker.direction), cut


# ---- sweep grids --------------------------------------------------------

def default_grid(family: RuleFamily, n: int = 21) -> list[ThresholdRule]:
    """The plausible range of k for each family, as a sweep grid."""
    if family in ("sd", "mad"):
        ks = np.linspace(1.0, 3.0, n)
    elif family == "percentile":
        ks = np.linspace(90.0, 99.0, n)
    else:
        raise ValueError(f"unknown rule family {family!r}")
    return [ThresholdRule(family, float(k)) for k in ks]


def all_families_grid(n: int = 21) -> list[ThresholdRule]:
    """Every family's grid concatenated, for a cross-family sweep."""
    out: list[ThresholdRule] = []
    for fam in ("sd", "mad", "percentile"):
        out.extend(default_grid(fam, n))  # type: ignore[arg-type]
    return out


def sample_rule(rng: np.random.Generator) -> ThresholdRule:
    """Draw one rule uniformly over family, then uniformly over that family's k.

    Uniform-over-family is the honest prior when the lab has not committed to a
    thresholding convention. If Dey's group does commit to one, restrict this
    and the sweep narrows accordingly.
    """
    fam: RuleFamily = rng.choice(["sd", "mad", "percentile"])  # type: ignore
    k = rng.uniform(90.0, 99.0) if fam == "percentile" else rng.uniform(1.0, 3.0)
    return ThresholdRule(fam, float(k))
