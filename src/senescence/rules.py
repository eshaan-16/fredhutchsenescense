"""Panel logic: combining per-marker calls into one senescence call.

With n markers there are many ways to turn n booleans into one. The ones that
appear in the senescence literature, and which this module enumerates:

  AND over a subset        strict, e.g. p16+ AND LMNB1-
  OR over a subset         permissive, e.g. p21+ OR SA-bgal+
  m-of-n over a subset     the middle ground, e.g. 2 of 4

The reason to enumerate rather than pick: SenNet's own recommendation is that
no single marker defines senescence, and the field has no consensus panel
definition. Which combination you choose is a free parameter exactly like the
cutoff is, and it belongs in the same sensitivity analysis. Ashraf et al. 2023
sharpens this further, since marker intensity is graded and tracks duration of
cell-cycle withdrawal, so any binary panel rule is a lossy projection of a
continuous state. The point of scanning rule space is to find out how lossy.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Iterator, Literal, Sequence

import numpy as np

Logic = Literal["and", "or", "mofn"]


@dataclasses.dataclass(frozen=True)
class PanelRule:
    """A combination rule over named markers."""

    markers: tuple[str, ...]
    logic: Logic
    m: int = 1  # only meaningful for logic == "mofn"

    def __post_init__(self) -> None:
        if not self.markers:
            raise ValueError("panel rule needs at least one marker")
        if self.logic == "mofn" and not 1 <= self.m <= len(self.markers):
            raise ValueError(
                f"m-of-n rule needs 1 <= m <= {len(self.markers)}, got m={self.m}"
            )

    def __str__(self) -> str:
        if len(self.markers) == 1:
            return self.markers[0]
        if self.logic == "and":
            return " AND ".join(self.markers)
        if self.logic == "or":
            return " OR ".join(self.markers)
        return f"{self.m}-of-{len(self.markers)}[{','.join(self.markers)}]"

    @property
    def n_markers(self) -> int:
        return len(self.markers)

    def apply(self, calls: dict[str, np.ndarray]) -> np.ndarray:
        """Collapse per-marker boolean arrays into one boolean array."""
        missing = [m for m in self.markers if m not in calls]
        if missing:
            raise KeyError(f"rule {self} needs markers not in calls: {missing}")
        stack = np.vstack([np.asarray(calls[m], dtype=bool) for m in self.markers])
        if self.logic == "and":
            return stack.all(axis=0)
        if self.logic == "or":
            return stack.any(axis=0)
        return stack.sum(axis=0) >= self.m


def enumerate_rules(
    markers: Sequence[str],
    max_subset: int | None = None,
    include_singletons: bool = True,
) -> list[PanelRule]:
    """Every distinct panel rule over subsets of `markers`.

    Deduplicated: for a subset of size n, AND is the same as n-of-n and OR is
    the same as 1-of-n, so only the m-of-n form is emitted for 1 < m < n.
    """
    markers = tuple(markers)
    hi = len(markers) if max_subset is None else min(max_subset, len(markers))
    out: list[PanelRule] = []

    for size in range(1, hi + 1):
        if size == 1 and not include_singletons:
            continue
        for subset in itertools.combinations(markers, size):
            if size == 1:
                out.append(PanelRule(subset, "and"))
                continue
            out.append(PanelRule(subset, "and"))  # == n-of-n
            out.append(PanelRule(subset, "or"))   # == 1-of-n
            for m in range(2, size):
                out.append(PanelRule(subset, "mofn", m))
    return out


def sample_rule(
    markers: Sequence[str],
    rng: np.random.Generator,
    max_subset: int | None = None,
) -> PanelRule:
    """Draw one panel rule, roughly uniformly over the enumerated space."""
    pool = enumerate_rules(markers, max_subset=max_subset)
    return pool[int(rng.integers(len(pool)))]


def iter_rules(
    markers: Sequence[str], max_subset: int | None = None
) -> Iterator[PanelRule]:
    yield from enumerate_rules(markers, max_subset=max_subset)
