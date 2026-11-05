"""Scoring a senescence call against the controls.

The quantity the lab cares about is percent-senescent. The quantity that tells
you whether a threshold is any good is how cleanly it separates the positive
control from the negative control. These are different numbers and this module
keeps them apart.

Headline metric is Youden's J (sensitivity + specificity - 1) on the controls,
because it is the one that answers "does this cutoff distinguish induced from
uninduced cells" without being gamed by unequal control sizes the way raw
accuracy is. AUC is reported alongside for continuous marker scores, since AUC
is threshold-free and so measures the marker's information content rather than
the cutoff's placement.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class ControlSeparation:
    sensitivity: float   # fraction of positive-control cells called senescent
    specificity: float   # fraction of negative-control cells called not
    youden_j: float
    pct_pos_control: float
    pct_neg_control: float
    n_pos: int
    n_neg: int

    @property
    def fpr(self) -> float:
        return 1.0 - self.specificity


def control_separation(
    pos_calls: np.ndarray,
    neg_calls: np.ndarray,
) -> ControlSeparation:
    """How well a boolean call separates positive from negative controls."""
    p = np.asarray(pos_calls, dtype=bool)
    n = np.asarray(neg_calls, dtype=bool)
    if p.size == 0 or n.size == 0:
        raise ValueError(
            f"need non-empty controls, got n_pos={p.size} n_neg={n.size}"
        )
    sens = float(p.mean())
    spec = float(1.0 - n.mean())
    return ControlSeparation(
        sensitivity=sens,
        specificity=spec,
        youden_j=sens + spec - 1.0,
        pct_pos_control=100.0 * sens,
        pct_neg_control=100.0 * float(n.mean()),
        n_pos=int(p.size),
        n_neg=int(n.size),
    )


def pct_senescent(calls: np.ndarray) -> float:
    c = np.asarray(calls, dtype=bool)
    return 100.0 * float(c.mean()) if c.size else float("nan")


def auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Threshold-free separation via the Mann-Whitney U identity.

    Computed from rank sums rather than by sweeping thresholds, so ties are
    handled correctly (average ranks) and the result is exact.
    """
    p = np.asarray(pos_scores, dtype=float)
    n = np.asarray(neg_scores, dtype=float)
    p, n = p[np.isfinite(p)], n[np.isfinite(n)]
    if p.size == 0 or n.size == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1, dtype=float)
    # average ranks within tie groups
    srt = allv[order]
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    r_pos = ranks[: p.size].sum()
    u = r_pos - p.size * (p.size + 1) / 2.0
    return float(u / (p.size * n.size))


def oriented_score(values: np.ndarray, direction: str) -> np.ndarray:
    """Flip a 'down' marker so higher always means more senescent.

    Needed before AUC: an un-flipped down-marker scores ~1-AUC, which looks
    like a uselessly anti-correlated marker when in fact it is informative.
    """
    v = np.asarray(values, dtype=float)
    return -v if direction == "down" else v


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD standardized difference, for effect sizes on continuous data."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)
