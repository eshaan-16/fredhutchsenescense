"""Panel configuration loading and validation.

The panel config is the single source of truth for which markers exist, which
direction each one moves with senescence, and how the controls are labeled.
Loading it through here (rather than reading YAML ad hoc) means a typo in a
marker name fails immediately instead of producing a silently empty column.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import yaml

Direction = Literal["up", "down"]
Level = Literal["per_cell", "per_well"]


@dataclasses.dataclass(frozen=True)
class Marker:
    name: str
    column: str
    direction: Direction
    canonical: bool = True

    def __post_init__(self) -> None:
        if self.direction not in ("up", "down"):
            raise ValueError(
                f"marker {self.name!r}: direction must be 'up' or 'down', "
                f"got {self.direction!r}"
            )


@dataclasses.dataclass(frozen=True)
class Grouping:
    condition_column: str = "condition"
    replicate_column: str = "replicate"
    batch_column: str = "plate"
    cell_id_column: str = "cell_id"


@dataclasses.dataclass(frozen=True)
class Panel:
    markers: tuple[Marker, ...]
    negative_controls: tuple[str, ...]
    positive_controls: tuple[str, ...]
    test_conditions: tuple[str, ...]
    grouping: Grouping
    level: Level = "per_cell"
    modality: str = "immunofluorescence"
    log_transform: bool = True
    log_pseudocount: float = 1.0
    covariates: tuple[str, ...] = ()

    # ---- lookups -------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.markers)

    @property
    def canonical(self) -> tuple[Marker, ...]:
        return tuple(m for m in self.markers if m.canonical)

    def __getitem__(self, name: str) -> Marker:
        for m in self.markers:
            if m.name == name:
                return m
        raise KeyError(f"no marker named {name!r}; have {self.names}")

    @property
    def is_single_cell(self) -> bool:
        return self.level == "per_cell"

    # ---- construction --------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Panel":
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Panel":
        meas = raw.get("measurement", {})
        markers = tuple(
            Marker(
                name=m["name"],
                column=m["column"],
                direction=m["direction"],
                canonical=bool(m.get("canonical", True)),
            )
            for m in raw["markers"]
        )
        if not markers:
            raise ValueError("panel defines no markers")
        dupes = {n for n in (m.name for m in markers) if
                 sum(1 for x in markers if x.name == n) > 1}
        if dupes:
            raise ValueError(f"duplicate marker names in panel: {sorted(dupes)}")

        controls = raw.get("controls", {})
        neg = tuple(controls.get("negative") or ())
        pos = tuple(controls.get("positive") or ())
        if not neg:
            raise ValueError(
                "panel defines no negative control. Every threshold rule here "
                "is calibrated against the negative distribution, so this is "
                "required, not optional."
            )

        g = raw.get("grouping", {})
        grouping = Grouping(
            condition_column=g.get("condition_column", "condition"),
            replicate_column=g.get("replicate_column", "replicate"),
            batch_column=g.get("batch_column", "plate"),
            cell_id_column=g.get("cell_id_column", "cell_id"),
        )

        return cls(
            markers=markers,
            negative_controls=neg,
            positive_controls=pos,
            test_conditions=tuple(controls.get("test") or ()),
            grouping=grouping,
            level=meas.get("level", "per_cell"),
            modality=meas.get("modality", "immunofluorescence"),
            log_transform=bool(meas.get("log_transform", True)),
            log_pseudocount=float(meas.get("log_pseudocount", 1.0)),
            covariates=tuple(raw.get("covariates") or ()),
        )


DEFAULT_PANEL_PATH = Path(__file__).resolve().parents[2] / "config" / "panel.yaml"


def load_panel(path: str | Path | None = None) -> Panel:
    return Panel.from_yaml(path or DEFAULT_PANEL_PATH)
