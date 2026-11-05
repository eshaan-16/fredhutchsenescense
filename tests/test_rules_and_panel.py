import numpy as np
import pytest

from senescence.panel import Marker, Panel, load_panel
from senescence.rules import PanelRule, enumerate_rules


def test_and_or_mofn_semantics():
    calls = {
        "a": np.array([True, True, False, False]),
        "b": np.array([True, False, True, False]),
    }
    assert PanelRule(("a", "b"), "and").apply(calls).tolist() == [
        True, False, False, False
    ]
    assert PanelRule(("a", "b"), "or").apply(calls).tolist() == [
        True, True, True, False
    ]
    assert PanelRule(("a", "b"), "mofn", 1).apply(calls).tolist() == [
        True, True, True, False
    ]


def test_mofn_with_m_equal_n_is_and():
    rng = np.random.default_rng(0)
    calls = {k: rng.random(200) < 0.5 for k in "abc"}
    a = PanelRule(("a", "b", "c"), "and").apply(calls)
    b = PanelRule(("a", "b", "c"), "mofn", 3).apply(calls)
    assert (a == b).all()


def test_enumerate_is_deduplicated():
    """No subset should emit both AND and its n-of-n twin."""
    got = [str(r) for r in enumerate_rules(["a", "b", "c"])]
    assert len(got) == len(set(got))
    # size-3 subset: AND, OR, and 2-of-3 only
    triples = [g for g in got if g.count(",") == 2 or g.count(" AND ") == 2
               or g.count(" OR ") == 2]
    assert "2-of-3[a,b,c]" in got
    assert "3-of-3[a,b,c]" not in got
    assert "1-of-3[a,b,c]" not in got


def test_enumerate_respects_max_subset():
    rules = enumerate_rules(["a", "b", "c", "d"], max_subset=2)
    assert max(r.n_markers for r in rules) == 2


def test_rule_rejects_bad_m():
    with pytest.raises(ValueError, match="1 <= m"):
        PanelRule(("a", "b"), "mofn", 5)


def test_rule_on_missing_marker_raises_loudly():
    with pytest.raises(KeyError, match="not in calls"):
        PanelRule(("a", "zz"), "and").apply({"a": np.array([True])})


def test_panel_requires_a_negative_control():
    with pytest.raises(ValueError, match="no negative control"):
        Panel.from_dict(
            {
                "markers": [{"name": "p16", "column": "c", "direction": "up"}],
                "controls": {"positive": ["etoposide"]},
            }
        )


def test_panel_rejects_duplicate_markers():
    with pytest.raises(ValueError, match="duplicate marker"):
        Panel.from_dict(
            {
                "markers": [
                    {"name": "p16", "column": "a", "direction": "up"},
                    {"name": "p16", "column": "b", "direction": "up"},
                ],
                "controls": {"negative": ["untreated"]},
            }
        )


def test_marker_rejects_bad_direction():
    with pytest.raises(ValueError, match="must be 'up' or 'down'"):
        Marker("x", "x", "sideways")  # type: ignore[arg-type]


def test_shipped_panel_config_loads_and_is_coherent():
    p = load_panel()
    assert p.negative_controls and p.positive_controls
    assert p.names
    assert all(m.direction in ("up", "down") for m in p.markers)
    assert p["LMNB1"].direction == "down"
    assert p["p16"].direction == "up"
