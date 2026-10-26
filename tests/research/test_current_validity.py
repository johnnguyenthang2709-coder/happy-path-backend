"""Current discovery never widens source invalidation or dependency masks."""

import itertools
import json

import pytest

from happy_path.pretrip.integrated import IntegratedBaseline, database, route_record

FLAGS = [
    dict(include_inferred=a, include_unresolved=b)
    for a, b in itertools.product([False, True], repeat=2)
]


@pytest.fixture
def current(tmp_path):
    db = database(tmp_path)
    for eid, inf, unres in [
        ("place", 0, 0),
        ("road", 0, 0),
        ("infer", 1, 0),
        ("unknown", 0, 1),
    ]:
        db.execute(
            "INSERT INTO entities VALUES (?,?,?,?,?,?,?)",
            (eid, "place", eid, eid, inf, unres, json.dumps({"entity_id": eid})),
        )
    for rid, inf in [("r", 0), ("ir", 1)]:
        row = route_record(
            {
                "route_id": rid,
                "inferred": inf,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[106, 11], [106.001, 11]],
                },
            }
        )
        db.execute("INSERT INTO routes VALUES (?,?,?,?,?,?,?,?,?)", row)
    for eid, inf, unres in [("place", 0, 0), ("infer", 1, 0), ("unknown", 0, 1)]:
        payload = json.dumps({"entity_id": eid, "provider_routed_approach": True})
        db.execute(
            "INSERT INTO relationships VALUES (?,?,?,?,?,?,?)",
            ("rel-" + eid, eid, "r", inf, unres, 0, payload),
        )
        db.execute(
            "INSERT INTO place_links VALUES (?,?,?,?,?,?,?)",
            ("link-" + eid, "r", eid, inf, unres, "rel-" + eid, payload),
        )
    db.execute("INSERT INTO sources VALUES (?,?,?,?)", ("source", "public", 1, "{}"))
    db.execute(
        "INSERT INTO claims VALUES (?,?,?,?,?,?,?)",
        ("claim", "road", "source", "claim", "bounded", 1, "{}"),
    )
    db.execute(
        "INSERT INTO claim_links VALUES (?,?,?,?,?,?)",
        ("claim-link", "r", "claim", "bounded", 1, "{}"),
    )
    db.execute(
        "INSERT INTO road_links VALUES (?,?,?,?,?,?,?,?,?)",
        ("road-link", "r", "road", "component", 0, "bounded", 50, "[]", "{}"),
    )
    db.commit()
    yield db, IntegratedBaseline(tmp_path, enabled=True)
    db.close()


@pytest.mark.parametrize("flags", FLAGS)
def test_invalid_entity_never_current_even_with_discovery(current, flags):
    db, api = current
    db.execute(
        "INSERT INTO invalidations VALUES (?,?)", ("place", "quarantined_or_superseded")
    )
    db.commit()
    assert api.query("entity", entity_id="place", **flags)["status"] == "not_found"
    assert not api.query("search", text="place", **flags)["records"]
    assert not any(
        p["entity_id"] == "place"
        for p in api.query("places", route_id="r", **flags)["places"]
    )


@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("rid", ["r", "ir"])
def test_invalidated_route_never_reintroduced(current, flags, rid):
    db, api = current
    db.execute("INSERT INTO invalidations VALUES (?,?)", (rid, "invalidated_route"))
    db.commit()
    for operation in ["route", "places", "evidence", "coverage"]:
        assert api.query(operation, route_id=rid, **flags)["status"] == "not_found"
    if rid == "r":
        assert not api.query("entity", entity_id="place", **flags)[
            "current_relationships"
        ]


@pytest.mark.parametrize("invalid_id", ["rel-place", "link-place"])
@pytest.mark.parametrize("flags", FLAGS)
def test_access_dependency_invalidation(current, flags, invalid_id):
    db, api = current
    db.execute("INSERT INTO invalidations VALUES (?,?)", (invalid_id, "held"))
    db.commit()
    assert not any(
        p["entity_id"] == "place"
        for p in api.query("places", route_id="r", **flags)["places"]
    )


@pytest.mark.parametrize(
    "invalid_id", ["road", "claim", "source", "claim-link", "road-link"]
)
def test_related_evidence_masks(current, invalid_id):
    db, api = current
    db.execute("INSERT INTO invalidations VALUES (?,?)", (invalid_id, "superseded"))
    db.commit()
    result = api.query(
        "evidence", route_id="r", include_inferred=True, include_unresolved=True
    )
    if invalid_id != "road-link":
        assert not result["claims"]
    if invalid_id in ["road", "road-link"]:
        assert not result["road_context"]


def test_valid_inferred_and_unresolved_discovery_are_independent(current):
    _, api = current
    assert api.query("entity", entity_id="infer")["status"] == "not_found"
    assert (
        api.query("entity", entity_id="infer", include_inferred=True)["status"] == "ok"
    )
    assert (
        api.query("entity", entity_id="unknown", include_inferred=True)["status"]
        == "not_found"
    )
    assert (
        api.query("entity", entity_id="unknown", include_unresolved=True)["status"]
        == "ok"
    )
    assert api.query("route", route_id="ir", include_inferred=True)["status"] == "ok"
    assert {
        p["entity_id"]
        for p in api.query("places", route_id="r", include_inferred=True)["places"]
    } == {"place", "infer"}
    assert {
        p["entity_id"]
        for p in api.query("places", route_id="r", include_unresolved=True)["places"]
    } == {"place", "unknown"}


def test_cached_comparison_cannot_reintroduce_invalidated_route(current):
    db, api = current
    db.execute(
        "INSERT INTO comparisons VALUES (?,?)",
        ("case", json.dumps({"routes": [{"route_id": "r"}], "comparisons": []})),
    )
    db.execute("INSERT INTO invalidations VALUES (?,?)", ("r", "superseded"))
    db.commit()
    result = api.query(
        "compare", case_id="case", include_inferred=True, include_unresolved=True
    )
    assert result["status"] == "stale_comparison" and result["comparison"] is None


def test_claim_binding_does_not_outlive_its_place_relationship(current):
    db, api = current
    db.execute("UPDATE claims SET entity_id='place'")
    db.execute(
        "UPDATE claim_links SET payload=?",
        (json.dumps({"detail": {"place_link_id": "link-place"}}),),
    )
    db.execute(
        "INSERT INTO invalidations VALUES (?,?)", ("rel-place", "quarantined_approach")
    )
    db.commit()
    assert not api.query(
        "evidence", route_id="r", include_inferred=True, include_unresolved=True
    )["claims"]


def test_cached_iconic_evidence_cannot_outlive_its_source(current):
    db, api = current
    cached = {
        "routes": [{"route_id": "r", "iconic": {"nearby_only_entities": ["road"]}}],
        "comparisons": [],
    }
    db.execute(
        "UPDATE claims SET payload=?",
        (json.dumps({"claim_kind": "national_historic_site_designation"}),),
    )
    db.execute("INSERT INTO comparisons VALUES (?,?)", ("case", json.dumps(cached)))
    db.commit()
    assert api.query("compare", case_id="case")["status"] == "ok"
    db.execute("INSERT INTO invalidations VALUES (?,?)", ("source", "superseded"))
    db.commit()
    assert (
        api.query("compare", case_id="case", include_inferred=True)["status"]
        == "stale_comparison"
    )
