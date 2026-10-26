"""Analytic mechanics plus opt-in checks on the actual integrated release."""

import json
import os
import sqlite3
from pathlib import Path

import pytest
from shapely.geometry import LineString

from happy_path.pretrip.integrated import (
    AUDIT_ID,
    IntegratedBaseline,
    MeasuredLine,
    attach_places,
    database,
    directed_intervals,
    interval_length,
    readonly,
    source_public,
    split_landscape,
    union_intervals,
    validate,
)


def test_disabled_has_no_filesystem_or_runtime_dependency():
    assert IntegratedBaseline("not-a-path").query() == {
        "status": "disabled",
        "records": [],
    }


def test_overlap_union_has_no_double_counting():
    assert union_intervals([[0, 60], [50, 90], [90, 100], [200, 210]]) == [
        [0, 100],
        [200, 210],
    ]
    assert interval_length([[0, 60], [50, 90]]) == 90


def test_indexed_chainage_slicing_matches_original_vertices():
    from shapely.ops import substring

    line = LineString([(0, 0), (20, 0), (20, 30), (60, 30)])
    measured = MeasuredLine(line)
    for a, b in [(0, 50), (10, 70), (20, 50), (50, 90)]:
        assert measured.slice(a, b).equals_exact(substring(line, a, b), 1e-9)


def test_direction_and_parallel_separation():
    a = LineString([(0, 0), (100, 0)])
    b = LineString([(20, 0), (80, 0)])
    assert interval_length(directed_intervals(a, b, 0.01)) == pytest.approx(60.02)
    assert not directed_intervals(a, LineString(list(b.coords)[::-1]), 0.01)
    assert not directed_intervals(a, LineString([(20, 30), (80, 30)]), 15)


def test_partial_shared_cells_preserve_length_and_observation_support(tmp_path):
    db = database(tmp_path)
    line = LineString([(0, 0), (100, 0)])
    db.execute(
        "INSERT INTO routes VALUES (?,?,?,?,?,?,?,?,?)",
        (
            "r",
            "priority",
            "case",
            "forward",
            "g",
            line.wkb,
            100,
            0,
            json.dumps({"canonical_reversed": False}),
        ),
    )
    for i in range(2):
        c = {
            "start_m": i * 50,
            "end_m": (i + 1) * 50,
            "cell_length_m": 50,
            "roadside_tree_fraction": 0.5,
            "roadside_water_fraction": 0,
            "roadside_cropland_fraction": 0.5,
            "roadside_sample_coverage": 1,
            "dsm_ruggedness_rms_m": 5,
        }
        db.execute(
            "INSERT INTO cells VALUES (?,?,?,?,?,?)",
            ("g", i, c["start_m"], c["end_m"], 50, json.dumps(c)),
        )
    policy = {
        "scenic_weights": {
            "tree_cover": 0.35,
            "water": 0.25,
            "cropland": 0.1,
            "ruggedness": 0.3,
        },
        "ruggedness_saturation_m": 10,
        "minimum_landscape_coverage": 0.8,
    }
    result = split_landscape(db, "r", line, LineString([(25, 0), (75, 0)]), policy)
    assert result["shared"]["length_m"] == pytest.approx(50)
    assert result["differentiating"]["length_m"] == pytest.approx(50)
    assert result["shared"]["scenic"]["score"] == pytest.approx(
        result["differentiating"]["scenic"]["score"]
    )
    db.close()


def test_local_files_are_not_public_independent_evidence():
    assert not source_public({"canonical_family": "/private/local.json"})
    assert source_public({"original_urls": ["https://vietnam.travel/example"]})


def test_schema_defaults_exclude_holds_inference_and_audit(tmp_path):
    db = database(tmp_path)
    for eid, inf, unres in [
        ("supported", 0, 0),
        ("hypothesis", 1, 0),
        ("unresolved", 0, 1),
    ]:
        db.execute(
            "INSERT INTO entities VALUES (?,?,?,?,?,?,?)",
            (eid, "place", eid, eid, inf, unres, "{}"),
        )
    db.execute(
        "INSERT INTO routes VALUES (?,?,?,?,?,?,?,?,?)",
        (AUDIT_ID, "original_audit_reference", "case", "a", None, None, 1, 0, "{}"),
    )
    db.execute(
        "INSERT INTO relationships VALUES (?,?,?,?,?,?,?)",
        ("held", "supported", "r", 0, 0, 0, "{}"),
    )
    db.execute(
        "INSERT INTO invalidations VALUES (?,?)",
        ("held", "invalidated_provider_identity"),
    )
    db.commit()
    assert [r[0] for r in db.execute("SELECT entity_id FROM effective_entities")] == [
        "supported"
    ]
    assert not db.execute("SELECT * FROM effective_routes").fetchall()
    assert not db.execute("SELECT * FROM effective_relationships").fetchall()
    db.close()


def test_restart_registration_is_idempotent(tmp_path):
    for _ in range(2):
        db = database(tmp_path)
        db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", ("version", "v1"))
        db.commit()
        db.close()
    db = readonly(tmp_path / "integrated.sqlite")
    assert db.execute("SELECT count(*) FROM metadata").fetchone()[0] == 1
    with pytest.raises(sqlite3.OperationalError):
        db.execute("DELETE FROM metadata")
    db.close()


def test_access_join_keeps_multiple_source_variants_and_is_idempotent(tmp_path):
    from happy_path.pretrip.integrated import route_record

    db = database(tmp_path)
    r = {
        "route_id": "captured",
        "geometry": {"type": "LineString", "coordinates": [[106, 11], [106.002, 11]]},
    }
    db.execute("INSERT INTO routes VALUES (?,?,?,?,?,?,?,?,?)", route_record(r))
    p = {
        "entity_id": "place",
        "name": "Example",
        "position": [106.001, 11],
        "coordinate_role": "provider_pin_not_entrance",
    }
    db.execute(
        "INSERT INTO entities VALUES (?,?,?,?,?,?,?)",
        ("place", "place", "Example", "example", 0, 0, json.dumps(p)),
    )
    for rid, detour in [("approach", False), ("compatible", True)]:
        payload = {
            "permitted_uses": {
                "provider_routed_approach": True,
                "compatible_detour": detour,
            }
        }
        db.execute(
            "INSERT INTO relationships VALUES (?,?,?,?,?,?,?)",
            (rid, "place", "captured", 0, 0, 0, json.dumps(payload)),
        )
    for _ in range(2):
        attach_places(db)
        assert (
            db.execute("SELECT count(*) FROM effective_place_links").fetchone()[0] == 2
        )
        assert {
            r[0] for r in db.execute("SELECT relationship_id FROM place_links")
        } == {"approach", "compatible"}
    db.close()


@pytest.fixture(scope="module")
def actual():
    path = os.getenv("HAPPY_PATH_INTEGRATED_TEST_DATA")
    if not path:
        pytest.skip(
            "Actual private/provider snapshot not bundled; set HAPPY_PATH_INTEGRATED_TEST_DATA explicitly"
        )
    directory = Path(path)
    assert (directory / "release_report.json").exists(), (
        "Explicit actual-data test input must be complete"
    )
    return directory


def test_actual_populated_inventory_and_route(actual):
    api = IntegratedBaseline(actual, enabled=True)
    assert api.query()["counts"]["routes"] >= 32
    result = api.query("route", route_id="route:5dc7cfdc7b7e718e5df5")
    assert result["status"] == "ok" and result["scenic"]["ranking_eligible"]
    assert result["places"] and result["claims"]
    assert (
        result["raster_provenance"]["method"]
        == "identical_registered_geometry_same_crs_and_version"
    )


def test_actual_scope_and_quarantine(actual):
    db = readonly(actual / "integrated.sqlite")
    assert not validate(db)
    assert not db.execute(
        "SELECT * FROM effective_place_links WHERE relationship_id='phase410:c96dfa804c4a803105f4'"
    ).fetchall()
    assert not db.execute(
        "SELECT * FROM effective_routes WHERE route_id=?", (AUDIT_ID,)
    ).fetchall()
    for (payload,) in db.execute(
        "SELECT l.payload FROM effective_claim_links l JOIN claims c USING(claim_id) WHERE c.entity_id IN ('HPOI_7E839CD3CCE0F80A','HPOI_2D4B29FF7EA0D79A')"
    ):
        r = json.loads(payload)
        assert r["application"] == "nearby_place_context"
        assert "intervals_m" not in r.get("detail", {})
    db.close()


def test_actual_score_consistency_bounds_and_tunnels(actual):
    db = readonly(actual / "integrated.sqlite")
    for gid, payload in db.execute("SELECT geometry_id,payload FROM scores"):
        r = json.loads(payload)
        lo, hi = r["missing_evidence_bounds"]
        assert 0 <= lo <= hi <= 100
        assert r["ranking_eligible"] == (r["evidence_weight_coverage"] >= 0.8)
        if r["observed_score"] is not None:
            assert sum(
                c["conditional_points"] or 0
                for c in r["component_contributions"].values()
            ) == pytest.approx(r["observed_score"])
    assert (
        db.execute(
            "SELECT count(*) FROM cells WHERE json_extract(payload,'$.supported_tunnel_excluded')=1"
        ).fetchone()[0]
        > 0
    )
    assert (
        db.execute(
            "SELECT count(*) FROM cells WHERE json_extract(payload,'$.supported_tunnel_excluded')=1 AND json_extract(payload,'$.dsm_ruggedness_rms_m') IS NOT NULL"
        ).fetchone()[0]
        == 0
    )
    db.close()


def test_actual_comparison_shared_lengths_and_no_forced_winner(actual):
    result = IntegratedBaseline(actual, enabled=True).query(
        "compare", case_id="hcm_dalat"
    )
    assert result["status"] == "ok" and len(result["comparison"]["routes"]) == 3
    for comparison in result["comparison"]["comparisons"]:
        assert comparison["winner"] is None
        for split in comparison["landscape_shared_and_differentiating"]:
            assert (
                split["shared"]["length_m"] >= 0
                and split["differentiating"]["length_m"] >= 0
            )
