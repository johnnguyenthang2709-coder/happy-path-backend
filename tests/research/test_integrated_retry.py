"""Retry mechanics use real native raster fixtures; no release data is written."""

import json
import sqlite3

import numpy as np
import pytest
from rasterio.transform import from_origin

from happy_path.pretrip import integrated as module
from happy_path.pretrip.raster_scenic import RasterStore
from happy_path.pretrip.raster_scenic_jobs import database as raster_database
from test_raster_scenic import POLICY, asset


class NoTunnel:
    def match(self, *args):
        return None, "unmatched"


@pytest.fixture
def job(tmp_path, monkeypatch):
    raster = tmp_path / "raster"
    raster.mkdir()
    config = {
        "scoring": {
            **POLICY,
            "iconic_entity_saturation": 3,
            "express_eta_weight": 0.7,
            "express_distance_weight": 0.3,
        }
    }
    (raster / "config.json").write_text(json.dumps(config))
    raster_database(raster).close()
    db = module.database(tmp_path / "out")
    row = module.route_record(
        {
            "route_id": "r",
            "case_id": "case",
            "provider_distance_m": 100,
            "provider_duration_s": 60,
            "geometry": {
                "type": "LineString",
                "coordinates": [[106, 11], [106.001, 11]],
            },
        },
        role="priority",
    )
    db.execute("INSERT INTO routes VALUES (?,?,?,?,?,?,?,?,?)", row)
    db.commit()
    osm = tmp_path / "osm.parquet"
    osm.write_bytes(b"test matcher input")
    monkeypatch.setattr(module, "route_matcher", lambda *a, **kw: NoTunnel())
    yield db, raster, tmp_path, osm
    db.close()


def add_dem(raster):
    asset(
        raster / "raster_cache",
        "dem",
        np.full((40, 40), 100),
        from_origin(105.99, 11.01, 0.001, 0.001),
    )


def add_wc(raster, *, partial=False, nodata=False):
    asset(
        raster / "raster_cache",
        "wc",
        np.full((40 if not partial else 12, 40), -9999 if nodata else 10),
        from_origin(105.99, 11.01, 0.001, 0.001),
        source="worldcover",
    )


def run(job, *, retry=False, online=False):
    db, raster, root, osm = job
    return module.score_routes(
        db, raster, root, online, osm_path=osm, retry_missing=retry
    )


def score(db):
    return json.loads(db.execute("SELECT payload FROM scores").fetchone()[0])


def test_offline_missing_then_online_retry_acquires_and_refreshes_comparison(
    job, monkeypatch
):
    db, raster, _, _ = job
    run(job)
    assert score(db)["score"] is None
    module.compare_routes(
        db, json.loads((raster / "config.json").read_text())["scoring"]
    )
    assert db.execute("SELECT count(*) FROM comparisons").fetchone()[0] == 1
    acquisitions = []

    def acquire(self, kind, coords):
        assert self.online
        acquisitions.append(kind)
        add_dem(raster) if kind == "dem" else add_wc(raster)
        # Reopen indexed local fixtures as an official-window acquisition stand-in.
        # The extractor/sampler itself is real; no external requests in unit tests.
        folder = raster / "raster_cache"
        for p in folder.glob("*.tif"):
            receipt = json.loads(p.with_suffix(".meta.json").read_text())
            if not any(a["key"] == receipt["cache_key"] for a in self.assets):
                self._add(p, receipt, False)
        self._index()

    monkeypatch.setattr(RasterStore, "_acquire", acquire)
    assert run(job, retry=True, online=True)["retry_results_this_run"] == {
        "improved": 1
    }
    assert {"worldcover", "dem"} <= set(acquisitions)
    assert score(db)["evidence_weight_coverage"] == pytest.approx(1)
    registered = {row[0] for row in db.execute("SELECT asset_id FROM raster_assets")}
    for (payload,) in db.execute("SELECT payload FROM cells"):
        cell = json.loads(payload)
        refs = [s["native_pixel_ref"] for s in cell["roadside_samples"]]
        refs += [cell["dsm_center_ref"], *cell["dsm_neighbor_refs"]]
        assert all(ref and ref[0] in registered for ref in refs)
    assert db.execute("SELECT count(*) FROM comparisons").fetchone()[0] == 0
    module.compare_routes(
        db, json.loads((raster / "config.json").read_text())["scoring"]
    )
    result = json.loads(db.execute("SELECT payload FROM comparisons").fetchone()[0])
    assert result["routes"][0]["scenic"]["score"] == pytest.approx(score(db)["score"])


def test_partial_inputs_are_retryable_even_when_score_exists(job):
    db, raster, _, _ = job
    add_dem(raster)
    add_wc(raster, partial=True)
    run(job)
    first = score(db)
    assert first["score"] is not None and first["evidence_weight_coverage"] < 1
    # Add missing half on the same native grid without replacing earlier assets.
    asset(
        raster / "raster_cache",
        "wc-south",
        np.full((30, 40), 10),
        from_origin(105.99, 10.998, 0.001, 0.001),
        source="worldcover",
    )
    assert run(job, retry=True)["retry_results_this_run"] == {"improved": 1}
    assert score(db)["evidence_weight_coverage"] == pytest.approx(1)


def test_completed_scores_skipped_without_store_or_duplicate_work(job, monkeypatch):
    db, raster, _, _ = job
    add_dem(raster)
    add_wc(raster)
    run(job)
    before = db.execute("SELECT * FROM scores").fetchall()
    monkeypatch.setattr(
        module,
        "RasterStore",
        lambda *a, **k: pytest.fail("Complete result must not open raster store"),
    )
    assert run(job, retry=True)["retry_results_this_run"] == {
        "complete_or_intentionally_excluded": 1
    }
    assert db.execute("SELECT * FROM scores").fetchall() == before
    assert not db.execute("SELECT * FROM score_retry_history").fetchall()


def test_failed_retry_preserves_checkpoint_and_evidence(job, monkeypatch):
    db, raster, _, _ = job
    add_dem(raster)
    run(job)
    before = (
        db.execute("SELECT * FROM scores").fetchall(),
        db.execute("SELECT * FROM cells").fetchall(),
    )

    def fail(*a, **kw):
        raise OSError("bounded acquisition failure")

    monkeypatch.setattr(module, "compute_cells", fail)
    assert run(job, retry=True)["retry_results_this_run"] == {
        "failed_checkpoint_preserved": 1
    }
    assert before == (
        db.execute("SELECT * FROM scores").fetchall(),
        db.execute("SELECT * FROM cells").fetchall(),
    )
    assert db.execute("SELECT successful FROM score_retry_history").fetchone()[0] == 0
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_covered_source_nodata_not_endlessly_retried(job, monkeypatch):
    db, raster, _, _ = job
    add_dem(raster)
    add_wc(raster, nodata=True)
    run(job)
    monkeypatch.setattr(
        module,
        "compute_cells",
        lambda *a, **k: pytest.fail("NoData is not an absent download"),
    )
    for _ in range(2):
        assert run(job, retry=True)["retry_results_this_run"] == {
            "terminal_or_complete": 1
        }


def test_tunnel_and_degenerate_tangent_are_terminal(job, monkeypatch):
    db, _, _, _ = job

    class Tunnel:
        def match(self, *a):
            return {
                "source": {"entity_id": "tunnel", "tags": {"tunnel": "yes"}},
                "distance_m": 0,
            }, "supported_geometry_not_field_verified"

    monkeypatch.setattr(module, "route_matcher", lambda *a, **k: Tunnel())
    run(job)
    monkeypatch.setattr(
        module,
        "RasterStore",
        lambda *a, **k: pytest.fail("Intentional exclusion must not acquire"),
    )
    for _ in range(2):
        assert run(job, retry=True)["retry_results_this_run"] == {
            "complete_or_intentionally_excluded": 1
        }
    from shapely.geometry import LineString

    cell = {
        "start_m": 0,
        "end_m": 2,
        "roadside_sample_coverage": 0,
        "dsm_ruggedness_rms_m": None,
    }
    assert not module.cell_missing_inputs(cell, LineString([(0, 0), (1, 0), (0, 0)]))
