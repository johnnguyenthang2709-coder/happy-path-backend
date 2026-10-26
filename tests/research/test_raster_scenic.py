"""Analytic fixtures test extraction mechanics; delivered observations use real rasters."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from happy_path.pretrip.raster_scenic import RasterStore, StorageBudgetError, checksum
from happy_path.pretrip.raster_scenic_jobs import (
    add_section,
    canonical,
    compute_cells,
    database,
    score_geometry,
)
from happy_path.pretrip.scoring import scenic_score

POLICY = {
    "scenic_weights": {
        "tree_cover": 0.35,
        "water": 0.25,
        "cropland": 0.1,
        "ruggedness": 0.3,
    },
    "ruggedness_saturation_m": 10,
    "minimum_landscape_coverage": 0.8,
}


def asset(folder, name, values, affine, source="dem", crs=4326):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + ".tif")
    a = np.asarray(values, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=a.shape[1],
        height=a.shape[0],
        count=1,
        dtype=a.dtype,
        crs=crs,
        transform=affine,
        nodata=-9999,
    ) as ds:
        ds.write(a, 1)
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "source": source,
                "success": True,
                "year": 2021,
                "cache_key": name,
                "crop_method": "native_pixel_center_partition_v2",
                "sha256": checksum(path),
            }
        )
    )
    return path


def test_crop_edge_recovers_native_eight_neighbors_and_nodata(tmp_path):
    # 3x3 native ramp cut across two crops; center is on the crop's edge.
    inherited = tmp_path / "inherited"
    asset(
        inherited, "left", [[1, 2], [4, 5], [7, 8]], from_origin(106, 11, 0.001, 0.001)
    )
    asset(inherited, "right", [[3], [6], [9]], from_origin(106.002, 11, 0.001, 0.001))
    store = RasterStore(tmp_path / "cache", [inherited])
    z, rms, support, _, refs = store.terrain([[106.0015, 10.9985]])
    assert z[0] == 5
    assert rms[0] == pytest.approx(np.sqrt(60 / 8))
    assert support.tolist() == [9]
    assert len({r[0] for r in refs[0]}) == 2
    assert np.isnan(store.terrain([[106.0005, 10.9995]])[1][0])
    store.close()


def test_invalid_codes_overlap_conflict_and_coordinate_order(tmp_path):
    inherited = tmp_path / "inherited"
    asset(
        inherited, "a", [[10, 80, 99]], from_origin(106, 11, 0.001, 0.001), "worldcover"
    )
    asset(inherited, "b", [[40]], from_origin(106, 11, 0.001, 0.001), "worldcover")
    store = RasterStore(tmp_path / "cache", [inherited])
    values, _, _, refs = store.sample(
        "worldcover",
        [
            [106.0005, 10.9995],
            [106.0015, 10.9995],
            [106.0025, 10.9995],
            [10.9995, 106.0015],
        ],
    )
    assert np.isnan(values[[0, 2, 3]]).all()
    assert values[1] == 80 and refs[0] is None
    store.close()


def test_wrong_crs_rejected(tmp_path):
    inherited = tmp_path / "inherited"
    asset(inherited, "bad", [[5]], from_origin(0, 100, 30, 30), crs=32648)
    with pytest.raises(ValueError, match="CRS"):
        RasterStore(tmp_path / "cache", [inherited])


def test_product_edge_can_revisit_neighbor_tile_for_another_block(
    tmp_path, monkeypatch
):
    import happy_path.pretrip.raster_scenic as module

    source = tmp_path / "sources"
    left = asset(
        source,
        "left",
        np.full((2, 2048), 5),
        from_origin(103.9525, 11.0005, 0.001, 0.001),
    )
    right = asset(
        source,
        "right",
        np.full((2, 2), 9),
        from_origin(106.0005, 11.0005, 0.001, 0.001),
    )
    store = RasterStore(tmp_path / "cache", online=True, reserve_bytes=0)
    handles = {"left": rasterio.open(left), "right": rasterio.open(right)}
    monkeypatch.setattr(
        module, "raster_url", lambda kind, lat, lon: "left" if lon < 106 else "right"
    )
    monkeypatch.setattr(store, "_handle", lambda url: handles[url])
    # Left product is visited first for its western block; the half-pixel edge
    # then requires revisiting it for its eastern block at the unchanged point.
    values, _, _, refs = store.sample("dem", [[103.953, 11], [106, 11]])
    assert values.tolist() == [5, 5] and all(refs)
    assert len(list((tmp_path / "cache").glob("*.tif"))) == 2
    for ds in handles.values():
        ds.close()
    store.close()


def test_eviction_never_touches_inherited_and_budget_is_enforced(tmp_path):
    inherited = tmp_path / "inherited"
    cache = tmp_path / "cache"
    RasterStore(cache).close()
    original = asset(inherited, "kept", [[5]], from_origin(106, 11, 0.001, 0.001))
    owned = asset(cache, "owned", [[6]], from_origin(106.01, 11, 0.001, 0.001))
    oldhash = checksum(original)
    store = RasterStore(
        cache, [inherited], budget_bytes=owned.stat().st_size + 100, reserve_bytes=0
    )
    store.budget_check(200)
    assert not owned.exists() and original.exists() and checksum(original) == oldhash
    with pytest.raises(StorageBudgetError):
        store.budget_check(store.budget + 1)
    store.close()


def test_populated_foreign_cache_cannot_be_adopted_or_evicted(tmp_path):
    foreign = asset(
        tmp_path / "foreign", "original", [[5]], from_origin(106, 11, 0.001, 0.001)
    )
    before = checksum(foreign)
    with pytest.raises(StorageBudgetError, match="ownership marker"):
        RasterStore(tmp_path / "foreign", budget_bytes=1)
    assert checksum(foreign) == before


def test_reverse_geometry_reuse_parallel_separation_and_idempotency(tmp_path):
    line = LineString([(106, 11), (106.001, 11)])
    reverse = LineString(list(line.coords)[::-1])
    assert canonical(line)[0] == canonical(reverse)[0]
    assert canonical(line)[2] != canonical(reverse)[2]
    assert (
        canonical(LineString([(106, 11.00001), (106.001, 11.00001)]))[0]
        != canonical(line)[0]
    )
    db = database(tmp_path)
    for _ in range(2):
        add_section(db, "a", line, "osm", {"source_version": 1})
    add_section(db, "b", reverse, "osm", {"source_version": 1})
    assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM sections").fetchone()[0] == 2
    db.close()


def test_direction_view_preserves_oneway_reverse_chainage_and_side(tmp_path):
    from happy_path.pretrip.raster_scenic_reports import directional_views

    db = database(tmp_path)
    sid, gid = add_section(
        db,
        "oneway-reverse",
        LineString([(106, 11), (106.002, 11)]),
        "osm",
        {"source_version": 1, "tags_json": json.dumps({"oneway": "-1"})},
    )
    length = db.execute(
        "SELECT length_m FROM jobs WHERE geometry_id=?", (gid,)
    ).fetchone()[0]
    db.execute(
        "INSERT INTO cells(cell_id,geometry_id,cell_index,start_m,end_m,length_m,payload) VALUES (?,?,?,?,?,?,?)",
        ("first", gid, 0, 0, 50, 50, "{}"),
    )
    db.execute(
        "INSERT INTO cells(cell_id,geometry_id,cell_index,start_m,end_m,length_m,payload) VALUES (?,?,?,?,?,?,?)",
        ("last", gid, 1, 50, length, length - 50, "{}"),
    )
    directional_views(db)
    rows = db.execute(
        "SELECT direction,cell_id,travel_start_m,travel_end_m,canonical_left_to_travel_left_multiplier FROM directional_cells WHERE section_id=? ORDER BY travel_start_m",
        (sid,),
    ).fetchall()
    assert [(r[0], r[1], r[4]) for r in rows] == [
        ("reverse_source_order", "last", -1),
        ("reverse_source_order", "first", -1),
    ]
    assert rows[0][2] == pytest.approx(0)
    assert rows[0][3] == pytest.approx(rows[1][2])
    assert rows[-1][3] == pytest.approx(length)
    db.close()


class AnalyticStore:
    def sample(self, kind, coords):
        n = len(coords)
        return np.resize([10, 80, 40, 10, 80, 40], n), None, None, [None] * n

    def terrain(self, coords):
        n = len(coords)
        return (
            np.full(n, 12.0),
            np.full(n, 5.0),
            np.full(n, 9),
            [None] * n,
            [[None] * 9 for _ in coords],
        )


def test_cell_length_offsets_tunnel_raw_retention_and_restart():
    line = LineString([(650000, 1200000), (650123, 1200000)])
    rows = compute_cells("x", line, 32648, AnalyticStore())
    assert [r["cell_length_m"] for r in rows] == [50, 50, 23]
    assert compute_cells("x", line, 32648, AnalyticStore(), start_index=2) == rows[2:]
    assert [r["offset_left_m"] for r in rows[0]["roadside_samples"]] == [
        -250,
        -150,
        -50,
        50,
        150,
        250,
    ]
    assert rows[0]["roadside_tree_fraction"] == pytest.approx(1 / 3)
    masked = compute_cells("x", line, 32648, AnalyticStore(), tunnel=True)
    assert masked[0]["roadside_tree_fraction"] is None
    assert masked[0]["raw_sampled_fractions"]["tree"] == pytest.approx(1 / 3)
    assert masked[0]["raw_dsm_ruggedness_rms_m"] == 5
    score = scenic_score(masked, POLICY)
    assert score["score"] is None and score["missing_evidence_bounds"] == [0, 100]


def test_length_weighting_unknown_bounds_and_historical_policy_compatibility(tmp_path):
    rows = compute_cells(
        "x", LineString([(650000, 1200000), (650100, 1200000)]), 32648, AnalyticStore()
    )
    rows[1].update(
        roadside_tree_fraction=None,
        roadside_water_fraction=None,
        roadside_cropland_fraction=None,
        dsm_ruggedness_rms_m=None,
        roadside_sample_coverage=0,
    )
    score = scenic_score(rows, POLICY)
    assert score["evidence_weight_coverage"] == pytest.approx(0.5)
    assert score["score"] is None and score["observed_score"] is not None
    assert score["missing_evidence_bounds"][1] - score["missing_evidence_bounds"][
        0
    ] == pytest.approx(50)
    db = database(tmp_path)
    for r in rows:
        db.execute(
            "INSERT INTO cells(cell_id,geometry_id,cell_index,payload) VALUES (?,?,?,?)",
            (r["cell_id"], "x", r["cell_index"], json.dumps(r)),
        )
    result = score_geometry(db, "x", POLICY)
    assert all(result[k] == score[k] for k in score)
    assert sum(
        c["conditional_points"] or 0 for c in result["component_contributions"].values()
    ) == pytest.approx(score["observed_score"])
    db.close()


def test_protected_originals_unchanged():
    root = Path(__file__).resolve().parents[2]
    path = (
        root / "exports/route_intelligence/raster_scenic_nationwide/input_manifest.json"
    )
    if not path.exists():
        pytest.skip("Acquisition manifest not distributed with source checkout")
    manifest = json.loads(path.read_text("utf8"))
    for field in ["inputs_sha256", "protected_runtime_data_sha256"]:
        for name, expected in manifest.get(field, {}).items():
            assert checksum(root / name.replace("\\", "/")) == expected, name
