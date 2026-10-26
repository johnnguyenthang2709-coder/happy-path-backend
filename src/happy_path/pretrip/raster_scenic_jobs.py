"""Durable opt-in jobs for native raster Scenic features; never registered in runtime."""

import json
import math
import shutil
import sqlite3
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer
from shapely import from_wkb, get_coordinates, line_interpolate_point, reverse
from shapely.geometry import shape
from shapely.ops import transform

from happy_path.core.road_sources import atomic, json_bytes, now
from happy_path.pretrip.calibration import validated_scenic
from happy_path.pretrip.raster_scenic import (
    VERSION,
    RasterStore,
    StorageBudgetError,
    digest,
)


def read(path):
    return json.loads(Path(path).read_text("utf8"))


def save(out, name, obj):
    atomic(out / name, json_bytes(obj))


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@lru_cache(maxsize=8)
def projection(crs, inverse=False):
    return Transformer.from_crs(
        crs if inverse else 4326, 4326 if inverse else crs, always_xy=True
    )


def database(out):
    db = sqlite3.connect(out / "scenic.sqlite", timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-32768")
    db.execute("PRAGMA wal_autocheckpoint=2000")
    db.execute("PRAGMA journal_size_limit=16777216")
    db.executescript("""CREATE TABLE IF NOT EXISTS jobs(geometry_id TEXT PRIMARY KEY,geometry BLOB,crs INTEGER,length_m REAL,partition_id TEXT,priority INTEGER,spatial_key TEXT,status TEXT DEFAULT 'pending',last_cell INTEGER DEFAULT -1,score_json TEXT,error TEXT);
CREATE TABLE IF NOT EXISTS sections(section_id TEXT PRIMARY KEY,source_id TEXT,geometry_id TEXT,source_version TEXT,kind TEXT,category TEXT,partition_id TEXT,direction TEXT,source_reversed INTEGER,payload TEXT,disposition TEXT);
CREATE INDEX IF NOT EXISTS job_queue ON jobs(priority,partition_id,spatial_key,status);
CREATE INDEX IF NOT EXISTS section_geometry ON sections(geometry_id);
CREATE INDEX IF NOT EXISTS section_source ON sections(source_id);
CREATE TABLE IF NOT EXISTS cells(cell_id TEXT PRIMARY KEY,geometry_id TEXT,cell_index INTEGER,start_m REAL,end_m REAL,length_m REAL,lon REAL,lat REAL,tree REAL,water REAL,cropland REAL,ruggedness REAL,sample_coverage REAL,tunnel INTEGER,payload TEXT,UNIQUE(geometry_id,cell_index));
CREATE INDEX IF NOT EXISTS cells_geometry ON cells(geometry_id,cell_index);
CREATE TABLE IF NOT EXISTS raster_assets(cache_key TEXT PRIMARY KEY,product TEXT,payload TEXT);
CREATE TABLE IF NOT EXISTS runs(run_id INTEGER PRIMARY KEY,stage TEXT,payload TEXT);
CREATE TABLE IF NOT EXISTS failure_queue(failure_id TEXT PRIMARY KEY,kind TEXT,geometry_id TEXT,payload TEXT,resolved INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS route_section_matches(route_section_id TEXT,source_section_id TEXT,status TEXT,distance_m REAL,payload TEXT,PRIMARY KEY(route_section_id,source_section_id));
CREATE TABLE IF NOT EXISTS checkpoints(name TEXT PRIMARY KEY,value TEXT);
""")
    return db


def canonical(geometry, crs=32648):
    rev = tuple(geometry.coords) > tuple(reversed(geometry.coords))
    g = reverse(geometry) if rev else geometry
    return digest([g.wkb.hex(), crs, VERSION]), g, rev


def add_section(
    db,
    source_id,
    geometry,
    kind,
    payload,
    *,
    category=None,
    partition="priority",
    priority=0,
    disposition="eligible",
):
    crs = 32648 if kind == "route" else 32600 + int((geometry.centroid.x + 180) / 6) + 1
    gid, g, rev = canonical(geometry, crs)
    f = projection(crs).transform
    line = transform(f, g)
    length = line.length
    sid = (
        kind
        + ":"
        + digest(
            [source_id, payload.get("source_version"), geometry.wkb.hex(), VERSION]
        )[:32]
    )
    db.execute(
        "INSERT OR IGNORE INTO sections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            sid,
            source_id,
            gid,
            str(payload.get("source_version", "")),
            kind,
            category,
            partition,
            payload.get("direction", "source_geometry_order"),
            rev,
            encode(payload),
            disposition,
        ),
    )
    if disposition == "eligible":
        spatial = (
            f"{math.floor(g.centroid.y * 20):04d}:{math.floor(g.centroid.x * 20):04d}"
        )
        db.execute(
            "INSERT OR IGNORE INTO jobs(geometry_id,geometry,crs,length_m,partition_id,priority,spatial_key) VALUES (?,?,?,?,?,?,?)",
            (gid, g.wkb, crs, length, partition, priority, spatial),
        )
    return sid, gid


def initialize(root, out):
    db = database(out)
    config = read(out / "config.json")
    if not db.execute(
        "SELECT 1 FROM checkpoints WHERE name='routes_registered'"
    ).fetchone():
        register = read(out / "route_register.json")
        for r in register["routes"]:
            add_section(
                db,
                r["register_id"],
                shape(r["geometry"]),
                "route",
                {**r, "source_version": r["geometry_hash"]},
                priority=0 if r["register_role"] == "priority" else 1,
            )
        db.execute("INSERT INTO checkpoints VALUES ('routes_registered','1')")
        db.commit()
    if not db.execute(
        "SELECT 1 FROM checkpoints WHERE name='osm_inventory_registered'"
    ).fetchone():
        inventory = Counter()
        lengths = Counter()
        for batch in pq.ParquetFile(
            root / "exports/road_experience/enriched/nationwide/base_roads.parquet"
        ).iter_batches(batch_size=2000):
            for r in batch.to_pylist():
                tags = json.loads(r["tags_json"])
                disposition = "eligible"
                if r["category"] not in config["eligible_classes"]:
                    disposition = "excluded_undeclared_class"
                elif any(
                    tags.get(k)
                    in [
                        "no",
                        "private",
                        "agricultural",
                        "forestry",
                        "delivery",
                        "customers",
                        "destination",
                        "permit",
                    ]
                    for k in ["access", "vehicle", "motor_vehicle", "motorcar"]
                ):
                    disposition = "excluded_or_restricted_car_access"
                elif any(
                    k in tags
                    for k in [
                        "access:conditional",
                        "vehicle:conditional",
                        "motor_vehicle:conditional",
                        "motorcar:conditional",
                    ]
                ):
                    disposition = "unresolved_conditional_car_access"
                elif r["geometry_type"] != "LineString":
                    disposition = "excluded_geometry_type"
                g = from_wkb(r["geometry_wkb"])
                if g.is_empty or not g.is_valid or g.length == 0:
                    disposition = "excluded_invalid_geometry"
                inventory[disposition] += 1
                lengths[disposition] += r["length_m"]
                payload = {k: v for k, v in r.items() if k != "geometry_wkb"}
                payload["source_version"] = (
                    str(r["osm_version"]) + ":" + r["osm_timestamp"]
                )
                add_section(
                    db,
                    r["entity_id"],
                    g,
                    "osm",
                    payload,
                    category=r["category"],
                    partition=r["partition_id"],
                    priority=2
                    if r["category"]
                    in [
                        "motorway",
                        "trunk",
                        "primary",
                        "motorway_link",
                        "trunk_link",
                        "primary_link",
                    ]
                    else 3,
                    disposition=disposition,
                )
            db.commit()
        db.execute("INSERT INTO checkpoints VALUES ('osm_inventory_registered','1')")
        db.commit()
        save(
            out,
            "road_inventory.json",
            {
                "source_records": sum(inventory.values()),
                "dispositions": dict(inventory),
                "source_length_m_by_disposition": dict(lengths),
                "eligible_classes": config["eligible_classes"],
                "geographic_partitions": "Source whole-way midpoint one-degree grids, not provinces or legal borders",
                "scope_limitation": "Existing extract contains major classes only. Residential, service, unclassified roads and tracks absent from this inventory are not claimed scored.",
            },
        )
    return db


def inherited_roots(root):
    roots = []
    for parent in [
        root / "exports/road_experience",
        root / "exports/route_intelligence",
    ]:
        for folder in parent.rglob("raster"):
            if any(part.lower().startswith("mapillary") for part in folder.parts):
                continue
            roots += [
                folder / kind
                for kind in ["worldcover", "dem"]
                if (folder / kind).is_dir()
            ]
    return sorted(set(roots))


def historical_cells(root):
    path = (
        root / "exports/route_intelligence/phase4_6/final/route_cell_features.parquet"
    )
    f = pq.ParquetFile(path)
    wanted = [
        k
        for k in f.schema_arrow.names
        if k
        in [
            "route_id",
            "cell_index",
            "start_m",
            "end_m",
            "midpoint_longitude",
            "midpoint_latitude",
            "road_match_status",
            "source_road_tags_json",
            "surface_context_status",
            "scenic_surface_context_unavailable",
            "supported_tunnel_excluded",
        ]
    ]
    result = {}
    for b in f.iter_batches(columns=wanted):
        for r in b.to_pylist():
            result[(r["route_id"], r["cell_index"])] = r
    return result


def compute_cells(
    gid,
    line,
    crs,
    store,
    start_index=0,
    *,
    max_cells=None,
    tunnel=False,
    historical=None,
    matcher=None,
    travel_reversed=False,
):
    n = math.ceil(line.length / 50)
    indices = np.arange(start_index, min(n, start_index + (max_cells or n)))
    starts = indices * 50.0
    ends = np.minimum(starts + 50, line.length)
    middle = (starts + ends) / 2
    inv = projection(crs, True)
    xy = get_coordinates(line_interpolate_point(line, middle))
    a = get_coordinates(line_interpolate_point(line, np.maximum(0, middle - 2)))
    b = get_coordinates(
        line_interpolate_point(line, np.minimum(line.length, middle + 2))
    )
    direction = b - a
    norm = np.linalg.norm(direction, axis=1)
    direction = np.divide(
        direction, norm[:, None], out=np.zeros_like(direction), where=norm[:, None] > 0
    )
    lon, lat = inv.transform(xy[:, 0], xy[:, 1])
    centers = np.column_stack([lon, lat])
    offsets = np.array([-250, -150, -50, 50, 150, 250])
    side = (
        xy[:, None, :]
        + np.column_stack([-direction[:, 1], direction[:, 0]])[:, None, :]
        * offsets[None, :, None]
    )
    sx, sy = inv.transform(side[:, :, 0].ravel(), side[:, :, 1].ravel())
    samplecoords = np.column_stack([sx, sy])
    wc, _, _, wc_refs = store.sample("worldcover", samplecoords)
    wc = wc.reshape(-1, 6)
    elevation, rugged, support, demref, neighbors = store.terrain(centers)
    rows = []
    for i, cell_index in enumerate(indices):
        valid = np.isfinite(wc[i]) & (norm[i] > 0)
        count = int(valid.sum())
        context = "no_established_tunnel_evidence"
        excluded = tunnel
        matched_id = None
        match_status = None
        match_distance = None
        if matcher is not None:
            from shapely.geometry import Point

            road, match_status = matcher.match(
                Point(xy[i]), -direction[i] if travel_reversed else direction[i]
            )
            if road:
                matched_id = road["source"]["entity_id"]
                match_distance = road["distance_m"]
                if road["source"]["tags"].get("tunnel") in ["yes", "building_passage"]:
                    if match_status == "supported_geometry_not_field_verified":
                        excluded = True
                        context = "supported_source_tunnel_identity"
                    else:
                        context = "tunnel_tagged_candidate_unresolved_attribution"
        if historical and int(cell_index) in historical:
            old = historical[int(cell_index)]
            # Exact coordinate/grid agreement is required before applying a historical mask.
            if (
                abs(old["midpoint_longitude"] - lon[i]) < 1e-9
                and abs(old["midpoint_latitude"] - lat[i]) < 1e-9
            ):
                context = old.get(
                    "surface_context_status", "no_established_tunnel_evidence"
                )
                tags = json.loads(old.get("source_road_tags_json") or "{}")
                excluded |= (
                    bool(old.get("supported_tunnel_excluded"))
                    or bool(old.get("scenic_surface_context_unavailable"))
                    or (
                        old.get("road_match_status")
                        == "supported_geometry_not_field_verified"
                        and tags.get("tunnel") in ["yes", "building_passage"]
                    )
                )
        if tunnel:
            context = "supported_source_tunnel_identity"
        fractions = {
            name: float(np.sum((wc[i] == code) & valid) / count) if count else None
            for name, code in [("tree", 10), ("water", 80), ("cropland", 40)]
        }
        r = {
            "cell_id": "cell:" + digest([gid, int(cell_index)])[:32],
            "geometry_id": gid,
            "cell_index": int(cell_index),
            "start_m": float(starts[i]),
            "end_m": float(ends[i]),
            "cell_length_m": float(ends[i] - starts[i]),
            "midpoint_longitude": float(lon[i]),
            "midpoint_latitude": float(lat[i]),
            "analysis_crs": f"EPSG:{crs}",
            "source_geometry_hash": gid,
            "extraction_version": VERSION,
            "roadside_sample_coverage": count / 6 if norm[i] > 0 else 0,
            "roadside_valid_samples": count,
            "roadside_tree_fraction": fractions["tree"],
            "roadside_water_fraction": fractions["water"],
            "roadside_cropland_fraction": fractions["cropland"],
            "dsm_elevation_m": float(elevation[i])
            if np.isfinite(elevation[i])
            else None,
            "dsm_ruggedness_rms_m": float(rugged[i])
            if np.isfinite(rugged[i])
            else None,
            "dsm_3x3_valid_samples": int(support[i]),
            "dsm_center_ref": demref[i],
            "dsm_neighbor_refs": neighbors[i],
            "roadside_samples": [
                {
                    "offset_left_m": int(offsets[j]),
                    "longitude": float(sx[i * 6 + j]),
                    "latitude": float(sy[i * 6 + j]),
                    "class": int(wc[i, j]) if valid[j] else None,
                    "native_pixel_ref": wc_refs[i * 6 + j],
                }
                for j in range(6)
            ],
            "surface_context_status": context,
            "supported_tunnel_excluded": bool(excluded),
            "visibility_status": "nearby_landscape_proxy_not_road_visible",
            "raw_sampled_fractions": fractions,
            "raw_dsm_ruggedness_rms_m": float(rugged[i])
            if np.isfinite(rugged[i])
            else None,
        }
        if excluded:
            r.update(
                roadside_tree_fraction=None,
                roadside_water_fraction=None,
                roadside_cropland_fraction=None,
                roadside_sample_coverage=0,
                dsm_ruggedness_rms_m=None,
            )
        r.update(
            matched_source_osm_id=matched_id,
            road_match_status=match_status,
            source_road_distance_m=match_distance,
        )
        rows.append(r)
    return rows


def score_geometry(db, gid, policy):
    # Indexed per-geometry query, never a global per-cell table scan.
    rows = [
        json.loads(r[0])
        for r in db.execute(
            "SELECT payload FROM cells WHERE geometry_id=? ORDER BY cell_index", (gid,)
        )
    ]
    result = validated_scenic(rows, policy)
    result["component_contributions"] = {
        k: {
            "conditional_points": 100
            * policy["scenic_weights"][k]
            * result["component_coverage"][k]
            * result["component_means"][k]
            / result["evidence_weight_coverage"]
            if result["component_means"][k] is not None
            and result["evidence_weight_coverage"]
            else None,
            "known_lower_bound_points": 100
            * policy["scenic_weights"][k]
            * result["component_coverage"][k]
            * result["component_means"][k]
            if result["component_means"][k] is not None
            else 0,
            "unobserved_upper_addition_points": 100
            * policy["scenic_weights"][k]
            * (1 - result["component_coverage"][k]),
        }
        for k in policy["scenic_weights"]
    }
    result["contributions_scope"] = (
        "Unconditional observed contributions to lower bound; unknown components are separately bounded, not observed zeros."
    )
    result["supported_tunnel_excluded_m"] = sum(
        r["cell_length_m"] for r in rows if r["supported_tunnel_excluded"]
    )
    result["uncertain_surface_context_m"] = sum(
        r["cell_length_m"]
        for r in rows
        if r["surface_context_status"]
        == "tunnel_tagged_candidate_unresolved_attribution"
    )
    return result


def record_assets(db, store):
    if not hasattr(store, "recorded_asset_keys"):
        store.recorded_asset_keys = {
            r[0] for r in db.execute("SELECT cache_key FROM raster_assets")
        }
    pending = [a for a in store.assets if a["key"] not in store.recorded_asset_keys]
    db.executemany(
        "INSERT OR IGNORE INTO raster_assets VALUES (?,?,?)",
        [
            (
                a["key"],
                a["receipt"]["source"],
                encode(
                    {
                        **a["receipt"],
                        "local_path": str(a["path"]),
                        "native_transform": list(a["transform"]),
                        "shape": list(a["shape"]),
                        "pixel_reference_format": "[asset key, native cropped row, native cropped column]",
                    }
                ),
            )
            for a in pending
        ],
    )
    store.recorded_asset_keys.update(a["key"] for a in pending)
    for fail in store.failures:
        db.execute(
            "INSERT OR IGNORE INTO failure_queue VALUES (?,?,?,?,0)",
            (digest(fail), "raster_acquisition", None, encode(fail)),
        )


def export_incremental(db, out):
    start = time.perf_counter()
    folder = out / "parquet"
    folder.mkdir(exist_ok=True)
    for table in ["cells", "score_events"]:
        key = "export_" + table
        row = db.execute(
            "SELECT value FROM checkpoints WHERE name=?", (key,)
        ).fetchone()
        last = int(row[0]) if row else 0
        while True:
            cursor = db.execute(
                "SELECT rowid,* FROM "
                + table
                + " WHERE rowid>? ORDER BY rowid LIMIT 10000",
                (last,),
            )
            [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            if not rows:
                break
            # Keep a fixed portable schema; the JSON payload retains full raw observations.
            records = [
                {
                    "sequence": int(r[0]),
                    "record_id": str(r[1]),
                    "geometry_id": str(r[2]),
                    "payload_json": r[-1],
                }
                for r in rows
            ]
            schema = pa.schema(
                [
                    ("sequence", pa.int64()),
                    ("record_id", pa.string()),
                    ("geometry_id", pa.string()),
                    ("payload_json", pa.string()),
                ]
            )
            target = (
                folder
                / f"{table}-{int(rows[0][0]):012d}-{int(rows[-1][0]):012d}.parquet"
            )
            temp = target.with_suffix(".tmp.parquet")
            pq.write_table(
                pa.Table.from_pylist(records, schema=schema), temp, compression="zstd"
            )
            temp.replace(target)
            last = int(rows[-1][0])
            db.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?,?)", (key, str(last))
            )
    db.commit()
    return time.perf_counter() - start


def route_matcher(root, geographic, crs, *, source_path=None):
    from happy_path.pretrip.matching import DirectedIndex, spatial_filter

    rows = pq.read_table(
        source_path
        or root / "exports/road_experience/enriched/nationwide/base_roads.parquet",
        filters=spatial_filter(geographic.bounds),
        columns=[
            "entity_id",
            "name",
            "tags_json",
            "geometry_wkb",
            "osm_version",
            "osm_timestamp",
        ],
    ).to_pylist()
    for row in rows:
        row["tags"] = json.loads(row["tags_json"])
    projection = Transformer.from_crs(4326, crs, always_xy=True).transform
    return DirectedIndex(
        rows, [transform(projection, from_wkb(r["geometry_wkb"])) for r in rows]
    )


def process(root, out, *, online=False, stage="baseline", limit=None, retry=False):
    outer_start = time.perf_counter()
    config = read(out / "config.json")
    if (
        config["version"] != VERSION
        or config["cell_spacing_m"] != 50
        or config["offsets_m"] != [-250, -150, -50, 50, 150, 250]
        or config["workers"] != 1
    ):
        raise ValueError(
            "This extraction recipe supports the registered 50m / six-offset grid and one worker only; use a new version for a changed sampling policy"
        )
    db = initialize(root, out)
    db.execute(
        "CREATE TABLE IF NOT EXISTS score_events(event_id INTEGER PRIMARY KEY,geometry_id TEXT,created_at TEXT,payload TEXT)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS job_pending_queue ON jobs(status,priority,partition_id,spatial_key,geometry_id)"
    )
    db.execute("UPDATE jobs SET status='pending' WHERE status='running'")
    db.execute(
        "UPDATE jobs SET status='processed_tunnel_excluded' WHERE status='processed_missing_inputs' AND json_extract(score_json,'$.supported_tunnel_excluded_m')>=length_m-0.000001"
    )
    if retry:
        db.execute(
            "UPDATE jobs SET status='pending',last_cell=-1 WHERE status='processed_missing_inputs'"
        )
    db.commit()
    if stage == "nationwide" and (
        not (out / "benchmark_results.json").exists()
        or not read(out / "benchmark_results.json").get("benchmark_complete")
    ):
        raise RuntimeError(
            "Complete priority baseline benchmark before nationwide expansion"
        )
    inventory_init_s = time.perf_counter() - outer_start
    index_start = time.perf_counter()
    store = RasterStore(
        Path(config["cache_directory"]),
        inherited_roots(root),
        online=online,
        budget_bytes=config["cache_budget_bytes"],
        reserve_bytes=config["free_space_reserve_bytes"],
        ram_bytes=config["raster_ram_bytes"],
    )
    old = historical_cells(root)
    raster_index_init_s = time.perf_counter() - index_start
    metrics = {
        "stage": stage,
        "started_at": now(),
        "worker_count": 1,
        "jobs_processed": 0,
        "cells_processed": 0,
        "unique_length_m": 0.0,
        "sampled_cell_length_m": 0.0,
        "inventory_init_s": inventory_init_s,
        "raster_index_and_mask_init_s": raster_index_init_s,
        "extraction_s": 0.0,
        "database_write_s": 0.0,
        "score_aggregation_s": 0.0,
        "export_s": 0.0,
        "peak_worker_rss_bytes": psutil.Process().memory_info().rss,
        "minimum_system_available_bytes": psutil.virtual_memory().available,
    }
    start_run = time.perf_counter()
    pending_commits = 0
    stop = None
    try:
        while True:
            if limit is not None and metrics["jobs_processed"] >= limit:
                break
            if stage == "baseline":
                where = "priority<=1"
            elif stage == "benchmark_sample":
                db.execute(
                    "CREATE TEMP TABLE IF NOT EXISTS benchmark_selection(geometry_id TEXT PRIMARY KEY)"
                )
                if not db.execute(
                    "SELECT count(*) FROM benchmark_selection"
                ).fetchone()[0]:
                    db.execute(
                        "INSERT INTO benchmark_selection SELECT geometry_id FROM (SELECT geometry_id,ROW_NUMBER() OVER(PARTITION BY partition_id ORDER BY priority,spatial_key,geometry_id) rn,partition_id FROM jobs WHERE priority>=2) ORDER BY rn,partition_id LIMIT 1000"
                    )
                where = "geometry_id IN (SELECT geometry_id FROM benchmark_selection)"
            else:
                where = "priority>=2"
            jobs = db.execute(
                "SELECT geometry_id,geometry,crs,length_m,last_cell FROM jobs WHERE status='pending' AND "
                + where
                + " ORDER BY priority,partition_id,spatial_key LIMIT 50"
            ).fetchall()
            if not jobs:
                break
            for gid, wkb, crs, length, last in jobs:
                if limit is not None and metrics["jobs_processed"] >= limit:
                    break
                if psutil.virtual_memory().available < 256 * 1024**2:
                    raise StorageBudgetError(
                        "System available RAM fell below 256MiB; safe transaction checkpoint retained"
                    )
                if shutil.disk_usage(out).free < config["free_space_reserve_bytes"]:
                    raise StorageBudgetError(
                        "Free-space reserve reached before the next database job; committed checkpoint retained"
                    )
                if (out / "scenic.sqlite").stat().st_size > config[
                    "database_budget_bytes"
                ]:
                    raise StorageBudgetError(
                        "Configured SQLite database size budget reached"
                    )
                sources = db.execute(
                    "SELECT section_id,kind,payload,source_reversed FROM sections WHERE geometry_id=? AND disposition='eligible'",
                    (gid,),
                ).fetchall()
                payload = json.loads(sources[0][2])
                kind = sources[0][1]
                canonical_reversed = bool(sources[0][3])
                g = from_wkb(wkb)
                forward = projection(crs).transform
                line = transform(forward, g)
                tags = (
                    json.loads(payload.get("tags_json", "{}")) if kind == "osm" else {}
                )
                tunnel = tags.get("tunnel") in ["yes", "building_passage"]
                historical = (
                    {
                        idx: r
                        for (rid, idx), r in old.items()
                        if rid == payload.get("historical_route_id")
                    }
                    if kind == "route" and not canonical_reversed
                    else None
                )
                matcher = route_matcher(root, g, crs) if kind == "route" else None
                db.execute(
                    "UPDATE jobs SET status='running' WHERE geometry_id=?", (gid,)
                )
                if retry and last == -1:
                    db.execute("DELETE FROM cells WHERE geometry_id=?", (gid,))
                for offset in range(last + 1, math.ceil(length / 50), 1000):
                    begin = time.perf_counter()
                    old_acquisition = store.acquisition_s
                    rows = compute_cells(
                        gid,
                        line,
                        crs,
                        store,
                        offset,
                        max_cells=1000,
                        tunnel=tunnel,
                        historical=historical,
                        matcher=matcher,
                        travel_reversed=canonical_reversed,
                    )
                    metrics["extraction_s"] += (
                        time.perf_counter()
                        - begin
                        - (store.acquisition_s - old_acquisition)
                    )
                    begin = time.perf_counter()
                    db.executemany(
                        "INSERT OR REPLACE INTO cells VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [
                            (
                                r["cell_id"],
                                gid,
                                r["cell_index"],
                                r["start_m"],
                                r["end_m"],
                                r["cell_length_m"],
                                r["midpoint_longitude"],
                                r["midpoint_latitude"],
                                r["roadside_tree_fraction"],
                                r["roadside_water_fraction"],
                                r["roadside_cropland_fraction"],
                                r["dsm_ruggedness_rms_m"],
                                r["roadside_sample_coverage"],
                                r["supported_tunnel_excluded"],
                                encode(r),
                            )
                            for r in rows
                        ],
                    )
                    db.execute(
                        "UPDATE jobs SET last_cell=? WHERE geometry_id=?",
                        (rows[-1]["cell_index"], gid),
                    )
                    metrics["database_write_s"] += time.perf_counter() - begin
                    metrics["cells_processed"] += len(rows)
                    metrics["sampled_cell_length_m"] += sum(
                        r["cell_length_m"] for r in rows
                    )
                    metrics["peak_worker_rss_bytes"] = max(
                        metrics["peak_worker_rss_bytes"],
                        psutil.Process().memory_info().rss,
                    )
                    metrics["minimum_system_available_bytes"] = min(
                        metrics["minimum_system_available_bytes"],
                        psutil.virtual_memory().available,
                    )
                    if len(rows) >= 1000:
                        record_assets(db, store)
                        db.commit()
                begin = time.perf_counter()
                score = score_geometry(db, gid, config["scoring"])
                metrics["score_aggregation_s"] += time.perf_counter() - begin
                status = (
                    "processed_scored"
                    if score["ranking_eligible"]
                    else "processed_tunnel_excluded"
                    if score["supported_tunnel_excluded_m"] >= length - 1e-6
                    else "processed_missing_inputs"
                )
                db.execute(
                    "UPDATE jobs SET status=?,score_json=?,error=NULL WHERE geometry_id=?",
                    (status, encode(score), gid),
                )
                db.execute(
                    "INSERT INTO score_events(geometry_id,created_at,payload) VALUES (?,?,?)",
                    (
                        gid,
                        now(),
                        encode(
                            {
                                "geometry_id": gid,
                                "length_m": length,
                                "status": status,
                                "scenic": score,
                                "version": VERSION,
                            }
                        ),
                    ),
                )
                metrics["jobs_processed"] += 1
                metrics["unique_length_m"] += length
                pending_commits += 1
                if pending_commits >= 50 or kind == "route":
                    tick = time.perf_counter()
                    record_assets(db, store)
                    db.commit()
                    metrics["database_write_s"] += time.perf_counter() - tick
                    pending_commits = 0
                    if metrics["jobs_processed"] % 500 == 0 or kind == "route":
                        save(
                            out,
                            "active_run_checkpoint.json",
                            {
                                **metrics,
                                "checkpoint_at": now(),
                                "wall_s": time.perf_counter() - outer_start,
                                "acquisition_s": store.acquisition_s,
                                "new_cache_bytes": store.acquired_bytes,
                                "network": store.network.snapshot(),
                                "state": "committed checkpoint, not a completed run",
                            },
                        )
                        print(
                            json.dumps(
                                {
                                    "stage": stage,
                                    "completed_jobs": metrics["jobs_processed"],
                                    "cells": metrics["cells_processed"],
                                    "km": round(metrics["unique_length_m"] / 1000, 2),
                                    "elapsed_s": round(
                                        time.perf_counter() - start_run, 1
                                    ),
                                    "last_geometry": gid,
                                    "score_status": status,
                                }
                            ),
                            flush=True,
                        )
                if metrics["jobs_processed"] % 1000 == 0:
                    metrics["export_s"] += export_incremental(db, out)
        record_assets(db, store)
        db.commit()
        metrics["export_s"] += export_incremental(db, out)
    except StorageBudgetError as error:
        stop = str(error)
        record_assets(db, store)
        db.commit()
        metrics["export_s"] += export_incremental(db, out)
    finally:
        memory = psutil.Process().memory_info()
        # Windows exposes an OS-maintained process lifetime working-set peak.
        # Keep periodic samples separate on platforms without that counter.
        metrics["os_process_lifetime_peak_rss_bytes"] = getattr(
            memory, "peak_wset", None
        )
        metrics.update(
            wall_s=time.perf_counter() - outer_start,
            processing_wall_s=time.perf_counter() - start_run,
            acquisition_s=store.acquisition_s,
            new_cache_bytes=store.acquired_bytes,
            native_uncompressed_window_bytes=store.source_bytes,
            network=store.network.snapshot(),
            cache_reads=store.cache_reads,
            cache_bytes=store.owned_bytes,
            stop_reason=stop,
            ended_at=now(),
        )
        metrics["cells_per_second"] = (
            metrics["cells_processed"] / metrics["wall_s"]
            if metrics["wall_s"]
            else None
        )
        metrics["km_per_second"] = (
            metrics["sampled_cell_length_m"] / 1000 / metrics["wall_s"]
            if metrics["wall_s"]
            else None
        )
        metrics["sizes"] = {
            str(p.relative_to(out)): p.stat().st_size
            for p in out.glob("scenic.sqlite*")
        }
        db.execute(
            "INSERT INTO runs(stage,payload) VALUES (?,?)", (stage, encode(metrics))
        )
        db.commit()
        record_assets(db, store)
        db.commit()
        store.close()
        db.close()
        save(out, "runs/" + stage + "-" + digest(metrics)[:20] + ".json", metrics)
        save(out, "last_progress.json", metrics)
    return metrics


def dispatch(root, out, args):
    if args.stage == "baseline":
        return process(
            root,
            out,
            online=args.online,
            stage="baseline",
            limit=args.limit,
            retry=args.retry_missing,
        )
    if args.stage == "nationwide":
        return process(
            root,
            out,
            online=args.online,
            stage="nationwide",
            limit=args.limit,
            retry=args.retry_missing,
        )
    if args.stage == "benchmark":
        from happy_path.pretrip.raster_scenic_reports import benchmark

        return benchmark(root, out, args.online)
    if args.stage == "report":
        from happy_path.pretrip.raster_scenic_reports import report

        return report(root, out)
    if args.stage == "query":
        db = sqlite3.connect(
            "file:" + (out / "scenic.sqlite").as_posix() + "?mode=ro", uri=True
        )
        rows = db.execute(
            "SELECT s.section_id,s.source_id,s.disposition,j.status,j.score_json,s.payload,s.geometry_id FROM sections s LEFT JOIN jobs j USING(geometry_id) WHERE s.source_id=? OR s.section_id=? LIMIT 100",
            (args.entity_id, args.entity_id),
        ).fetchall()
        print(
            json.dumps(
                [
                    {
                        "section_id": sid,
                        "source_id": source,
                        "disposition": disposition,
                        "status": status,
                        "scenic": json.loads(score) if score else None,
                        "source_record": json.loads(payload),
                        "geometry_id": gid,
                    }
                    for sid, source, disposition, status, score, payload, gid in rows
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        db.close()
