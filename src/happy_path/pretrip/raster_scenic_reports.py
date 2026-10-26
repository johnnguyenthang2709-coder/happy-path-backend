"""Measured baseline/national raster benchmarks and portable inspection outputs."""

import itertools
import json
import math
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Geod
from shapely import from_wkb
from shapely.geometry import LineString, box, mapping, shape
from shapely.ops import transform, unary_union

from .raster_scenic import VERSION, RasterStore, checksum, digest
from .raster_scenic_jobs import (
    compute_cells,
    database,
    encode,
    process,
    projection,
    read,
    save,
)
from .scoring import scenic_score


def asset_size(folder):
    return (
        sum(p.stat().st_size for p in Path(folder).rglob("*") if p.is_file())
        if Path(folder).exists()
        else 0
    )


def unique_inventory_length(db, out):
    if (out / "unique_road_length.json").exists():
        return read(out / "unique_road_length.json")
    db.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS job_spatial USING rtree(id,xmin,xmax,ymin,ymax)"
    )
    if not db.execute("SELECT count(*) FROM job_spatial").fetchone()[0]:
        cursor = db.execute("SELECT rowid,geometry FROM jobs WHERE priority>=2")
        while batch := cursor.fetchmany(2000):
            vals = []
            for rid, wkb in batch:
                g = from_wkb(wkb)
                xmin, ymin, xmax, ymax = g.bounds
                vals.append((rid, xmin, xmax, ymin, ymax))
            db.executemany(
                "INSERT OR REPLACE INTO job_spatial VALUES (?,?,?,?,?)", vals
            )
        db.commit()
    bounds = db.execute(
        "SELECT min(xmin),min(ymin),max(xmax),max(ymax) FROM job_spatial"
    ).fetchone()
    geod = Geod(ellps="WGS84")
    total = 0
    tiles = []
    start = time.perf_counter()

    def region(bounds, depth=0):
        w, s, e, n = bounds
        query = "SELECT j.geometry FROM job_spatial r JOIN jobs j ON j.rowid=r.id WHERE r.xmin<=? AND r.xmax>=? AND r.ymin<=? AND r.ymax>=?"
        params = (e, w, n, s)
        count = db.execute(
            "SELECT count(*) FROM job_spatial WHERE xmin<=? AND xmax>=? AND ymin<=? AND ymax>=?",
            params,
        ).fetchone()[0]
        if count > 5000 and depth < 6:
            x = (w + e) / 2
            y = (s + n) / 2
            return sum(
                region(b, depth + 1)
                for b in [(w, s, x, y), (x, s, e, y), (w, y, x, n), (x, y, e, n)]
            )
        polygon = box(w, s, e, n)
        excluded_boundary = unary_union(
            [LineString([(e, s), (e, n)]), LineString([(w, n), (e, n)])]
        )
        geoms = []
        for (wkb,) in db.execute(query, params):
            g = from_wkb(wkb).intersection(polygon).difference(excluded_boundary)
            if not g.is_empty:
                geoms.append(g)
        if not geoms:
            return 0.0
        return float(geod.geometry_length(unary_union(geoms)))

    for y in range(math.floor(bounds[1]), math.ceil(bounds[3])):
        for x in range(math.floor(bounds[0]), math.ceil(bounds[2])):
            length = region((x, y, x + 1, y + 1))
            if length:
                tiles.append({"grid": f"N{y:02d}E{x:03d}", "unique_geodesic_m": length})
                total += length
    result = {
        "unique_eligible_geodesic_m": total,
        "unique_source_geometry_projected_m": db.execute(
            "SELECT sum(length_m) FROM jobs WHERE priority>=2"
        ).fetchone()[0],
        "method": "Bounded tile-wise topology union of exact source centerlines, geodesic length; half-open north/east boundary ownership. No snapping or parallel-road merging. Adaptive subdivisions cap candidate geometry per working tile at 5000 except unavoidable long-way intersections.",
        "analysis_s": time.perf_counter() - start,
        "tiles": tiles,
    }
    save(out, "unique_road_length.json", result)
    return result


def cold_warm(root, out, db, online):
    # Twelve actual eligible OSM geometries, across twelve source latitude bands/partitions.
    rows = db.execute(
        "SELECT geometry_id,geometry,crs,length_m FROM (SELECT *,ROW_NUMBER() OVER(PARTITION BY partition_id ORDER BY spatial_key,geometry_id) rn FROM jobs WHERE priority>=2) WHERE rn=1 ORDER BY partition_id"
    ).fetchall()
    if not rows:
        raise ValueError("No eligible benchmark geometry")
    selected = [
        rows[int(i)]
        for i in np.linspace(0, len(rows) - 1, min(12, len(rows)), dtype=int)
    ]
    folder = out / "benchmark"
    folder.mkdir(exist_ok=True)
    cache = folder / "cold_cache"
    results = []
    fingerprints = []
    for mode in ["cold", "warm"]:
        if mode == "cold" and any(cache.glob("*.tif")):
            raise RuntimeError(
                "Cold benchmark cache is already populated; retain prior benchmark result instead of claiming cold"
            )
        start = time.perf_counter()
        store = RasterStore(
            cache,
            (),
            online=online if mode == "cold" else False,
            budget_bytes=2 * 1024**3,
            reserve_bytes=8 * 1024**3,
        )
        target = sqlite3.connect(folder / (mode + ".sqlite"))
        target.execute(
            "CREATE TABLE IF NOT EXISTS cells(cell_id TEXT PRIMARY KEY,payload TEXT)"
        )
        extraction = 0.0
        write_s = 0.0
        score_s = 0.0
        cells = 0
        length = 0.0
        scores = []
        all_hashes = []
        peak = psutil.Process().memory_info().rss
        for gid, wkb, crs, n in selected:
            line = transform(projection(crs).transform, from_wkb(wkb))
            tick = time.perf_counter()
            prior = store.acquisition_s
            rows = compute_cells(gid, line, crs, store)
            extraction += time.perf_counter() - tick - (store.acquisition_s - prior)
            tick = time.perf_counter()
            target.executemany(
                "INSERT OR REPLACE INTO cells VALUES (?,?)",
                [(r["cell_id"], encode(r)) for r in rows],
            )
            target.commit()
            write_s += time.perf_counter() - tick
            tick = time.perf_counter()
            scores.append(scenic_score(rows, read(out / "config.json")["scoring"]))
            score_s += time.perf_counter() - tick
            all_hashes.append(
                digest(
                    [
                        {
                            k: r[k]
                            for k in [
                                "cell_id",
                                "roadside_tree_fraction",
                                "roadside_water_fraction",
                                "roadside_cropland_fraction",
                                "dsm_ruggedness_rms_m",
                                "roadside_sample_coverage",
                            ]
                        }
                        for r in rows
                    ]
                )
            )
            cells += len(rows)
            length += sum(r["cell_length_m"] for r in rows)
            peak = max(peak, psutil.Process().memory_info().rss)
        target.close()
        wall = time.perf_counter() - start
        result = {
            "mode": mode,
            "geometries": len(selected),
            "cells": cells,
            "projected_work_m": length,
            "acquisition_s": store.acquisition_s,
            "extraction_s": extraction,
            "database_write_s": write_s,
            "score_aggregation_s": score_s,
            "wall_s": wall,
            "cells_per_s": cells / wall,
            "km_per_s": length / 1000 / wall,
            "network": store.network.snapshot(),
            "new_cache_bytes": store.acquired_bytes,
            "cache_bytes": store.owned_bytes,
            "peak_worker_rss_sampled_bytes": peak,
            "system_memory": psutil.virtual_memory()._asdict(),
            "database_bytes": (folder / (mode + ".sqlite")).stat().st_size,
            "failures": store.failures,
            "score_eligibility_count": sum(s["ranking_eligible"] for s in scores),
        }
        store.close()
        results.append(result)
        fingerprints.append(all_hashes)
    if fingerprints[0] != fingerprints[1]:
        raise ValueError(
            "Cold/warm actual component features differ; inspect failure queue rather than accepting a fake benchmark"
        )
    result = {
        "runs": results,
        "cold_warm_components_equal": True,
        "scope": "Twelve actual eligible geometries, geographically distributed through the available source grids. Cold uses a separate empty block cache, no inherited crops. Warm reopens that cache with acquisition disabled. Samples are neither synthetic nor formula-only timing.",
    }
    save(out, "cold_warm_results.json", result)
    return result


def benchmark(root, out, online):
    path = out / "benchmark_results.json"
    if path.exists() and read(path).get("benchmark_complete"):
        result = read(path)
        if "national_sample_runs" not in result:
            db = database(out)
            runs = [
                json.loads(p)
                for (p,) in db.execute(
                    "SELECT payload FROM runs WHERE stage='benchmark_sample' ORDER BY run_id"
                )
            ]
            db.close()
            result["national_sample_runs"] = runs
            for key in [
                "cells_processed",
                "jobs_processed",
                "sampled_cell_length_m",
                "wall_s",
                "acquisition_s",
                "extraction_s",
                "database_write_s",
                "score_aggregation_s",
                "export_s",
                "new_cache_bytes",
            ]:
                result["national_sample"][key] = sum(r.get(key, 0) for r in runs)
            sample = result["national_sample"]
            speed = sample["cells_processed"] / sample["wall_s"]
            estimate = result["national_resource_estimate"]
            estimate["measured_pipeline_cells_per_s"] = speed
            remaining = (
                estimate["estimated_total_cells_from_actual_geometry_lengths"]
                - sample["cells_processed"]
            )
            estimate["remaining_wall_hours_range"] = [
                remaining / speed / 3600 * 0.5,
                remaining / speed / 3600 * 3,
            ]
            save(out, "benchmark_results.json", result)
        return result
    db = database(out)
    if db.execute(
        "SELECT count(*) FROM jobs WHERE priority<=1 AND status IN ('pending','running')"
    ).fetchone()[0]:
        raise RuntimeError("Priority baseline is not finished")
    register = read(out / "route_register.json")
    actual = {
        (r["origin_id"], r["destination_id"])
        for r in register["routes"]
        if r["register_role"] == "priority"
    }
    expected = set(itertools.permutations(register["anchors"], 2))
    if expected - actual:
        raise RuntimeError(
            "Priority directed connections unavailable: "
            + str(sorted(expected - actual))
        )
    unique = unique_inventory_length(db, out)
    cold_path = out / "cold_warm_results.json"
    cold = read(cold_path) if cold_path.exists() else cold_warm(root, out, db, online)
    db.close()
    sample = process(root, out, online=online, stage="benchmark_sample")
    db = database(out)
    sample_runs = [
        json.loads(p)
        for (p,) in db.execute(
            "SELECT payload FROM runs WHERE stage='benchmark_sample' ORDER BY run_id"
        )
    ]
    aggregate_sample = dict(sample)
    for key in [
        "cells_processed",
        "jobs_processed",
        "sampled_cell_length_m",
        "wall_s",
        "acquisition_s",
        "extraction_s",
        "database_write_s",
        "score_aggregation_s",
        "export_s",
        "new_cache_bytes",
    ]:
        aggregate_sample[key] = sum(r.get(key, 0) for r in sample_runs)
    aggregate_sample["peak_worker_rss_bytes"] = max(
        r["peak_worker_rss_bytes"] for r in sample_runs
    )
    aggregate_sample["minimum_system_available_bytes"] = min(
        r["minimum_system_available_bytes"] for r in sample_runs
    )
    target_cells = db.execute(
        "SELECT sum(CAST((length_m+49.999999)/50 AS INTEGER)) FROM jobs WHERE priority>=2"
    ).fetchone()[0]
    recorded = db.execute(
        "SELECT sum(length_m),count(*) FROM jobs WHERE priority>=2 AND status LIKE 'processed%'"
    ).fetchone()
    remaining = (
        db.execute(
            "SELECT sum(length_m) FROM jobs WHERE priority>=2 AND status='pending'"
        ).fetchone()[0]
        or 0.0
    )
    sampled_cells = max(aggregate_sample["cells_processed"], 1)
    speed = (
        sampled_cells / aggregate_sample["wall_s"]
        if aggregate_sample["wall_s"]
        else None
    )
    dbsize = (out / "scenic.sqlite").stat().st_size
    cellcount = db.execute("SELECT count(*) FROM cells").fetchone()[0]
    bytespercell = dbsize / max(cellcount, 1)
    # Deliberately retain a broad range for more complex geometries, network/cache churn and source failures.
    estimate = {
        "national_unique_geodesic_km": unique["unique_eligible_geodesic_m"] / 1000,
        "registered_unique_geometry_projected_km": unique[
            "unique_source_geometry_projected_m"
        ]
        / 1000,
        "estimated_total_cells_from_actual_geometry_lengths": target_cells,
        "measured_processed_national_km": recorded[0] / 1000 if recorded[0] else 0,
        "remaining_projected_km": remaining / 1000,
        "measured_pipeline_cells_per_s": speed,
        "remaining_wall_hours_range": [
            max(0, target_cells - sampled_cells) / speed / 3600 * 0.5,
            max(0, target_cells - sampled_cells) / speed / 3600 * 3,
        ]
        if speed
        else None,
        "observed_database_bytes_per_cell_including_fixed_inventory": bytespercell,
        "projected_sqlite_bytes_range": [
            target_cells * bytespercell * 0.6,
            target_cells * bytespercell * 1.6,
        ],
        "assumptions": [
            "No latency extrapolation from formula evaluation alone; measured extraction/acquisition/writes/aggregation all included.",
            "The first geographically stratified 1000 source sections are not a preference ground truth.",
            "Hardware, Windows filesystem costs, source failures and raster cache misses may vary; interval is a planning range, not a confidence interval.",
            "Database-per-cell estimate includes fixed inventory overhead and is intentionally conservative; actual size budget and free space take precedence.",
            "Unique source centerline topology is distinct from summed provider-route traversals.",
        ],
    }
    result = {
        "version": VERSION,
        "benchmark_complete": not sample.get("stop_reason")
        and db.execute(
            "SELECT count(*) FROM jobs WHERE geometry_id IN (SELECT geometry_id FROM (SELECT geometry_id,ROW_NUMBER() OVER(PARTITION BY partition_id ORDER BY priority,spatial_key,geometry_id) rn,partition_id FROM jobs WHERE priority>=2) ORDER BY rn,partition_id LIMIT 1000) AND status IN ('pending','running')"
        ).fetchone()[0]
        == 0,
        "baseline_complete_directed_pairs": len(actual),
        "cold_warm": cold,
        "national_sample": aggregate_sample,
        "national_sample_runs": sample_runs,
        "national_resource_estimate": estimate,
        "hardware": read(out / "hardware.json"),
    }
    save(out, "benchmark_results.json", result)
    db.close()
    return result


def directional_views(db):
    db.executescript("""CREATE TABLE IF NOT EXISTS section_directions(section_id TEXT,direction TEXT,canonical_reversed INTEGER,source_oneway TEXT,status TEXT,PRIMARY KEY(section_id,direction));
CREATE VIEW IF NOT EXISTS directional_cells AS SELECT s.section_id,s.source_id,d.direction,c.cell_id,c.geometry_id,CASE WHEN d.canonical_reversed=1 THEN j.length_m-c.end_m ELSE c.start_m END travel_start_m,CASE WHEN d.canonical_reversed=1 THEN j.length_m-c.start_m ELSE c.end_m END travel_end_m,CASE WHEN d.canonical_reversed=1 THEN -1 ELSE 1 END canonical_left_to_travel_left_multiplier,c.payload FROM sections s JOIN section_directions d USING(section_id) JOIN jobs j USING(geometry_id) JOIN cells c USING(geometry_id);
""")
    for sid, kind, rev, payload in db.execute(
        "SELECT section_id,kind,source_reversed,payload FROM sections WHERE disposition='eligible'"
    ):
        p = json.loads(payload)
        if kind == "route":
            directions = [
                (
                    p.get("direction", "provider_capture_direction"),
                    rev,
                    None,
                    "captured_provider_direction_not_navigation_certified",
                )
            ]
        else:
            tags = json.loads(p["tags_json"])
            one = tags.get("oneway")
            conditional = tags.get("oneway:conditional")
            allowed = (
                [False]
                if one in ["yes", "1", "true"]
                else [True]
                if one == "-1"
                else [False, True]
            )
            directions = [
                (
                    "reverse_source_order" if backwards else "source_order",
                    bool(rev) ^ backwards,
                    one,
                    "conditional_direction_unresolved"
                    if conditional or one in ["reversible", "alternating"]
                    else "source_reported_direction_only_not_turn_graph_validation",
                )
                for backwards in allowed
            ]
        db.executemany(
            "INSERT OR REPLACE INTO section_directions VALUES (?,?,?,?,?)",
            [(sid, d, int(r), one, status) for d, r, one, status in directions],
        )
    db.commit()


def report(root, out):
    db = database(out)
    directional_views(db)
    # Retain every spatial association, including contradictions and ambiguity.
    # They are candidates for feature reuse, never permission to snap a new route.
    for sid, gid in db.execute(
        "SELECT section_id,geometry_id FROM sections WHERE kind='route'"
    ).fetchall():
        groups = defaultdict(list)
        for (payload,) in db.execute(
            "SELECT payload FROM cells WHERE geometry_id=? ORDER BY cell_index", (gid,)
        ):
            cell = json.loads(payload)
            if cell.get("matched_source_osm_id"):
                groups[cell["matched_source_osm_id"]].append(cell)
        for source_id, cells in groups.items():
            sections = db.execute(
                "SELECT section_id FROM sections WHERE kind='osm' AND source_id=?",
                (source_id,),
            ).fetchall()
            statuses = sorted({c["road_match_status"] for c in cells})
            for (source_sid,) in sections:
                db.execute(
                    "INSERT OR REPLACE INTO route_section_matches VALUES (?,?,?,?,?)",
                    (
                        sid,
                        source_sid,
                        statuses[0]
                        if len(statuses) == 1
                        else "mixed_attribution_status",
                        max(c["source_road_distance_m"] for c in cells),
                        encode(
                            {
                                "source_id": source_id,
                                "statuses": statuses,
                                "sampled_support_m": sum(
                                    c["cell_length_m"] for c in cells
                                ),
                                "sampled_chainage_intervals_m": [
                                    [c["start_m"], c["end_m"]] for c in cells
                                ],
                                "chainage_reference": "canonical geometry; use section_directions for travel direction",
                                "reuse_policy": "Exact canonical geometry reuses cells. Source identity association alone cannot reuse features from a different road alignment or grid.",
                            }
                        ),
                    ),
                )
    db.commit()
    from .raster_scenic_validation import validate

    validate(root, out, db)
    counts = {
        s: n for s, n in db.execute("SELECT status,count(*) FROM jobs GROUP BY status")
    }
    dispositions = {
        s: n
        for s, n in db.execute(
            "SELECT disposition,count(*) FROM sections WHERE kind='osm' GROUP BY disposition"
        )
    }
    byclass = []
    for cat, total, done, eligible, meters in db.execute(
        "SELECT s.category,count(*),sum(j.status LIKE 'processed%'),sum(j.status='processed_scored'),sum(CASE WHEN j.status LIKE 'processed%' THEN j.length_m ELSE 0 END) FROM sections s LEFT JOIN jobs j USING(geometry_id) WHERE s.kind='osm' GROUP BY s.category"
    ):
        byclass.append(
            {
                "category": cat,
                "source_sections": total,
                "processed_sections": done or 0,
                "score_eligible_sections": eligible or 0,
                "processed_source_traversal_m": meters or 0,
            }
        )
    grid = []
    for part, number, done, scored, length in db.execute(
        "SELECT s.partition_id,count(*),sum(j.status LIKE 'processed%'),sum(j.status='processed_scored'),sum(CASE WHEN j.status LIKE 'processed%' THEN j.length_m ELSE 0 END) FROM sections s LEFT JOIN jobs j USING(geometry_id) WHERE s.kind='osm' GROUP BY s.partition_id"
    ):
        grid.append(
            {
                "grid": part,
                "source_sections": number,
                "processed_sections": done or 0,
                "score_eligible_sections": scored or 0,
                "processed_source_traversal_m": length or 0,
            }
        )
    save(
        out,
        "coverage.json",
        {
            "jobs_by_status": counts,
            "osm_dispositions": dispositions,
            "by_class": byclass,
            "by_grid": grid,
            "cell_count": db.execute("SELECT count(*) FROM cells").fetchone()[0],
            "unique_processed_osm_geometry_m": db.execute(
                "SELECT sum(length_m) FROM jobs WHERE priority>=2 AND status LIKE 'processed%'"
            ).fetchone()[0],
            "scope": "Available declared major car-road classes in Geofabrik Vietnam snapshot; one-degree midpoint grids, not province boundaries. Class/grid source traversals can include duplicate source descriptions; unique geometry totals are separate.",
        },
    )
    assets = [json.loads(r[0]) for r in db.execute("SELECT payload FROM raster_assets")]
    save(out, "raster_manifest.json", assets)
    failures = [
        {
            "failure_id": r[0],
            "kind": r[1],
            "geometry_id": r[2],
            "evidence": json.loads(r[3]),
            "resolved": bool(r[4]),
        }
        for r in db.execute("SELECT * FROM failure_queue")
    ]
    save(out, "failure_queue.json", failures)
    routes = []
    for sid, gid, payload, status, score in db.execute(
        "SELECT s.section_id,s.geometry_id,s.payload,j.status,j.score_json FROM sections s JOIN jobs j USING(geometry_id) WHERE s.kind='route'"
    ):
        p = json.loads(payload)
        result = {
            "section_id": sid,
            "geometry_id": gid,
            "route_id": p["route_id"],
            "register_id": p["register_id"],
            "role": p["register_role"],
            "origin_id": p.get("origin_id"),
            "destination_id": p.get("destination_id"),
            "intent": p.get("intent", p.get("request_id")),
            "provider_distance_m": p["provider_distance_m"],
            "provider_duration_s": p["provider_duration_s"],
            "status": status,
            "scenic": json.loads(score) if score else None,
            "source_receipt": p.get("source_receipt"),
            "scope": p["scope"],
            "geometry_hash": p["geometry_hash"],
        }
        result["road_attribution_m_by_status"] = {
            status or "unmatched": length
            for status, length in db.execute(
                "SELECT json_extract(payload,'$.road_match_status'),sum(length_m) FROM cells WHERE geometry_id=? GROUP BY json_extract(payload,'$.road_match_status')",
                (gid,),
            )
        }
        result["navigation_readiness"] = (
            "Not established by raster scoring; source one-way contradictions and map-matching uncertainty remain explicit in road_attribution_m_by_status and route_section_matches."
        )
        routes.append(result)
    save(out, "route_features.json", routes)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "route_id": r["route_id"],
                    "register_id": r["register_id"],
                    "geometry_id": r["geometry_id"],
                    "status": r["status"],
                    "payload_json": encode(r),
                }
                for r in routes
            ]
        ),
        out / "route_features.parquet",
        compression="zstd",
    )
    comparisons = []
    for a, b in itertools.combinations(
        [r for r in routes if r["role"] == "priority"], 2
    ):
        if (a["origin_id"], a["destination_id"]) != (
            b["origin_id"],
            b["destination_id"],
        ):
            continue
        ids = []
        for r in [a, b]:
            ids.append(
                {
                    json.loads(c[0]).get("matched_source_osm_id")
                    for c in db.execute(
                        "SELECT payload FROM cells WHERE geometry_id=?",
                        (r["geometry_id"],),
                    )
                    if json.loads(c[0]).get("road_match_status")
                    == "supported_geometry_not_field_verified"
                }
                - {None}
            )
        comparisons.append(
            {
                "route_a": a["register_id"],
                "route_b": b["register_id"],
                "distance_delta_m": a["provider_distance_m"] - b["provider_distance_m"],
                "provider_eta_delta_s": a["provider_duration_s"]
                - b["provider_duration_s"],
                "scenic_a": a["scenic"],
                "scenic_b": b["scenic"],
                "shared_supported_source_sections": sorted(ids[0] & ids[1]),
                "differentiating_source_sections_a": sorted(ids[0] - ids[1]),
                "differentiating_source_sections_b": sorted(ids[1] - ids[0]),
                "scope": "Geometry-supported source-section associations only, neither field-verified road identity nor human preference validation.",
            }
        )
    save(out, "baseline_comparisons.json", comparisons)
    rows = [
        "# Registered route landscape comparisons",
        "",
        "These are provisional nearby-landscape proxies from 2021 rasters. Numerical differentiation is not visibility verification, navigation readiness or empirical validation of preferences. Provider ETA is a captured estimate, not a guaranteed trip time.",
        "",
        "| Connection / intent | km | Provider min | Scenic | Input coverage | Unknown-input bounds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in routes:
        if r["role"] != "priority":
            continue
        s = r["scenic"] or {}
        bounds = s.get("missing_evidence_bounds", [None, None])
        score = s.get("score")
        rows.append(
            f"| {r['origin_id']} → {r['destination_id']} / {r['intent']} | {r['provider_distance_m'] / 1000:.2f} | {r['provider_duration_s'] / 60:.1f} | {score:.3f} | {s.get('evidence_weight_coverage', 0):.3%} | {bounds} |"
            if score is not None
            else f"| {r['origin_id']} → {r['destination_id']} / {r['intent']} | {r['provider_distance_m'] / 1000:.2f} | {r['provider_duration_s'] / 60:.1f} | unknown | {s.get('evidence_weight_coverage', 0):.3%} | {bounds} |"
        )
    rows += [
        "",
        "baseline_comparisons.json lists same-direction, same-endpoint differences in provider distance/ETA and shared/differentiating supported OSM section identities. These identity sets do not certify identical sampled road length. route_features.json retains all component means, contributions, support, unresolved surface context and road-attribution status lengths. Narrow missing-input bounds retain source age, DSM surface-object effects, alignment, visibility and human preference uncertainty. No winner or production ranking is assigned.",
    ]
    (out / "baseline_comparison_report.md").write_text(
        "\n".join(rows) + "\n", encoding="utf8"
    )
    # A flat latest-section snapshot complements append-only raw event payloads.
    fields = [
        ("section_id", pa.string()),
        ("source_id", pa.string()),
        ("geometry_id", pa.string()),
        ("road_class", pa.string()),
        ("grid", pa.string()),
        ("status", pa.string()),
        ("disposition", pa.string()),
        ("length_m", pa.float64()),
        ("score", pa.float64()),
        ("conditional_score", pa.float64()),
        ("evidence_weight_coverage", pa.float64()),
        ("missing_lower", pa.float64()),
        ("missing_upper", pa.float64()),
    ]
    for component in ["tree_cover", "water", "cropland", "ruggedness"]:
        fields += [
            (component + "_" + suffix, pa.float64())
            for suffix in ["mean", "coverage", "conditional_points"]
        ]
    schema = pa.schema(fields)
    cursor = db.execute(
        "SELECT s.section_id,s.source_id,s.geometry_id,s.category,s.partition_id,j.status,s.disposition,j.length_m,j.score_json FROM sections s LEFT JOIN jobs j USING(geometry_id) ORDER BY s.section_id"
    )
    with pq.ParquetWriter(
        out / "section_scores.parquet", schema, compression="zstd"
    ) as writer:
        while batch := cursor.fetchmany(2000):
            records = []
            for (
                sid,
                source,
                gid,
                category,
                grid_id,
                status,
                disposition,
                length,
                payload,
            ) in batch:
                s = json.loads(payload) if payload else {}
                bounds = s.get("missing_evidence_bounds", [None, None])
                row = dict(
                    zip(
                        [f[0] for f in fields[:8]],
                        [
                            sid,
                            source,
                            gid,
                            category,
                            grid_id,
                            status,
                            disposition,
                            length,
                        ],
                    )
                )
                row.update(
                    score=s.get("score"),
                    conditional_score=s.get("observed_score"),
                    evidence_weight_coverage=s.get("evidence_weight_coverage"),
                    missing_lower=bounds[0],
                    missing_upper=bounds[1],
                )
                for component in ["tree_cover", "water", "cropland", "ruggedness"]:
                    row[component + "_mean"] = s.get("component_means", {}).get(
                        component
                    )
                    row[component + "_coverage"] = s.get("component_coverage", {}).get(
                        component
                    )
                    row[component + "_conditional_points"] = (
                        s.get("component_contributions", {})
                        .get(component, {})
                        .get("conditional_points")
                    )
                records.append(row)
            writer.write_table(pa.Table.from_pylist(records, schema=schema))
    # Complete portable source-section metadata without materializing national geometries.
    folder = out / "parquet"
    folder.mkdir(exist_ok=True)
    cursor = db.execute(
        "SELECT section_id,source_id,geometry_id,kind,category,partition_id,disposition,payload FROM sections ORDER BY section_id"
    )
    i = 0
    while batch := cursor.fetchmany(10000):
        columns = [d[0] for d in cursor.description]
        table = pa.Table.from_pylist(
            [dict(zip(columns, row)) for row in batch],
            schema=pa.schema([(k, pa.string()) for k in columns]),
        )
        pq.write_table(table, folder / f"sections-{i:04d}.parquet", compression="zstd")
        i += 1
    save(
        out,
        "source_manifest.json",
        {
            "inputs": read(out / "input_manifest.json"),
            "config_sha256": checksum(out / "config.json"),
            "implementation_sha256": {
                str(p.relative_to(root)): checksum(p)
                for p in [
                    root / "scripts/run_raster_scenic.py",
                    *[
                        (root / "src/happy_path/pretrip") / name
                        for name in [
                            "raster_scenic.py",
                            "raster_scenic_jobs.py",
                            "raster_scenic_reports.py",
                            "raster_scenic_validation.py",
                        ]
                    ],
                ]
            },
            "raster_assets": len(assets),
            "held_mapping_policy": "Only current captured baseline routes were reused; original audit excluded. No quarantined mapping admitted to the baseline.",
            "provider_routes": "route_register.json; original request/cache references retained",
            "source_product_vintages": [
                "ESA WorldCover 2021 v200",
                "Copernicus GLO-30 Public DSM 2021 release",
            ],
            "scores": "New provisional nearby-landscape Scenic outputs only; no production ranking or historical score overwritten.",
        },
    )
    # Reuse the existing lightweight offline geographic renderer.
    source = root / read(out / "input_manifest.json")["latest_usable_release"].replace(
        "\\", "/"
    )
    template = (source / "inspection_map.html").read_text("utf8")
    features = []
    for r in read(out / "route_register.json")["routes"]:
        current = next(v for v in routes if v["register_id"] == r["register_id"])
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(
                    shape(r["geometry"]).simplify(0.0001, preserve_topology=True)
                ),
                "properties": {
                    "name": r.get("direction", r.get("case_id", "route"))
                    + " / "
                    + r.get("intent", r.get("request_id", "")),
                    "route_id": r["route_id"],
                    "register_role": r["register_role"],
                    "origin_id": r.get("origin_id"),
                    "destination_id": r.get("destination_id"),
                    "display_geometry": "Simplified at 0.0001 degrees for this overview only; full provider geometry remains in route_register.json and SQLite",
                    "scenic": current["scenic"],
                    "scope": "Nearby-landscape proxy; not actual visible scenery, safety or navigation readiness",
                },
            }
        )
    for r in grid:
        y = int(r["grid"][1:3])
        x = int(r["grid"][4:7])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [x + 0.5, y + 0.5]},
                "properties": {
                    "name": r["grid"] + " source-grid coverage",
                    "geometry_provider": "OpenStreetMap",
                    "scope": "Grid midpoint, not a province or road entrance",
                    **r,
                },
            }
        )
    start = template.index("const data=")
    _, used = json.JSONDecoder().raw_decode(template[start + 11 :])
    html = (
        template[:start]
        + "const data="
        + json.dumps(
            {"type": "FeatureCollection", "features": features}, ensure_ascii=False
        ).replace("</", "<\\/")
        + template[start + 11 + used :]
    )
    html = html.replace("Phase 4.12", "Nationwide raster Scenic").replace(
        "HappyPath national mapping inspection", "HappyPath raster Scenic inspection"
    )
    html = html.replace(
        "Private research. Geographic positions are not verified entrances. Lines include partial named-section captures and corridor candidates. Original IDs and scope flags remain visible.",
        "Registered Goong routes and one-degree source-grid coverage. Scenic values are provisional nearby-landscape proxies, not visibility or navigation certification. Full geometry and source records remain in the dataset.",
    )
    html = html.replace(
        "Orange: inferred discovery; purple points: representative areas; purple dashed: OSM source; red dashed: quarantined history. Captured corridors do not certify named-road extent or entrances.",
        "Teal lines: captured Goong baseline and validation routes. Grid points: coverage summaries, not road locations or provinces. Select a route for component coverage and missing-input bounds; filter by endpoint ID, name or role.",
    )
    (out / "inspection_map.html").write_text(html, encoding="utf8")
    benchmark_data = (
        read(out / "benchmark_results.json")
        if (out / "benchmark_results.json").exists()
        else None
    )
    runs = [
        {"stage": stage, "metrics": json.loads(p)}
        for stage, p in db.execute("SELECT stage,payload FROM runs ORDER BY run_id")
    ]
    save(out, "run_metrics.json", runs)
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    text = f"""# Nationwide raster Scenic dataset

Isolated, opt-in output. No production API, trip session, memory/reminder code or existing numerical score was changed. These are provisional nearby-landscape proxies, not road-visible scenery, safety, current road condition, Iconic significance or calibrated human preferences.

## Scope

Priority: all 20 directed connections between TP.HCM (existing Thu Duc origin), Vũng Tàu, Đà Lạt, Nha Trang and the explicit Dầu Giây representative anchor in Đồng Nai. Provider-returned alternatives and existing ĐT725/QL27 alternatives are retained; no fabricated alternatives or seed polylines are driving geometry. The baseline does not cover every street in the five areas. Northern/central captures are separate validation references.

National inventory: existing Geofabrik OSM snapshot and the declared ten major car-road classes in config.json. Explicit no/private/purpose-restricted car access is excluded; conditional car access is unresolved. Residential, service, unclassified roads and tracks are not in this source inventory. Grid reports use midpoint 1-degree partitions, not province boundaries. road_inventory.json and coverage.json account for every original way.

## Extraction and policy

Projected cells are at most 50m. Six discrete roadside samples use ±50/150/250m offsets from the ±2m polyline tangent. The fractions are valid-sample fractions, never buffer-area fractions. Native DSM RMS is eight-neighbor squared differences from the center, with all nine valid compatible grid centers required. Crop and official-product edges recover actual neighboring pixels; no interpolation, nodata fill, bare-earth substitution or synthetic ocean elevation occurs. Initial bounded-cache implementation/results are preserved in attempts/native-v1; corrected v2 is the current dataset.

Exact forward/reverse geometry shares canonical features and direction-specific chainage/offset relationships. Canonical orientation is numeric coordinate order, never a proximity merge. A reverse grid can differ at the terminal short cell from independently partitioning from the other endpoint; historical comparisons use matching sampling locations. Parallel roads stay separate. Shared raster blocks are reused across overlapping candidates; source-section associations preserve ambiguity and oneway contradictions.

Geometry IDs (also stored as cell source_geometry_hash) hash canonical source WKB, analysis CRS and extraction version. They are extraction-recipe fingerprints, rather than pure WKB checksums. Original WKB stays in SQLite; provider geometry_hash and request references stay in route_register.json. Undefined tangents on very short closed source loops have zero roadside support and remain explicit insufficient-input cases; coincident sample coordinates in those raw records are not valid ±50/150/250m positions.

Tree .35, water .25, cropland .10 and ruggedness .30 match the original policy. Ruggedness clips RMS/10 to [0,1]. Existing scoring.py supplies support-weighted aggregation and missing-input bounds; the point-score gate stays at 80%. Supported tunnel masks exclude scored surface observations while retaining raw inputs. Candidate tunnel context remains uncertain. Missing inputs remain unknown, not observed zeros. Bounds exclude source errors, sampling errors, real visibility and preference uncertainty.

## Files and queries

Install the existing road extra and the isolated worker dependency with `uv pip install --python .venv/Scripts/python.exe -e ".[road]" -r configs/route_intelligence/raster_scenic_requirements.txt`. Exact executed package versions are in environment_manifest.json.

scenic.sqlite contains jobs, sections, cells, raster_assets, score_events, failure_queue and checkpoints. directional_cells exposes travel chainage and the left-side multiplier. Parquet files are incremental cell/score event payloads plus normalized section metadata; score events retain updates while SQLite jobs expose latest scores. route_features.parquet, route_register.json and baseline_comparisons.json hold provider-level results and shared/differentiating source sections. Original geometry WKB and geometry IDs stay in jobs, provider geometry/request references in the route register, and pixel references resolve through raster_assets. Raster metadata includes CRS, native resolution, nodata, checksums, source URL, attribution, product date and acquisition time.

From the repository root:

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe -X utf8 scripts/run_raster_scenic.py audit
.venv/Scripts/python.exe -X utf8 scripts/run_raster_scenic.py register --online
.venv/Scripts/python.exe -X utf8 scripts/run_raster_scenic.py baseline --online
.venv/Scripts/python.exe -X utf8 scripts/run_raster_scenic.py benchmark --online
.venv/Scripts/python.exe -X utf8 scripts/run_raster_scenic.py nationwide --online
.venv/Scripts/python.exe -X utf8 scripts/run_raster_scenic.py report
.venv/Scripts/python.exe -X utf8 scripts/run_raster_scenic.py query --entity-id 'osm:way/1281338779'
```

Remove --online for cached reads only. --retry-missing retries processed missing-input jobs; known failed requests remain unknown/retryable, never successfully scored just for returning a response. Interrupted running jobs resume after the last committed cell. Transactions are incremental and inserts are idempotent. Raster cache eviction affects only task-owned reproducible blocks within its verified D: directory; inherited rasters are read-only. Configurable cache/database budgets and the free-space reserve stop work with an actual checkpoint. No background completion is claimed.

Current counts: {encode(counts)}. Source dispositions: {encode(dispositions)}. Each processed section has an eligible score or an explicit missing-input/tunnel disposition. This does not imply all roads in Vietnam are scored.

Benchmarks measure actual acquisition bytes via GDAL counters, acquisition time, extraction, database writes, score aggregation, exports, memory and file sizes. Cold/warm uses real geographically distributed geometries. Resource estimates are measured planning ranges, not formula-only extrapolation. See benchmark_results.json and run_metrics.json. New regression results are recorded separately from historical test logs.

Attribution: OpenStreetMap contributors, ODbL-1.0; ESA WorldCover 2021, CC-BY-4.0, contains modified Copernicus Sentinel data processed by the ESA WorldCover consortium; Copernicus WorldDEM-30 public DSM under its public license. Goong captures remain subject to account terms; private research exports do not grant public redistribution rights.
"""
    (out / "README.md").write_text(text, encoding="utf8")
    benchmark_text = [
        "# Actual pipeline benchmark",
        "",
        "One worker on Intel Core 7 240H (10 physical / 16 logical cores), Windows 11, 16.87 GB system RAM. Exact environment and hardware snapshots are retained separately. Times include actual work, not just formula evaluation.",
        "",
        "| Run | Cells | Wall s | Acquisition s | Extraction s | SQLite write s | Scoring s | HTTP response body bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if benchmark_data:
        b = benchmark_data
        sample = dict(b["national_sample"])
        sample["network_body_bytes"] = sum(
            sum(
                m.get("downloaded_bytes", 0)
                for m in r.get("network", {}).get("methods", {}).values()
            )
            for r in b.get("national_sample_runs", [sample])
        )
        for label, r in [
            ("Cold distributed subset", b["cold_warm"]["runs"][0]),
            ("Warm identical subset", b["cold_warm"]["runs"][1]),
            ("1000 actual national source sections", sample),
        ]:
            transferred = r.get(
                "network_body_bytes",
                sum(
                    m.get("downloaded_bytes", 0)
                    for m in r.get("network", {}).get("methods", {}).values()
                ),
            )
            benchmark_text.append(
                f"| {label} | {r.get('cells', r.get('cells_processed'))} | {r['wall_s']:.3f} | {r['acquisition_s']:.3f} | {r['extraction_s']:.3f} | {r['database_write_s']:.3f} | {r['score_aggregation_s']:.3f} | {transferred} |"
            )
        estimate = b["national_resource_estimate"]
        benchmark_text += [
            "",
            f"Exact-source centerline union: {estimate['national_unique_geodesic_km']:,.3f} geodesic km. Projected registered geometry work: {estimate['registered_unique_geometry_projected_km']:,.3f} km. Estimated cell count from the actual geometry lengths: {estimate['estimated_total_cells_from_actual_geometry_lengths']:,}.",
            "",
            f"Dispersed-sample planning time range: {estimate['remaining_wall_hours_range']} hours. SQLite planning size range: {estimate['projected_sqlite_bytes_range']} bytes. These are broad planning ranges, not confidence intervals; spatially batched later throughput can be substantially faster. Live disk/cache limits take precedence.",
            "",
            "Cold/warm feature values matched exactly. The sample was resumed after a measured system-memory stop; both runs contribute to the full 1000-section benchmark. Acquisition accounts for approximately 88% of sample wall time. The worker orders jobs by priority, source grid and spatial locality, reuses raster blocks, and avoids per-cell national database scans.",
            "",
            "Peak sampled worker RSS, OS process-lifetime peak when available, system available RAM, actual HTTP counters, cache/SQLite/WAL/temp/export sizes and all substage timings are retained in benchmark_results.json, run_metrics.json and file_sizes.json. HTTP body bytes differ from compressed cache bytes and uncompressed sampled-window bytes. Unattributed setup, serialization and other overhead remain in wall time; stage times are not substituted for full-pipeline throughput.",
        ]
    benchmark_text += [
        "",
        "Later measured acquisition/extraction runs are in run_metrics.json. Source and sample-surface uncertainty are not preference confidence.",
        "The first nationwide worker was interrupted before its final timing/HTTP counter snapshot. Its committed features, native receipts and progress logs survived. interrupted_run_accounting.json records the lower-bound progress log; complete nationwide wall time and transfer-byte totals are unavailable, and must not be reconstructed as exact totals from the remaining completed runs. The complete 1000-section benchmark has both measured run records.",
    ]
    (out / "benchmark_report.md").write_text(
        "\n".join(benchmark_text) + "\n",
        encoding="utf8",
    )
    save(
        out,
        "file_sizes.json",
        {
            "database_bytes": (out / "scenic.sqlite").stat().st_size,
            "raster_cache_bytes": asset_size(out / "raster_cache"),
            "parquet_bytes": asset_size(out / "parquet"),
            "temporary_bytes": sum(
                p.stat().st_size for p in out.rglob("*.tmp*") if p.is_file()
            ),
            "wal_bytes": sum(p.stat().st_size for p in out.glob("*-wal")),
            "shared_memory_bytes": sum(p.stat().st_size for p in out.glob("*-shm")),
            "provider_cache_bytes": asset_size(out / "provider_cache"),
            "benchmark_bytes": asset_size(out / "benchmark"),
            "preserved_attempt_bytes": asset_size(out / "attempts"),
            "total_output_directory_bytes": asset_size(out),
            "available_disk_bytes": __import__("shutil").disk_usage(out).free,
        },
    )
    pending = sum(n for status, n in counts.items() if status in ["pending", "running"])
    save(
        out,
        "completion_manifest.json",
        {
            "extraction_version": VERSION,
            "all_eligible_jobs_accounted": pending == 0,
            "status": "complete_with_explicit_dispositions"
            if pending == 0
            else "checkpoint_not_complete",
            "pending_jobs": pending,
            "jobs_by_status": counts,
            "score_eligibility_is_not_full_geographic_or_navigation_verification": True,
            "resume_command": ".venv/Scripts/python.exe -X utf8 scripts/run_raster_scenic.py nationwide --online",
            "last_run": read(out / "last_progress.json"),
            "validation_status": read(out / "dataset_validation.json")["status"],
        },
    )
    # Receipts already checksum every native asset. Hash portable dataset artifacts
    # after the writer closes; do not hash an actively changing SQLite snapshot.
    output_files = [
        p
        for p in out.iterdir()
        if p.is_file()
        and p.name not in ["output_manifest.json"]
        and not p.name.endswith(("-wal", "-shm"))
    ]
    output_files += list((out / "parquet").glob("*.parquet"))
    save(
        out,
        "output_manifest.json",
        {
            "files": [
                {
                    "path": str(p.relative_to(out)),
                    "bytes": p.stat().st_size,
                    "sha256": checksum(p),
                }
                for p in output_files
            ],
            "raster_assets": "raster_manifest.json contains native asset checksums, URL, version and windows; evicted blocks remain reproducible from receipts",
            "scope": "Snapshot after report writer closure. Rerun report after any resumption to refresh dataset hashes.",
        },
    )
    return counts
