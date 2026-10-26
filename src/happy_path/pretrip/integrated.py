"""Opt-in integrated research queries; no FastAPI, solver or session integration."""

import itertools
import json
import math
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from happy_path.core.road_sources import now

import numpy as np
from shapely import from_wkb, reverse
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

from .raster_scenic import VERSION as EXTRACTION_VERSION
from .raster_scenic import RasterStore, checksum, digest
from .raster_scenic_jobs import canonical, compute_cells, projection, route_matcher
from .scoring import express_scores, iconic_score, scenic_score

VERSION = "phase4.12-integrated-v1"
AUDIT_ID = "route:16eb27ed23706ab8fa86"


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text("utf8"))


def fold(value):
    return " ".join(
        "".join(
            c
            for c in unicodedata.normalize("NFD", str(value).casefold()).replace(
                "đ", "d"
            )
            if unicodedata.category(c) != "Mn"
        ).split()
    )


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def database(out):
    out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(out / "integrated.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA cache_size=-32768")
    db.executescript(
        Path(__file__).with_name("integrated_schema.sql").read_text("utf8")
    )
    current_views(db)
    return db


def current_views(db, include_inferred=False, include_unresolved=False):
    """Connection-local current views work on old releases without modifying them.

    Discovery flags widen only geographic status. Validity and dependency masks
    apply unconditionally, including to claims, road context and access links.
    """
    inferred, unresolved = int(bool(include_inferred)), int(bool(include_unresolved))
    db.executescript(f"""
    CREATE TEMP VIEW effective_entities AS
      SELECT e.* FROM main.entities e WHERE ({inferred} OR e.inferred=0)
      AND ({unresolved} OR e.unresolved=0) AND e.kind!='original_audit_reference'
      AND NOT EXISTS(SELECT 1 FROM main.invalidations i WHERE i.output_id=e.entity_id);
    CREATE TEMP VIEW effective_routes AS
      SELECT r.* FROM main.routes r WHERE ({inferred} OR r.inferred=0)
      AND r.role!='original_audit_reference'
      AND NOT EXISTS(SELECT 1 FROM main.invalidations i WHERE i.output_id=r.route_id);
    CREATE TEMP VIEW effective_relationships AS
      SELECT s.* FROM main.relationships s JOIN effective_entities e USING(entity_id)
      WHERE ({inferred} OR s.inferred=0) AND ({unresolved} OR s.unresolved=0)
      AND s.invalidated=0 AND s.route_id!='{AUDIT_ID}'
      AND NOT EXISTS(SELECT 1 FROM main.invalidations i WHERE i.output_id IN (s.relationship_id,s.route_id))
      AND (NOT EXISTS(SELECT 1 FROM main.routes r WHERE r.route_id=s.route_id)
        OR EXISTS(SELECT 1 FROM effective_routes r WHERE r.route_id=s.route_id))
      AND (NOT EXISTS(SELECT 1 FROM main.entities e WHERE e.entity_id=s.route_id)
        OR EXISTS(SELECT 1 FROM effective_entities e WHERE e.entity_id=s.route_id));
    CREATE TEMP VIEW effective_claims AS
      SELECT c.* FROM main.claims c JOIN main.sources s USING(source_id)
      JOIN effective_entities e USING(entity_id) WHERE c.usable=1 AND s.is_public=1
      AND NOT EXISTS(SELECT 1 FROM main.invalidations i WHERE i.output_id IN (c.claim_id,c.canonical_id,c.source_id));
    CREATE TEMP VIEW effective_claim_links AS
      SELECT l.* FROM main.claim_links l JOIN effective_claims c USING(claim_id)
      JOIN effective_routes r USING(route_id) WHERE l.usable=1
      AND NOT EXISTS(SELECT 1 FROM main.invalidations i WHERE i.output_id=l.link_id)
      AND (json_extract(l.payload,'$.detail.place_link_id') IS NULL OR EXISTS(
        SELECT 1 FROM effective_place_links p WHERE p.link_id=json_extract(l.payload,'$.detail.place_link_id')))
      AND (l.application NOT IN ('general_corridor_context','historical_bounded_section_context') OR EXISTS(
        SELECT 1 FROM effective_road_links b WHERE b.route_id=l.route_id AND b.entity_id=c.entity_id));
    CREATE TEMP VIEW effective_place_links AS
      SELECT l.* FROM main.place_links l JOIN effective_entities e USING(entity_id)
      JOIN effective_routes r USING(route_id)
      WHERE ({inferred} OR l.inferred=0) AND ({unresolved} OR l.unresolved=0)
      AND NOT EXISTS(SELECT 1 FROM main.invalidations i WHERE i.output_id=l.link_id)
      AND (l.relationship_id IS NULL OR EXISTS(
        SELECT 1 FROM effective_relationships s WHERE s.relationship_id=l.relationship_id
        AND s.entity_id=l.entity_id AND s.route_id=l.route_id));
    CREATE TEMP VIEW effective_road_links AS
      SELECT l.* FROM main.road_links l JOIN effective_entities e USING(entity_id)
      JOIN effective_routes r USING(route_id) WHERE ({inferred} OR l.inferred=0)
      AND NOT EXISTS(SELECT 1 FROM main.invalidations i WHERE i.output_id IN (l.link_id,l.source_route_id));
    """)


def validity_fingerprint(db):
    return digest(
        list(
            db.execute("SELECT output_id,reason FROM invalidations ORDER BY output_id")
        )
    )


def union_intervals(intervals):
    result = []
    for a, b in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if result and a <= result[-1][1] + 1e-7:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def interval_length(intervals):
    return sum(b - a for a, b in union_intervals(intervals))


class MeasuredLine:
    """Exact source vertices with indexed chainage; bounded slices avoid O(n²)."""

    def __init__(self, line):
        self.xy = np.asarray(line.coords)
        self.cumulative = np.r_[
            0.0, np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))
        ]

    def point(self, distance):
        d = float(np.clip(distance, 0, self.cumulative[-1]))
        i = min(
            len(self.xy) - 2,
            max(0, int(np.searchsorted(self.cumulative, d, side="right") - 1)),
        )
        length = self.cumulative[i + 1] - self.cumulative[i]
        return self.xy[i] + (self.xy[i + 1] - self.xy[i]) * (
            (d - self.cumulative[i]) / length if length else 0
        )

    def slice(self, start, end):
        from shapely.geometry import LineString

        a = max(0, float(start))
        b = min(float(end), self.cumulative[-1])
        inside = self.xy[
            np.searchsorted(self.cumulative, a, side="right") : np.searchsorted(
                self.cumulative, b, side="left"
            )
        ]
        return LineString(np.vstack([self.point(a), inside, self.point(b)]))


def directed_intervals(route, source, tolerance=15):
    """Geometric proximity, not road identity; do not use this to copy raster scores.

    Local cell projection avoids projecting all repeated route coordinates onto
    their first occurrence. Heading rejects opposing carriageways/crossings.
    Every clipped part remains bounded by the actual <=50m route cell.
    """
    intervals = []
    buffered = source.buffer(tolerance)
    if not route.intersects(buffered):
        return []
    measured = MeasuredLine(route)
    envelope = buffered.envelope
    west, south, east, north = buffered.bounds
    selected = set()
    for i in range(0, len(measured.xy) - 1, 127):
        end = min(i + 127, len(measured.xy) - 1)
        xy = measured.xy[i : end + 1]
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        if hi[0] < west or hi[1] < south or lo[0] > east or lo[1] > north:
            continue
        selected.update(
            range(
                max(0, math.floor(measured.cumulative[i] / 50)),
                min(
                    math.ceil(route.length / 50),
                    math.ceil(measured.cumulative[end] / 50),
                ),
            )
        )
    for index in sorted(selected):
        start = index * 50.0
        stop = min(start + 50.0, route.length)
        cell = measured.slice(start, stop)
        if cell.is_empty or cell.length <= 0 or not cell.envelope.intersects(envelope):
            continue
        cut = cell.intersection(buffered)
        parts = (
            [cut] if cut.geom_type == "LineString" else list(getattr(cut, "geoms", []))
        )
        for part in parts:
            if part.geom_type != "LineString" or part.length <= 0:
                continue
            point = part.interpolate(0.5, normalized=True)
            chain = source.project(point)
            a = source.interpolate(max(0, chain - 2))
            b = source.interpolate(min(source.length, chain + 2))
            v = np.array([b.x - a.x, b.y - a.y])
            x = np.array(cell.coords[-1]) - np.array(cell.coords[0])
            norm = np.linalg.norm(v) * np.linalg.norm(x)
            if not norm or np.dot(v, x) / norm < math.cos(math.radians(25)):
                continue
            ends = [
                cell.project(Point(part.coords[0])),
                cell.project(Point(part.coords[-1])),
            ]
            intervals.append([float(start + min(ends)), float(start + max(ends))])
    return union_intervals(intervals)


def source_public(record):
    values = [record.get("canonical_family", ""), *record.get("original_urls", [])]
    return any(urlsplit(str(x)).scheme in ["http", "https"] for x in values)


def matching_input(path):
    path = Path(path)
    return {
        "logical_name": "national_osm_matching_inventory",
        "filename": path.name,
        "sha256": checksum(path),
        "bytes": path.stat().st_size,
        "purpose": "Direction-aware OSM attribution and supported tunnel masks; not a visibility certificate",
    }


def insert_sources(db, source):
    for row in source.execute("SELECT source_id,payload FROM sources"):
        r = json.loads(row[1])
        family = r.get("canonical_family") or row[0]
        db.execute(
            "INSERT OR IGNORE INTO sources VALUES (?,?,?,?)",
            (row[0], family, int(source_public(r)), row[1]),
        )
    for row in source.execute(
        "SELECT entity_id,kind,name,search_text,inferred,unresolved,payload FROM current_entities"
    ):
        db.execute("INSERT OR IGNORE INTO entities VALUES (?,?,?,?,?,?,?)", row)
        if row[4] or row[5]:
            db.execute(
                "INSERT OR IGNORE INTO unresolved VALUES (?,?)",
                (
                    "entity:" + row[0],
                    encode(
                        {
                            "entity_id": row[0],
                            "kind": row[1],
                            "name": row[2],
                            "issue": "unresolved_geographic_position"
                            if row[5]
                            else "explicit_inference_not_independent_verification",
                            "original_current_record": json.loads(row[6]),
                        }
                    ),
                ),
            )
    for row in source.execute("SELECT output_id,reason FROM invalidations"):
        db.execute("INSERT OR IGNORE INTO invalidations VALUES (?,?)", row)
    for row in source.execute(
        "SELECT relationship_id,entity_id,route_id,inferred,unresolved,invalidated,payload FROM current_relationships"
    ):
        db.execute("INSERT OR IGNORE INTO relationships VALUES (?,?,?,?,?,?,?)", row)
    for bid, eid, rid, invalidated, payload in source.execute(
        "SELECT binding_id,entity_id,route_id,invalidated,payload FROM current_geometry_bindings"
    ):
        inferred = source.execute(
            "SELECT inferred FROM current_entities WHERE entity_id=?", (eid,)
        ).fetchone()
        db.execute(
            "INSERT OR IGNORE INTO captures VALUES (?,?,?,?,?,?)",
            (bid, eid, rid, inferred[0] if inferred else 1, invalidated, payload),
        )
    families = {
        sid: family
        for sid, family in db.execute("SELECT source_id,family FROM sources")
    }
    canonical_claims = {}
    for cid, eid, payload in source.execute(
        "SELECT claim_id,entity_id,payload FROM claims ORDER BY claim_id"
    ):
        c = json.loads(payload)
        sid = c.get("source_id")
        scope = c.get("geographic_application", "unlocated_context")
        key = digest(
            [
                eid,
                families.get(sid),
                fold(c.get("statement", "")),
                scope,
                c.get("claim_kind"),
            ]
        )
        canonical_id = canonical_claims.setdefault(key, cid)
        usable = bool(
            c.get("active_asset_allowed")
            and c.get("source_supported")
            and sid in families
        )
        db.execute(
            "INSERT OR IGNORE INTO claims VALUES (?,?,?,?,?,?,?)",
            (cid, eid, sid, canonical_id, scope, int(usable), payload),
        )
        if canonical_id != cid:
            db.execute(
                "INSERT OR IGNORE INTO corrections VALUES (?,?)",
                (
                    digest(["duplicate", cid]),
                    encode(
                        {
                            "kind": "duplicate_claim_group_not_independent_corroboration",
                            "claim_id": cid,
                            "canonical_claim_id": canonical_id,
                            "original_retained": True,
                        }
                    ),
                ),
            )
        if not sid in families:
            db.execute(
                "INSERT OR IGNORE INTO unresolved VALUES (?,?)",
                (
                    "claim:" + cid,
                    encode(
                        {
                            "kind": "broken_source_reference",
                            "claim_id": cid,
                            "source_id": sid,
                        }
                    ),
                ),
            )
    db.commit()


def route_record(record, role="catalog_capture"):
    g = shape(record["geometry"])
    if g.geom_type != "LineString" or not g.is_valid or g.is_empty or g.length == 0:
        raise ValueError("Invalid captured line")
    if any(not (100 <= x <= 116 and 6 <= y <= 25) for x, y, *_ in g.coords):
        raise ValueError("Coordinate order/range error")
    gid, _canonical_g, rev = canonical(g, 32648)
    line = transform(projection(32648).transform, g)
    case = record.get("case_id") or (
        record.get("origin_id", "") + "->" + record.get("destination_id", "")
    )
    if case == "->":
        case = None
    return (
        record["route_id"],
        role,
        case,
        record.get("direction", "captured_coordinate_order"),
        gid,
        g.wkb,
        line.length,
        int(record.get("inferred", False)),
        encode(
            {
                **record,
                "canonical_reversed": rev,
                "geometry_fingerprint": digest(g.wkb.hex()),
                "raster_geometry_id": gid,
                "analysis_crs": "EPSG:32648",
                "navigation_readiness": "not_established_by_this_research_release",
            }
        ),
    )


def insert_routes(db, source, raster_directory):
    register = read(raster_directory / "route_register.json")
    for r in register["routes"]:
        db.execute(
            "INSERT OR IGNORE INTO routes VALUES (?,?,?,?,?,?,?,?,?)",
            route_record(r, r["register_role"]),
        )
    for rid, payload in source.execute(
        "SELECT route_id,payload FROM geometries WHERE route_id IN (SELECT route_id FROM effective_geometry_bindings)"
    ):
        if db.execute("SELECT 1 FROM routes WHERE route_id=?", (rid,)).fetchone():
            continue
        r = json.loads(payload)
        parents = db.execute(
            "SELECT min(inferred) FROM captures WHERE route_id=? AND invalidated=0",
            (rid,),
        ).fetchone()[0]
        r["inferred"] = bool(parents)
        db.execute(
            "INSERT OR IGNORE INTO routes VALUES (?,?,?,?,?,?,?,?,?)",
            route_record(
                r, "original_audit_reference" if rid == AUDIT_ID else "catalog_capture"
            ),
        )
    db.commit()


def cell_missing_inputs(cell, line):
    if cell.get("supported_tunnel_excluded"):
        return False
    midpoint = (cell["start_m"] + cell["end_m"]) / 2
    if (
        line.interpolate(max(0, midpoint - 2)).distance(
            line.interpolate(min(line.length, midpoint + 2))
        )
        < 1e-9
    ):
        return False  # Terminal tangent degeneracy, not an unavailable download.
    return (
        cell.get("roadside_sample_coverage", 0) < 1
        or cell.get("dsm_ruggedness_rms_m") is None
    )


def cell_retryable(cell, line, store):
    if not cell_missing_inputs(cell, line):
        return False
    missing = [s for s in cell.get("roadside_samples", []) if s.get("class") is None]
    if missing:
        coords = [[s["longitude"], s["latitude"]] for s in missing]
        values, _, _, _ = store.sample("worldcover", coords, acquire=False)
        if (
            np.isfinite(values).any()
            or not store.native_coverage("worldcover", coords).all()
        ):
            return True
    if cell.get("dsm_ruggedness_rms_m") is None:
        center = [[cell["midpoint_longitude"], cell["midpoint_latitude"]]]
        if np.isfinite(store.terrain(center, acquire=False)[1][0]):
            return True
        values, xy, res, _ = store.sample("dem", center, acquire=False)
        if not np.isfinite(values[0]):
            return not store.native_coverage("dem", center).all()
        neighbors = xy[0] + res[0] * np.array(
            [(dx, dy) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]
        )
        return not store.native_coverage("dem", neighbors).all()
    return False  # Covered native NoData / conflicting pixels stay unknown.


def merge_retry_cell(old, new):
    for key in [
        "geometry_id",
        "cell_index",
        "start_m",
        "end_m",
        "extraction_version",
        "midpoint_longitude",
        "midpoint_latitude",
        "supported_tunnel_excluded",
    ]:
        if old.get(key) != new.get(key):
            raise ValueError(
                "Retry changed geometry, sampling version/location or exclusion mask"
            )
    merged = dict(old)
    samples = []
    for a, b in zip(old["roadside_samples"], new["roadside_samples"], strict=True):
        if (a["longitude"], a["latitude"], a["offset_left_m"]) != (
            b["longitude"],
            b["latitude"],
            b["offset_left_m"],
        ):
            raise ValueError("Retry changed roadside sample support")
        if (
            a["class"] is not None
            and b["class"] is not None
            and a["class"] != b["class"]
        ):
            raise ValueError("Conflicting native evidence; prior checkpoint preserved")
        samples.append(b if a["class"] is None and b["class"] is not None else a)
    valid = [s for s in samples if s["class"] is not None]
    merged.update(
        roadside_samples=samples,
        roadside_valid_samples=len(valid),
        roadside_sample_coverage=len(valid) / 6,
    )
    fractions = {}
    for name, code in [("tree", 10), ("water", 80), ("cropland", 40)]:
        fractions[name] = (
            sum(s["class"] == code for s in valid) / len(valid) if valid else None
        )
        merged["roadside_" + name + "_fraction"] = fractions[name]
    merged["raw_sampled_fractions"] = fractions
    if (
        old.get("dsm_ruggedness_rms_m") is None
        and new.get("dsm_ruggedness_rms_m") is not None
    ):
        for key in [
            "dsm_elevation_m",
            "dsm_ruggedness_rms_m",
            "raw_dsm_ruggedness_rms_m",
            "dsm_3x3_valid_samples",
            "dsm_center_ref",
            "dsm_neighbor_refs",
        ]:
            merged[key] = new[key]
    return merged


def cells_score(cells, policy):
    score = scenic_score(cells, policy)
    score["supported_tunnel_excluded_m"] = sum(
        c["cell_length_m"] for c in cells if c["supported_tunnel_excluded"]
    )
    score["uncertain_surface_context_m"] = sum(
        c["cell_length_m"]
        for c in cells
        if c["surface_context_status"]
        == "tunnel_tagged_candidate_unresolved_attribution"
    )
    score["component_contributions"] = contributions(score, policy)
    return score


def record_raster_assets(db, store):
    # Ownership governs cache eviction, not evidence retention. Register inherited
    # and newly acquired pixel receipts in the same transaction as their score.
    registered = {row[0] for row in db.execute("SELECT asset_id FROM raster_assets")}
    for asset in store.assets:
        if asset["key"] in registered:
            continue
        db.execute(
            "INSERT OR REPLACE INTO raster_assets VALUES (?,?)",
            (
                asset["key"],
                encode(
                    {**asset["receipt"], "native_transform": list(asset["transform"])}
                ),
            ),
        )
        registered.add(asset["key"])


def retry_geometry(db, gid, line, geo, rev, store, matcher, policy, existing):
    rows = {
        r[0]: json.loads(r[1])
        for r in db.execute(
            "SELECT cell_index,payload FROM cells WHERE geometry_id=? ORDER BY cell_index",
            (gid,),
        )
    }
    selected = [i for i, c in rows.items() if cell_retryable(c, line, store)]
    if not selected:
        return "terminal_or_complete"
    before = {i: rows[i] for i in selected}
    failure_start = len(store.failures)
    try:
        for _, group in itertools.groupby(
            enumerate(selected), lambda pair: pair[1] - pair[0]
        ):
            indices = [i for _, i in group]
            for start in range(indices[0], indices[-1] + 1, 1000):
                cells = compute_cells(
                    gid,
                    line,
                    32648,
                    store,
                    start,
                    max_cells=min(1000, indices[-1] + 1 - start),
                    matcher=matcher,
                    travel_reversed=rev,
                )
                for cell in cells:
                    rows[cell["cell_index"]] = merge_retry_cell(
                        rows[cell["cell_index"]], cell
                    )
        improved = any(encode(rows[i]) != encode(before[i]) for i in selected)
        score = cells_score(list(rows.values()), policy)
        db.execute("SAVEPOINT retry_geometry")
        if improved:
            record_raster_assets(db, store)
            db.executemany(
                "UPDATE cells SET payload=? WHERE geometry_id=? AND cell_index=?",
                [(encode(rows[i]), gid, i) for i in selected],
            )
            provenance = json.loads(existing[2])
            provenance.update(
                method="integrated_derivative_native_retry_same_policy",
                retry_recomputed_cells=len(selected),
                prior_method=provenance.get("method"),
            )
            db.execute(
                "UPDATE scores SET payload=?,provenance=? WHERE geometry_id=?",
                (encode(score), encode(provenance), gid),
            )
            # A crash after this commit must not expose pre-retry comparisons.
            db.execute(
                "DELETE FROM comparisons WHERE case_id IN (SELECT case_id FROM routes WHERE geometry_id=?)",
                (gid,),
            )
        db.execute(
            "INSERT INTO score_retry_history(geometry_id,successful,payload) VALUES (?,?,?)",
            (
                gid,
                int(improved),
                encode(
                    {
                        "attempted_at": now(),
                        "previous_score": json.loads(existing[1]),
                        "previous_cells": list(before.values()),
                        "requested_cells": selected,
                        "failures": store.failures[failure_start:],
                    }
                ),
            ),
        )
        db.execute("RELEASE retry_geometry")
        db.commit()
        return "improved" if improved else "unchanged"
    except Exception as error:
        db.rollback()
        db.execute(
            "INSERT INTO score_retry_history(geometry_id,successful,payload) VALUES (?,?,?)",
            (
                gid,
                0,
                encode(
                    {
                        "attempted_at": now(),
                        "reason": str(error),
                        "requested_cells": selected,
                        "previous_checkpoint_preserved": True,
                    }
                ),
            ),
        )
        db.commit()
        return "failed_checkpoint_preserved"


def score_routes(
    db, raster_directory, root, online, *, osm_path=None, retry_missing=False
):
    raster = readonly(raster_directory / "scenic.sqlite")
    policy = read(raster_directory / "config.json")["scoring"]
    roots = [raster_directory / "raster_cache"]
    # Only validated source crops; never scan or consume imagery datasets.
    for folder in (root / "exports/route_intelligence").glob("phase4*/cache/raster"):
        roots += [folder / k for k in ["worldcover", "dem"] if (folder / k).is_dir()]
    store = None
    new = 0
    reused = 0
    failures = []
    retry_results = defaultdict(int)
    for gid, wkb, role in db.execute(
        "SELECT geometry_id,min(geometry_wkb),min(role) FROM routes WHERE role!='original_audit_reference' GROUP BY geometry_id"
    ).fetchall():
        existing = db.execute(
            "SELECT version,payload,provenance FROM scores WHERE geometry_id=?", (gid,)
        ).fetchone()
        if existing and existing[0] != EXTRACTION_VERSION:
            raise ValueError(
                "Extraction version changed; create a new derivative output"
            )
        if existing and not retry_missing:
            continue
        original = raster.execute(
            "SELECT geometry,crs,length_m,score_json FROM jobs WHERE geometry_id=?",
            (gid,),
        ).fetchone()
        forward_geometry = from_wkb(wkb)
        _, geo, rev = canonical(forward_geometry, 32648)
        line = transform(projection(32648).transform, geo)
        if existing:
            pending = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT payload FROM cells WHERE geometry_id=?", (gid,)
                )
            ]
            if not any(cell_missing_inputs(c, line) for c in pending):
                retry_results["complete_or_intentionally_excluded"] += 1
                continue
            if store is None:
                store = RasterStore(
                    Path(db.execute("PRAGMA database_list").fetchone()[2]).parent
                    / "raster_cache",
                    roots,
                    online=online,
                )
            matcher = (
                route_matcher(root, geo, 32648, source_path=osm_path)
                if osm_path and osm_path.exists()
                else None
            )
            if matcher is None:
                raise ValueError(
                    "Retry requires the declared OSM snapshot for tunnel masking"
                )
            retry_results[
                retry_geometry(
                    db, gid, line, geo, rev, store, matcher, policy, existing
                )
            ] += 1
            continue
        if (
            original
            and original[3]
            and original[1] == 32648
            and from_wkb(original[0]).equals_exact(geo, 0)
        ):
            for index, start, end, length, payload in raster.execute(
                "SELECT cell_index,start_m,end_m,length_m,payload FROM cells WHERE geometry_id=? ORDER BY cell_index",
                (gid,),
            ):
                db.execute(
                    "INSERT OR REPLACE INTO cells VALUES (?,?,?,?,?,?)",
                    (gid, index, start, end, length, payload),
                )
            score = json.loads(original[3])
            method = "identical_registered_geometry_same_crs_and_version"
            if (
                abs(
                    sum(
                        r[0]
                        for r in db.execute(
                            "SELECT length_m FROM cells WHERE geometry_id=?", (gid,)
                        )
                    )
                    - line.length
                )
                > 1e-6
            ):
                raise ValueError("Exact raster reuse length mismatch")
            reused += 1
        else:
            if store is None:
                store = RasterStore(
                    Path(db.execute("PRAGMA database_list").fetchone()[2]).parent
                    / "raster_cache",
                    roots,
                    online=online,
                )
            matcher = (
                route_matcher(root, geo, 32648, source_path=osm_path)
                if osm_path and osm_path.exists()
                else None
            )
            if matcher is None:
                raise ValueError(
                    "New geometry extraction requires the declared OSM snapshot for supported tunnel masking"
                )
            # Re-extract exact route cells using cached native pixels. A nearby
            # parallel source section or partial section score is never copied.
            for start in range(0, math.ceil(line.length / 50), 1000):
                cells = compute_cells(
                    gid,
                    line,
                    32648,
                    store,
                    start,
                    max_cells=1000,
                    matcher=matcher,
                    travel_reversed=rev,
                )
                db.executemany(
                    "INSERT OR REPLACE INTO cells VALUES (?,?,?,?,?,?)",
                    [
                        (
                            gid,
                            c["cell_index"],
                            c["start_m"],
                            c["end_m"],
                            c["cell_length_m"],
                            encode(c),
                        )
                        for c in cells
                    ],
                )
                db.commit()
            cells = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT payload FROM cells WHERE geometry_id=? ORDER BY cell_index",
                    (gid,),
                )
            ]
            score = scenic_score(cells, policy)
            score["supported_tunnel_excluded_m"] = sum(
                c["cell_length_m"] for c in cells if c["supported_tunnel_excluded"]
            )
            score["uncertain_surface_context_m"] = sum(
                c["cell_length_m"]
                for c in cells
                if c["surface_context_status"]
                == "tunnel_tagged_candidate_unresolved_attribution"
            )
            score["component_contributions"] = contributions(score, policy)
            method = "integrated_derivative_native_extraction_same_policy"
            new += 1
        score.setdefault("component_contributions", contributions(score, policy))
        coverage = {
            "method": method,
            "extraction_version": EXTRACTION_VERSION,
            "geometry_id": gid,
            "covered_route_m": line.length,
            "missing_geometry_m": 0.0,
            "unknown_inputs_remain_in_score_bounds": True,
            "sampling_direction": "canonical; routes carry canonical_reversed for travel chainage",
            "source_dataset": "raster_scenic_nationwide"
            if original and method.startswith("identical")
            else VERSION,
        }
        if store:
            record_raster_assets(db, store)
        db.execute(
            "INSERT OR REPLACE INTO scores VALUES (?,?,?,?)",
            (gid, EXTRACTION_VERSION, encode(score), encode(coverage)),
        )
        db.commit()
        print(
            encode(
                {"scored_geometry": gid, "reused_exact": reused, "new_extraction": new}
            ),
            flush=True,
        )
    # Pixel references on copied baseline cells resolve without changing the source.
    for aid, payload in raster.execute("SELECT cache_key,payload FROM raster_assets"):
        db.execute("INSERT OR IGNORE INTO raster_assets VALUES (?,?)", (aid, payload))
    if store:
        failures = store.failures
        record_raster_assets(db, store)
        metrics = {
            "acquisition_s": store.acquisition_s,
            "acquired_cache_bytes": store.acquired_bytes,
            "network": store.network.snapshot(),
            "failures": failures,
        }
        store.close()
    else:
        metrics = {"acquisition_s": 0, "acquired_cache_bytes": 0, "failures": []}
    raster.close()
    db.commit()
    return {
        "exact_geometry_reuses_this_run": reused,
        "new_native_extractions_this_run": new,
        "retry_results_this_run": dict(retry_results),
        **metrics,
    }


def contributions(score, policy):
    coverage = score["evidence_weight_coverage"]
    return {
        k: {
            "conditional_points": 100
            * policy["scenic_weights"][k]
            * score["component_coverage"][k]
            * score["component_means"][k]
            / coverage
            if coverage and score["component_means"][k] is not None
            else None,
            "component_coverage": score["component_coverage"][k],
        }
        for k in policy["scenic_weights"]
    }


def attach_road_context(db, source):
    rows = []
    geometries = []
    # OSM bounded source components provide narrower context than whole Goong
    # anchor corridors. Broad/inferred captures remain discoverable by ID only.
    for cid, eid, payload in source.execute(
        "SELECT component_id,entity_id,payload FROM current_source_components"
    ):
        c = json.loads(payload)
        if c.get("geometry"):
            g = transform(projection(32648).transform, shape(c["geometry"]))
            parts = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
            for i, p in enumerate(parts):
                if p.geom_type == "LineString" and p.length:
                    rows.append((eid, cid + ":" + str(i), c))
                    geometries.append(p)
    tree = STRtree(geometries)
    for rid, wkb, inferred in db.execute(
        "SELECT route_id,geometry_wkb,inferred FROM routes WHERE role!='original_audit_reference'"
    ).fetchall():
        route = transform(projection(32648).transform, from_wkb(wkb))
        for ix in tree.query(route.buffer(15).envelope):
            eid, cid, component = rows[int(ix)]
            for direction, g in [
                ("same_source_order", geometries[int(ix)]),
                ("reverse_source_order", reverse(geometries[int(ix)])),
            ]:
                intervals = directed_intervals(route, g)
                if not intervals:
                    continue
                lid = digest([rid, eid, cid, direction])
                payload = {
                    "direction": direction,
                    "source_component_id": cid,
                    "source_geometry": component,
                    "intervals_m": intervals,
                    "scope": "bounded_source_component_near_route_not_full_road_certification",
                    "uncertainty": "15m spatial tolerance and 25 degree tangent concordance are proximity evidence, not unique carriageway identity, road visibility or current access",
                    "raster_score_effect": "none",
                }
                db.execute(
                    "INSERT OR REPLACE INTO road_links VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        lid,
                        rid,
                        eid,
                        cid,
                        int(inferred),
                        "directed_bounded_source_component_proximity",
                        interval_length(intervals),
                        encode(intervals),
                        encode(payload),
                    ),
                )
        # Preserve source-explicit parent linkage for each own captured corridor.
        for bid, eid, inf, pay in db.execute(
            "SELECT binding_id,entity_id,inferred,payload FROM captures WHERE route_id=? AND invalidated=0",
            (rid,),
        ).fetchall():
            p = json.loads(pay)
            scope = "captured_corridor_identity_unverified"
            db.execute(
                "INSERT OR REPLACE INTO road_links VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    digest([rid, bid]),
                    rid,
                    eid,
                    rid,
                    int(inferred or inf),
                    "source_explicit_capture_parent",
                    0.0,
                    "[]",
                    encode(
                        {
                            "original_binding": p,
                            "scope": scope,
                            "extent_certified": False,
                            "not_a_whole_road_claim": True,
                        }
                    ),
                ),
            )
        db.commit()


def attach_places(db, radius_m=1000):
    db.execute("DELETE FROM place_links")
    places = []
    points = []
    for eid, kind, inf, unres, payload in db.execute(
        "SELECT entity_id,kind,inferred,unresolved,payload FROM entities WHERE kind NOT IN ('road','current_candidate','original_audit_reference')"
    ):
        p = json.loads(payload)
        pos = p.get("position")
        if pos and not unres and "area" not in (p.get("coordinate_role") or ""):
            places.append((eid, inf, p))
            points.append(transform(projection(32648).transform, Point(pos)))
    tree = STRtree(points)
    for rid, wkb, inf in db.execute(
        "SELECT route_id,geometry_wkb,inferred FROM routes WHERE role!='original_audit_reference'"
    ).fetchall():
        route = transform(projection(32648).transform, from_wkb(wkb))
        candidates = {
            places[int(i)][0]: (places[int(i)], points[int(i)])
            for i in tree.query(route.buffer(radius_m))
        }
        # Source-explicit access associations survive the configurable discovery radius.
        inherited = defaultdict(list)
        for relid, eid, payload in db.execute(
            "SELECT relationship_id,entity_id,payload FROM effective_relationships WHERE route_id=?",
            (rid,),
        ):
            inherited[eid].append((relid, json.loads(payload)))
        for eid in inherited:
            if len(inherited[eid]) > 1:
                db.execute(
                    "INSERT OR IGNORE INTO corrections VALUES (?,?)",
                    (
                        digest(["access_variants", rid, eid]),
                        encode(
                            {
                                "kind": "preserve_all_current_access_variants",
                                "route_id": rid,
                                "entity_id": eid,
                                "source_relationship_ids": [
                                    r[0] for r in inherited[eid]
                                ],
                                "rationale": "Do not arbitrarily collapse provider approaches or compatible/incompatible detour comparisons for the same entity and route.",
                            }
                        ),
                    ),
                )
            if eid not in candidates:
                row = db.execute(
                    "SELECT inferred,payload FROM entities WHERE entity_id=?", (eid,)
                ).fetchone()
                if row:
                    p = json.loads(row[1])
                    pos = p.get("position")
                    if pos and "area" not in (p.get("coordinate_role") or ""):
                        candidates[eid] = (
                            (eid, row[0], p),
                            transform(projection(32648).transform, Point(pos)),
                        )
        for eid, ((_, pinf, p), point) in candidates.items():
            distance = point.distance(route)
            chain = route.project(point)
            if distance > radius_m and eid not in inherited:
                continue
            approaches = inherited.get(eid, [(None, None)])
            relationship_id, access = approaches[0]
            payload = {
                "entity_id": eid,
                "name": p.get("name"),
                "coordinate_role": p.get("coordinate_role"),
                "position": p.get("position"),
                "geometry_distance_m": distance,
                "chainage_m": chain,
                "chainage_role": "travel_coordinate_order",
                "route_length_m": route.length,
                "association_method": "source_relationship_plus_reprojection"
                if access
                else "point_to_captured_route_proximity_discovery_only",
                "opportunity_class": "endpoint_area"
                if min(chain, route.length - chain) <= radius_m
                else "route_interior_nearby",
                "discovery_radius_m": radius_m,
                "provider_access": access,
                "provider_routed_approach": bool(
                    access
                    and access.get("permitted_uses", {}).get("provider_routed_approach")
                ),
                "compatible_detour": bool(
                    access and access.get("permitted_uses", {}).get("compatible_detour")
                ),
                "entrance_verified": False,
                "operational_stop_eligibility": False,
                "visibility": "not_established",
                "limitations": "Geographic position is separate from entrance, provider pin approach, parking and public access. A nearby place is not a feasible or road-visible stop.",
            }
            for relationship_id, access in approaches:
                variant = {
                    **payload,
                    "provider_access": access,
                    "provider_routed_approach": bool(
                        access
                        and access.get("permitted_uses", {}).get(
                            "provider_routed_approach"
                        )
                    ),
                    "compatible_detour": bool(
                        access
                        and access.get("permitted_uses", {}).get("compatible_detour")
                    ),
                }
                db.execute(
                    "INSERT OR REPLACE INTO place_links VALUES (?,?,?,?,?,?,?)",
                    (
                        digest([rid, eid, relationship_id]),
                        rid,
                        eid,
                        int(inf or pinf),
                        0,
                        relationship_id,
                        encode(variant),
                    ),
                )
        db.commit()


def attach_claims(db):
    db.execute("DELETE FROM claim_links")
    # Duplicate claims stay in storage but appear once per canonical source family.
    for cid, eid, scope, usable, payload in db.execute(
        "SELECT claim_id,entity_id,scope,usable,payload FROM claims WHERE claim_id=canonical_id"
    ).fetchall():
        c = json.loads(payload)
        usable = bool(
            usable
            and db.execute(
                "SELECT 1 FROM effective_entities WHERE entity_id=?", (eid,)
            ).fetchone()
        )
        if scope == "entity_identity_or_place_context_not_road_claim":
            links = [
                (
                    r,
                    "nearby_place_context",
                    {
                        "place_link_id": lid,
                        "chainage_point_m": json.loads(p)["chainage_m"],
                        "historical_markers_are_not_modern_geocoding": True,
                    },
                )
                for lid, r, p in db.execute(
                    "SELECT link_id,route_id,payload FROM place_links WHERE entity_id=?",
                    (eid,),
                )
            ]
        elif scope in [
            "general_corridor_context",
            "source_named_section_context_supported_osm_extent",
        ]:
            links = []
            for rid in {
                r[0]
                for r in db.execute(
                    "SELECT route_id FROM road_links WHERE entity_id=?", (eid,)
                )
            }:
                intervals = []
                for (v,) in db.execute(
                    "SELECT intervals FROM road_links WHERE route_id=? AND entity_id=?",
                    (rid, eid),
                ):
                    intervals += json.loads(v)
                if scope == "source_named_section_context_supported_osm_extent":
                    extent = c.get("source_geometry_scope", {}).get("geometry")
                    if not extent:
                        continue
                    route = transform(
                        projection(32648).transform,
                        shape(read_route(db, rid)["geometry"]),
                    )
                    g = transform(projection(32648).transform, shape(extent))
                    parts = list(g.geoms) if hasattr(g, "geoms") else [g]
                    intervals = []
                    for part in parts:
                        if part.geom_type == "LineString":
                            intervals += directed_intervals(
                                route, part
                            ) + directed_intervals(route, reverse(part))
                    if not intervals:
                        continue
                links.append(
                    (
                        rid,
                        "general_corridor_context"
                        if scope == "general_corridor_context"
                        else "historical_bounded_section_context",
                        {
                            "intervals_m": union_intervals(intervals),
                            "source_named_component_only": True,
                            "scope_warning": "Source-specific historical context; no current restriction or whole-road attribution. Empty intervals mean parent-capture context only, not a located claim.",
                        },
                    )
                )
        else:
            db.execute(
                "INSERT OR REPLACE INTO unresolved VALUES (?,?)",
                (
                    "claim:" + cid,
                    encode(
                        {
                            "claim_id": cid,
                            "entity_id": eid,
                            "issue": "localized_claim_extent_pending; no route interval asserted",
                            "source_scope": scope,
                        }
                    ),
                ),
            )
            continue
        for rid, application, detail in links:
            db.execute(
                "INSERT OR REPLACE INTO claim_links VALUES (?,?,?,?,?,?)",
                (
                    digest([rid, cid]),
                    rid,
                    cid,
                    application,
                    int(usable),
                    encode(
                        {
                            "original_claim_scope": c.get("scope"),
                            "application": application,
                            "detail": detail,
                            "source_id": c.get("source_id"),
                            "current_operational": False,
                            "score_effect": "none",
                        }
                    ),
                ),
            )
    db.commit()


class IntegratedBaseline:
    def __init__(self, directory, *, enabled=False):
        self.directory, self.enabled = Path(directory), enabled

    def query(
        self,
        operation="inventory",
        *,
        route_id=None,
        case_id=None,
        text="",
        include_inferred=False,
        include_unresolved=False,
        entity_id=None,
    ):
        if not self.enabled:
            return {"status": "disabled", "records": []}
        with readonly(self.directory / "integrated.sqlite") as db:
            current_views(db, include_inferred, include_unresolved)
            if operation == "inventory":
                return {
                    "status": "ok",
                    "version": VERSION,
                    "counts_scope": "all_retained_records_including_history_not_current_eligibility",
                    "counts": {
                        t: db.execute("SELECT count(*) FROM " + t).fetchone()[0]
                        for t in [
                            "entities",
                            "claims",
                            "sources",
                            "routes",
                            "scores",
                            "place_links",
                            "unresolved",
                        ]
                    },
                }
            if operation == "entity":
                row = db.execute(
                    "SELECT payload,inferred,unresolved FROM effective_entities WHERE entity_id=?",
                    (entity_id,),
                ).fetchone()
                if (
                    not row
                    or (row[1] and not include_inferred)
                    or (row[2] and not include_unresolved)
                ):
                    return {"status": "not_found", "records": []}
                return {
                    "status": "ok",
                    "entity": json.loads(row[0]),
                    "claims": [
                        {"claim": json.loads(c), "source": json.loads(s)}
                        for c, s in db.execute(
                            "SELECT c.payload,s.payload FROM effective_claims c JOIN sources s USING(source_id) WHERE entity_id=?",
                            (entity_id,),
                        )
                    ],
                    "current_relationships": [
                        json.loads(r[0])
                        for r in db.execute(
                            "SELECT payload FROM effective_relationships WHERE (entity_id=? OR route_id=?) AND route_id!=?",
                            (entity_id, entity_id, AUDIT_ID),
                        )
                    ],
                }
            if operation == "search":
                table = "effective_entities"
                clauses = ["search_text LIKE ?"]
                params = [
                    "%" + fold(text).replace("%", "\\%").replace("_", "\\_") + "%"
                ]
                if not include_inferred:
                    clauses.append("inferred=0")
                if not include_unresolved:
                    clauses.append("unresolved=0")
                return {
                    "status": "ok",
                    "records": [
                        json.loads(r[0])
                        for r in db.execute(
                            "SELECT payload FROM "
                            + table
                            + " WHERE "
                            + " AND ".join(clauses)
                            + " ORDER BY entity_id LIMIT 100",
                            params,
                        )
                    ],
                }
            if operation == "compare":
                r = db.execute(
                    "SELECT payload FROM comparisons WHERE case_id=?", (case_id,)
                ).fetchone()
                if r:
                    comparison = json.loads(r[0])
                    snapshot = db.execute(
                        "SELECT value FROM metadata WHERE key='comparison_validity_fingerprint'"
                    ).fetchone()
                    stale = bool(snapshot and snapshot[0] != validity_fingerprint(db))
                    for candidate in comparison.get("routes", []):
                        if not db.execute(
                            "SELECT 1 FROM effective_routes WHERE route_id=?",
                            (candidate["route_id"],),
                        ).fetchone():
                            stale = True
                        iconic = candidate.get("iconic", {})
                        cached_ids = set(
                            iconic.get("documented_road_entities", [])
                            + iconic.get("nearby_only_entities", [])
                        )
                        live_ids = {
                            row[0]
                            for row in db.execute(
                                "SELECT c.entity_id FROM effective_claim_links l JOIN claims c USING(claim_id) WHERE l.route_id=? AND json_extract(c.payload,'$.claim_kind') IN ('national_historic_site_designation','documented_1826_construction')",
                                (candidate["route_id"],),
                            )
                        }
                        if not cached_ids <= live_ids:
                            stale = True
                    for pair in comparison.get("comparisons", []):
                        for key, rid in [
                            ("candidate_specific_places_a", pair["route_a"]),
                            ("candidate_specific_places_b", pair["route_b"]),
                            ("shared_places", pair["route_a"]),
                            ("shared_places", pair["route_b"]),
                        ]:
                            current = {
                                p[0]
                                for p in db.execute(
                                    "SELECT entity_id FROM effective_place_links WHERE route_id=?",
                                    (rid,),
                                )
                            }
                            if not set(pair.get(key, [])) <= current:
                                stale = True
                    if stale:
                        return {
                            "status": "stale_comparison",
                            "comparison": None,
                            "reason": "Current validity/status filters changed comparison dependencies; rebuild a derivative before using cached rankings or place opportunities",
                        }
                return {
                    "status": "ok" if r else "not_found",
                    "comparison": json.loads(r[0]) if r else None,
                }
            if operation == "gaps":
                return {
                    "status": "ok",
                    "scope": "unresolved_history_queue_not_current_usable_records",
                    "records": [
                        json.loads(r[0])
                        for r in db.execute(
                            "SELECT payload FROM unresolved ORDER BY item_id LIMIT 100"
                        )
                    ],
                }
            if operation not in ["route", "places", "evidence", "coverage"]:
                raise ValueError("Unsupported opt-in operation")
            table = "effective_routes"
            row = db.execute(
                "SELECT payload,geometry_id FROM "
                + table
                + " WHERE route_id=? AND role!='original_audit_reference'",
                (route_id,),
            ).fetchone()
            if not row:
                return {"status": "not_found", "records": []}
            score = db.execute(
                "SELECT payload,provenance FROM scores WHERE geometry_id=?", (row[1],)
            ).fetchone()
            result = {
                "status": "ok",
                "route": json.loads(row[0]),
                "scenic": json.loads(score[0]) if score else None,
                "raster_provenance": json.loads(score[1]) if score else None,
            }
            if operation in ["route", "places"]:
                pt = "effective_place_links"
                result["places"] = [
                    json.loads(r[0])
                    for r in db.execute(
                        "SELECT payload FROM "
                        + pt
                        + " WHERE route_id=? ORDER BY entity_id",
                        (route_id,),
                    )
                ]
            if operation in ["route", "evidence"]:
                result["claims"] = [
                    {
                        "claim": json.loads(c),
                        "binding": json.loads(l),
                        "source": json.loads(s),
                    }
                    for c, l, s in db.execute(
                        "SELECT c.payload,l.payload,s.payload FROM effective_claim_links l JOIN claims c USING(claim_id) JOIN sources s USING(source_id) WHERE route_id=? ORDER BY claim_id",
                        (route_id,),
                    )
                ]
                result["road_context"] = [
                    json.loads(r[0])
                    for r in db.execute(
                        "SELECT payload FROM effective_road_links WHERE route_id=? AND (? OR inferred=0) ORDER BY link_id",
                        (route_id, int(include_inferred)),
                    )
                ]
                if include_unresolved:
                    result["unresolved_context"] = [
                        {
                            "claim": json.loads(p),
                            "application": "unlocated_source_description_not_a_route_section_fact",
                        }
                        for (p,) in db.execute(
                            "SELECT c.payload FROM effective_claims c WHERE c.scope='localized_claim_extent_pending' AND c.entity_id IN (SELECT entity_id FROM effective_road_links WHERE route_id=?)",
                            (route_id,),
                        )
                    ]
            return result


def compare_routes(db, policy):
    cases = [
        r[0]
        for r in db.execute(
            "SELECT DISTINCT case_id FROM effective_routes WHERE case_id IS NOT NULL AND case_id!='' AND role IN ('priority','geographic_validation_reference')"
        )
    ]
    for case in cases:
        routes = []
        for rid, payload, length, scenic in db.execute(
            "SELECT r.route_id,r.payload,r.length_m,s.payload FROM effective_routes r JOIN scores s USING(geometry_id) WHERE r.case_id=?",
            (case,),
        ):
            r = json.loads(payload)
            s = json.loads(scenic)
            route = {
                "route_id": rid,
                "intent": r.get("intent", r.get("request_id")),
                "provider_distance_m": r["provider_distance_m"],
                "provider_duration_s": r["provider_duration_s"],
                "provider_snapshot": r.get("source_receipt", r.get("receipt", {})),
                "route_ranking_eligible": False,
                "provider_comparison_eligible": True,
                "scenic": s,
                "iconic": iconic_score(0, length, [], [], policy),
                "interpretation": "Scenic, documented significance and captured provider costs are separate; no profile winner or navigation recommendation asserted.",
            }
            # Documented monument significance stays nearby, never a pass-wide index.
            ids = []
            for eid, kind in db.execute(
                "SELECT c.entity_id,json_extract(c.payload,'$.claim_kind') FROM effective_claim_links l JOIN claims c USING(claim_id) WHERE l.route_id=?",
                (rid,),
            ):
                if kind in [
                    "national_historic_site_designation",
                    "documented_1826_construction",
                ]:
                    ids.append(eid)
            route["iconic"] = iconic_score(0, length, [], ids, policy)
            routes.append(route)
        express = express_scores(routes, policy) if routes else {}
        for r in routes:
            r["express"] = express[r["route_id"]]
        comparisons = []
        for a, b in itertools.combinations(routes, 2):
            ga = shape(read_route(db, a["route_id"])["geometry"])
            gb = shape(read_route(db, b["route_id"])["geometry"])
            # Physical shared length is exact line topology, not a proximity
            # buffer and not repeated OSM IDs. Partial cells keep length support.
            pa = transform(projection(32648).transform, ga)
            pb = transform(projection(32648).transform, gb)
            shared = pa.intersection(pb)
            shared_m = shared.length
            place_sets = []
            for r in [a, b]:
                place_sets.append(
                    {
                        x[0]
                        for x in db.execute(
                            "SELECT entity_id FROM effective_place_links WHERE route_id=?",
                            (r["route_id"],),
                        )
                    }
                )
            sections = []
            for r, line in [(a, pa), (b, pb)]:
                sections.append(
                    split_landscape(db, r["route_id"], line, shared, policy)
                )
            comparisons.append(
                {
                    "route_a": a["route_id"],
                    "route_b": b["route_id"],
                    "distance_delta_m": a["provider_distance_m"]
                    - b["provider_distance_m"],
                    "provider_duration_delta_s": a["provider_duration_s"]
                    - b["provider_duration_s"],
                    "shared_exact_geometry_m": shared_m,
                    "landscape_shared_and_differentiating": sections,
                    "shared_places": sorted(place_sets[0] & place_sets[1]),
                    "candidate_specific_places_a": sorted(
                        place_sets[0] - place_sets[1]
                    ),
                    "candidate_specific_places_b": sorted(
                        place_sets[1] - place_sets[0]
                    ),
                    "comparison_eligibility": "research_proxy_comparison_only_not_navigation_readiness",
                    "winner": None,
                    "limitations": "Exact topology is a conservative shared-length lower bound; near-parallel carriageways are not merged. Partial-cell support uses the existing cell observation, not a new visibility measurement. Captured ETAs may have different snapshot dates.",
                }
            )
        db.execute(
            "INSERT OR REPLACE INTO comparisons VALUES (?,?)",
            (
                case,
                encode(
                    {
                        "case_id": case,
                        "routes": routes,
                        "comparisons": comparisons,
                        "recommendation": "no_forced_profile_winner",
                        "iconic_research_completeness": "unknown; identity-only claims are not significance evidence",
                    }
                ),
            ),
        )
    db.execute(
        "INSERT OR REPLACE INTO metadata VALUES (?,?)",
        ("comparison_validity_fingerprint", validity_fingerprint(db)),
    )
    db.commit()


def read_route(db, rid):
    return json.loads(
        db.execute("SELECT payload FROM routes WHERE route_id=?", (rid,)).fetchone()[0]
    )


def split_landscape(db, rid, line, shared, policy):
    row = db.execute(
        "SELECT geometry_id,payload FROM routes WHERE route_id=?", (rid,)
    ).fetchone()
    reverse_grid = read_route(db, rid)["canonical_reversed"]
    groups = {"shared": [], "differentiating": []}
    measured = MeasuredLine(line)
    # Index exact shared source vertices in small, adjacent chunks. Repeatedly
    # intersecting every cell with the entire multi-road geometry is quadratic.
    # Chunks share endpoints only; local union protects against any overlap.
    parts = list(shared.geoms) if hasattr(shared, "geoms") else [shared]
    chunks = []
    for part in parts:
        if part.geom_type != "LineString" or part.is_empty:
            continue
        coords = np.asarray(part.coords)
        for i in range(0, len(coords) - 1, 127):
            chunks.append(LineString(coords[i : i + 128]))
    shared_index = STRtree(chunks)
    for (payload,) in db.execute(
        "SELECT payload FROM cells WHERE geometry_id=? ORDER BY cell_index", (row[0],)
    ):
        c = json.loads(payload)
        start, end = (
            (line.length - c["end_m"], line.length - c["start_m"])
            if reverse_grid
            else (c["start_m"], c["end_m"])
        )
        cell_geometry = measured.slice(start, end)
        nearby = shared_index.query(cell_geometry.envelope)
        local = unary_union([chunks[int(i)] for i in nearby]) if len(nearby) else None
        n = cell_geometry.intersection(local).length if local is not None else 0.0
        n = min(c["cell_length_m"], max(0, n))
        for key, length in [("shared", n), ("differentiating", c["cell_length_m"] - n)]:
            if length > 1e-8:
                groups[key].append({**c, "cell_length_m": length})
    return {
        "route_id": rid,
        **{
            k: {
                "length_m": sum(c["cell_length_m"] for c in cells),
                "scenic": scenic_score(cells, policy) if cells else None,
                "method": "cell observation support apportioned by exact geometric intersection length; cells never counted twice",
            }
            for k, cells in groups.items()
        },
    }


def build(root, out, phase, raster, *, online=False, osm=None, retry_missing=False):
    root, out, phase, raster = (Path(p).resolve() for p in [root, out, phase, raster])
    if (
        out == phase
        or out == raster
        or out.is_relative_to(phase)
        or out.is_relative_to(raster)
    ):
        raise ValueError(
            "Integration output must be separate from preserved input releases"
        )
    required = [
        phase / "national_mapping.sqlite",
        raster / "scenic.sqlite",
        raster / "route_register.json",
        raster / "config.json",
    ]
    hashes = {
        str(p.relative_to(root)) if p.is_relative_to(root) else p.name: {
            "sha256": checksum(p),
            "bytes": p.stat().st_size,
        }
        for p in required
    }
    source = readonly(required[0])
    db = database(out)
    matching = matching_input(osm) if osm else None
    previous_matching = db.execute(
        "SELECT value FROM metadata WHERE key='matching_input'"
    ).fetchone()
    if previous_matching and json.loads(previous_matching[0]) != matching:
        raise ValueError("OSM matching input changed; create a new output version")
    db.execute(
        "INSERT OR IGNORE INTO metadata VALUES (?,?)",
        ("matching_input", encode(matching)),
    )
    db.commit()
    prior = db.execute(
        "SELECT value FROM metadata WHERE key='input_manifest'"
    ).fetchone()
    if prior and json.loads(prior[0]) != hashes:
        raise ValueError("Inputs changed; create a new derivative output version")
    db.execute(
        "INSERT OR IGNORE INTO metadata VALUES (?,?)",
        ("input_manifest", encode(hashes)),
    )
    db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", ("version", VERSION))
    db.commit()
    insert_sources(db, source)
    insert_routes(db, source, raster)
    acquisition = score_routes(
        db, raster, root, online, osm_path=osm, retry_missing=retry_missing
    )
    history = db.execute(
        "SELECT value FROM metadata WHERE key='acquisition_runs'"
    ).fetchone()
    history = json.loads(history[0]) if history else []
    history.append(acquisition)
    db.execute(
        "INSERT OR REPLACE INTO metadata VALUES (?,?)",
        ("acquisition_runs", encode(history)),
    )
    db.commit()
    attach_road_context(db, source)
    attach_places(db)
    attach_claims(db)
    compare_routes(db, read(raster / "config.json")["scoring"])
    errors = validate(db)
    counts = {
        t: db.execute("SELECT count(*) FROM " + t).fetchone()[0]
        for t in [
            "entities",
            "effective_entities",
            "sources",
            "claims",
            "effective_claims",
            "routes",
            "effective_routes",
            "scores",
            "cells",
            "road_links",
            "claim_links",
            "effective_claim_links",
            "place_links",
            "effective_place_links",
            "effective_relationships",
            "invalidations",
            "unresolved",
            "comparisons",
        ]
    }
    result = {
        "version": VERSION,
        "counts": counts,
        "acquisition": acquisition,
        "validation_errors": errors,
        "status": "ready_research_baseline" if not errors else "requires_review",
        "source_inputs": hashes,
        "matching_input": matching,
        "visibility": "nearby_landscape_not_visually_verified",
        "production_defaults_changed": False,
    }
    (out / "release_report.json").write_text(encode(result), encoding="utf8")
    for case, payload in db.execute("SELECT case_id,payload FROM comparisons"):
        folder = out / "comparisons"
        folder.mkdir(exist_ok=True)
        (folder / (digest(case)[:12] + ".json")).write_text(payload, encoding="utf8")
    db.commit()
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    source.close()
    return result


def validate(db):
    errors = []
    checks = {
        "broken_claim_source": "SELECT count(*) FROM claims c LEFT JOIN sources s USING(source_id) WHERE s.source_id IS NULL",
        "held_relationship_leak": "SELECT count(*) FROM effective_place_links WHERE relationship_id IN (SELECT output_id FROM invalidations)",
        "audit_in_default_routes": "SELECT count(*) FROM effective_routes WHERE route_id='"
        + AUDIT_ID
        + "'",
        "missing_active_route_score": "SELECT count(*) FROM effective_routes r LEFT JOIN scores s USING(geometry_id) WHERE s.geometry_id IS NULL",
        "route_cell_length_mismatch": "SELECT count(*) FROM (SELECT r.geometry_id,max(r.length_m) n,sum(c.length_m)/(SELECT count(*) FROM routes x WHERE x.geometry_id=r.geometry_id AND x.role!='original_audit_reference') observed FROM routes r JOIN cells c USING(geometry_id) WHERE r.role!='original_audit_reference' GROUP BY r.geometry_id HAVING abs(n-observed)>0.00001)",
    }
    for label, sql in checks.items():
        if db.execute(sql).fetchone()[0]:
            errors.append(label)
    for (payload,) in db.execute("SELECT payload FROM comparisons"):
        c = json.loads(payload)
        for comparison in c["comparisons"]:
            for split in comparison["landscape_shared_and_differentiating"]:
                n = db.execute(
                    "SELECT length_m FROM routes WHERE route_id=?", (split["route_id"],)
                ).fetchone()[0]
                if (
                    abs(
                        split["shared"]["length_m"]
                        + split["differentiating"]["length_m"]
                        - n
                    )
                    > 1e-5
                ):
                    errors.append("shared_length_double_count_or_gap")
    return errors
