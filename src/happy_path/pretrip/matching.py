"""Direction-aware, distance-bounded joins. Parallel ambiguity stays unresolved."""

import json
import math

import numpy as np
import pyarrow.parquet as pq
from shapely import wkb
from shapely.ops import transform
from shapely.strtree import STRtree


def tangent(line, distance):
    a = line.interpolate(max(0, distance - 2))
    b = line.interpolate(min(line.length, distance + 2))
    vector = np.array((b.x - a.x, b.y - a.y))
    length = np.linalg.norm(vector)
    return vector / length if length > 0 else None


class DirectedIndex:
    def __init__(self, rows, geometries):
        self.rows, self.geometries = rows, geometries
        self.tree = STRtree(geometries)

    def match(self, point, direction, tolerance=20, angle=25, ambiguity=5):
        candidates = []
        if direction is None:
            return None, "no_direction"
        for index in self.tree.query(point.buffer(tolerance).envelope):
            geometry = self.geometries[int(index)]
            distance = point.distance(geometry)
            if distance > tolerance:
                continue
            vector = tangent(geometry, geometry.project(point))
            if vector is None:
                continue
            dot = float(np.dot(direction, vector))
            difference = math.degrees(math.acos(min(1, abs(dot))))
            if difference > angle:
                continue
            candidates.append((distance, difference, int(index), dot))
        if not candidates:
            return None, "unmatched"
        candidates.sort(key=lambda v: (v[0], v[1], v[2]))
        distance, difference, index, dot = candidates[0]
        if len(candidates) > 1 and candidates[1][0] - distance <= ambiguity:
            return None, "ambiguous_parallel_or_junction"
        row = self.rows[index]
        tags = row.get("tags", {})
        oneway = str(tags.get("oneway", "")).casefold()
        conflict = (oneway in ("yes", "1", "true") and dot < 0) or (
            oneway == "-1" and dot > 0
        )
        return (
            dict(
                source=row,
                distance_m=distance,
                axis_difference_deg=difference,
                coordinate_order_alignment="same" if dot >= 0 else "reverse",
                oneway_conflict=conflict,
            ),
            "source_oneway_conflict"
            if conflict
            else "supported_geometry_not_field_verified",
        )


def spatial_filter(bounds, padding=0.02):
    west, south, east, north = bounds
    return [
        ("xmin", "<=", east + padding),
        ("xmax", ">=", west - padding),
        ("ymin", "<=", north + padding),
        ("ymax", ">=", south - padding),
    ]


def load_road_indexes(source, enriched, geographic_bounds, forward):
    national = pq.read_table(
        enriched / "nationwide/base_roads.parquet",
        filters=spatial_filter(geographic_bounds),
        columns=[
            "entity_id",
            "name",
            "tags_json",
            "geometry_wkb",
            "osm_timestamp",
            "osm_version",
            "source_evidence_id",
        ],
    ).to_pylist()
    for row in national:
        row["tags"] = json.loads(row["tags_json"])
    national_geometries = [
        transform(forward, wkb.loads(r["geometry_wkb"])) for r in national
    ]
    corridor = pq.read_table(enriched / "enriched_road_segments.parquet").to_pylist()
    audits = {
        r["segment_id"]: r
        for r in pq.read_table(source / "validated/segment_audit.parquet").to_pylist()
    }
    groups = json.loads(
        (enriched / "enriched_experience_segments.geojson").read_text(encoding="utf-8")
    )["features"]
    memberships = {
        identity: f["id"]
        for f in groups
        for identity in f["properties"]["original_segment_ids"]
    }
    eligible = []
    for row in corridor:
        if audits[row["segment_id"]]["uturn_overlap"]:
            continue
        row["original_binding_flags"] = audits[row["segment_id"]]["flags"]
        row["experience_group_id"] = memberships[row["segment_id"]]
        eligible.append(row)
    corridor_geometries = [
        transform(forward, wkb.loads(r["geometry_wkb"])) for r in eligible
    ]
    return DirectedIndex(national, national_geometries), DirectedIndex(
        eligible, corridor_geometries
    )


def matched_cells(route, road_index, corridor_index, rasters, forward, inverse, config):
    line = route["projected_geometry"]
    settings = config["matching"]
    spacing = settings["sample_spacing_m"]
    rows = []
    for start in np.arange(0, line.length, spacing):
        start = float(start)
        end = min(line.length, start + spacing)
        middle = (start + end) / 2
        point = line.interpolate(middle)
        direction = tangent(line, middle)
        road, status = road_index.match(
            point,
            direction,
            settings["distance_tolerance_m"],
            settings["bearing_tolerance_deg"],
            settings["ambiguity_margin_m"],
        )
        corridor, cstatus = corridor_index.match(
            point,
            direction,
            settings["distance_tolerance_m"],
            settings["bearing_tolerance_deg"],
            settings["ambiguity_margin_m"],
        )
        longitude, latitude = inverse(point.x, point.y)
        dem, ruggedness = rasters.lookup("dem", longitude, latitude)
        samples = []
        for offset in (-250, -150, -50, 50, 150, 250):
            if direction is None:
                continue
            x = point.x - direction[1] * offset
            y = point.y + direction[0] * offset
            lon, lat = inverse(x, y)
            observation, _ = rasters.lookup("worldcover", lon, lat)
            samples.append(
                dict(
                    offset_left_m=offset,
                    longitude=lon,
                    latitude=lat,
                    observation=observation,
                )
            )
        observed = [
            s["observation"]["value"] for s in samples if s["observation"] is not None
        ]
        source_road = road["source"] if road else None
        source_corridor = corridor["source"] if corridor else None
        rows.append(
            dict(
                route_id=route["route_id"],
                cell_index=len(rows),
                start_m=start,
                end_m=end,
                cell_length_m=end - start,
                midpoint_longitude=longitude,
                midpoint_latitude=latitude,
                road_match_status=status,
                matched_osm_id=source_road["entity_id"] if source_road else None,
                matched_osm_name=source_road["name"] if source_road else None,
                source_road_tags_json=source_road["tags_json"] if source_road else None,
                source_osm_timestamp=source_road["osm_timestamp"]
                if source_road
                else None,
                source_osm_version=source_road["osm_version"] if source_road else None,
                source_osm_snapshot_evidence_id=source_road["source_evidence_id"]
                if source_road
                else None,
                source_road_distance_m=road["distance_m"] if road else None,
                source_road_axis_difference_deg=road["axis_difference_deg"]
                if road
                else None,
                oneway_conflict=road["oneway_conflict"] if road else None,
                corridor_match_status=cstatus,
                original_segment_id=source_corridor["segment_id"]
                if source_corridor
                else None,
                experience_group_id=source_corridor["experience_group_id"]
                if source_corridor
                else None,
                original_binding_flags=source_corridor["original_binding_flags"]
                if source_corridor
                else [],
                dsm_elevation_m=dem["value"] if dem else None,
                dsm_ruggedness_rms_m=ruggedness,
                dsm_provenance_json=json.dumps(dem, sort_keys=True) if dem else None,
                roadside_worldcover_samples_json=json.dumps(samples, sort_keys=True),
                roadside_sample_coverage=len(observed) / 6,
                roadside_tree_fraction=observed.count(10) / len(observed)
                if observed
                else None,
                roadside_water_fraction=observed.count(80) / len(observed)
                if observed
                else None,
                roadside_cropland_fraction=observed.count(40) / len(observed)
                if observed
                else None,
                roadside_built_up_fraction=observed.count(50) / len(observed)
                if observed
                else None,
                roadside_class_sample_counts_json=json.dumps(
                    {str(int(k)): observed.count(k) for k in sorted(set(observed))}
                ),
                visibility_status="nearby_landscape_samples_not_visibility_evidence",
            )
        )
    if abs(sum(r["cell_length_m"] for r in rows) - line.length) > 1e-5:
        raise ValueError(
            "Route cell weights do not cover the source geometry exactly once"
        )
    return rows


def alignment_audit(route, cells, index, forward):
    """Record actual contiguous contradictions and nearby source alternatives, without repairing them."""
    runs, current = [], []
    for cell in cells:
        if cell["road_match_status"] == "source_oneway_conflict":
            current.append(cell)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    observations = []
    from shapely.geometry import Point

    for run in runs:
        representative = run[len(run) // 2]
        point = transform(
            forward,
            Point(
                representative["midpoint_longitude"],
                representative["midpoint_latitude"],
            ),
        )
        direction = tangent(
            route["projected_geometry"],
            (representative["start_m"] + representative["end_m"]) / 2,
        )
        candidates = []
        for i in index.tree.query(point.buffer(50).envelope):
            geometry = index.geometries[int(i)]
            distance = point.distance(geometry)
            vector = tangent(geometry, geometry.project(point))
            if distance > 50 or vector is None or direction is None:
                continue
            dot = float(np.dot(vector, direction))
            angle = math.degrees(math.acos(min(1, abs(dot))))
            if angle > 25:
                continue
            row = index.rows[int(i)]
            one = str(row["tags"].get("oneway", "")).casefold()
            conflict = (one in ("1", "yes", "true") and dot < 0) or (
                one == "-1" and dot > 0
            )
            candidates.append(
                dict(
                    source_osm_id=row["entity_id"],
                    name=row["name"],
                    ref=row["tags"].get("ref"),
                    source_oneway=one or None,
                    distance_m=distance,
                    axis_difference_deg=angle,
                    direction_contradicts_source_tag=conflict,
                    source_osm_timestamp=row["osm_timestamp"],
                    status="audit_candidate_only_not_a_resolved_binding",
                )
            )
        observations.append(
            dict(
                start_m=run[0]["start_m"],
                end_m=run[-1]["end_m"],
                length_m=sum(c["cell_length_m"] for c in run),
                closest_conflicting_source_ids=sorted(
                    {c["matched_osm_id"] for c in run}
                ),
                representative_coordinate=[
                    representative["midpoint_longitude"],
                    representative["midpoint_latitude"],
                ],
                nearby_candidates=sorted(candidates, key=lambda r: r["distance_m"]),
                resolution="unresolved; no source attributes or traffic restrictions promoted from conflicting bindings",
            )
        )
    return dict(
        route_id=route["route_id"],
        request_id=route["request_id"],
        provider_uturns=route["provider_uturns"],
        source_direction_conflict_m=sum(x["length_m"] for x in observations),
        source_direction_conflict_runs=observations,
        interpretation="Nearby opposite-direction or parallel carriageways can explain provider/OSM alignment differences; candidate evidence is not a road-legality determination",
    )
