"""Read-only dataset checks and actual historical-input comparisons."""

import json
from collections import Counter

import pyarrow.parquet as pq
from shapely import from_wkb
from shapely.ops import transform

from .raster_scenic import checksum
from .raster_scenic_jobs import projection, read, save


def validate(root, out, db):
    failures = []
    checks = {}
    queries = {
        "invalid_cell_length": "SELECT count(*) FROM cells WHERE length_m<=0 OR length_m>50.000001 OR abs(length_m-(end_m-start_m))>0.000001",
        "unsupported_coordinate_order": "SELECT count(*) FROM cells WHERE lon NOT BETWEEN 100 AND 116 OR lat NOT BETWEEN 6 AND 25",
        "tunnel_score_leak": "SELECT count(*) FROM cells WHERE tunnel=1 AND (tree IS NOT NULL OR water IS NOT NULL OR cropland IS NOT NULL OR ruggedness IS NOT NULL OR sample_coverage!=0)",
        "duplicate_cell_grid": "SELECT count(*) FROM (SELECT geometry_id,cell_index FROM cells GROUP BY geometry_id,cell_index HAVING count(*)>1)",
        "completed_length_gap": "SELECT count(*) FROM (SELECT j.geometry_id,j.length_m,sum(c.length_m) n FROM jobs j LEFT JOIN cells c USING(geometry_id) WHERE j.status LIKE 'processed%' GROUP BY j.geometry_id HAVING n IS NULL OR abs(j.length_m-n)>0.000001)",
    }
    for name, query in queries.items():
        checks[name] = db.execute(query).fetchone()[0]
        if checks[name]:
            failures.append(name)
    score_checks = Counter()
    for status, payload in db.execute(
        "SELECT status,score_json FROM jobs WHERE score_json IS NOT NULL"
    ):
        s = json.loads(payload)
        score_checks["checked"] += 1
        if s["ranking_eligible"] != (s["evidence_weight_coverage"] >= 0.8):
            score_checks["coverage_gate_mismatch"] += 1
        if s["observed_score"] is not None:
            total = sum(
                v["conditional_points"] or 0
                for v in s["component_contributions"].values()
            )
            if abs(total - s["observed_score"]) > 1e-8:
                score_checks["contribution_mismatch"] += 1
        lower, upper = s["missing_evidence_bounds"]
        if not 0 <= lower <= upper <= 100.000001:
            score_checks["invalid_bounds"] += 1
    failures += [k for k in score_checks if k != "checked" and score_checks[k]]
    manifest = read(out / "input_manifest.json")
    protected = {}
    for key in ["inputs_sha256", "protected_runtime_data_sha256"]:
        for name, expected in manifest.get(key, {}).items():
            protected[name] = checksum(root / name.replace("\\", "/")) == expected
    if not all(protected.values()):
        failures.append("protected_input_changed")
    # Compare actual old/new cell locations; never compare independent grids as if identical.
    columns = [
        "route_id",
        "cell_index",
        "midpoint_longitude",
        "midpoint_latitude",
        "roadside_tree_fraction",
        "roadside_water_fraction",
        "roadside_cropland_fraction",
        "roadside_sample_coverage",
        "dsm_ruggedness_rms_m",
        "supported_tunnel_excluded",
    ]
    old = {}
    for batch in pq.ParquetFile(
        root / "exports/route_intelligence/phase4_6/final/route_cell_features.parquet"
    ).iter_batches(columns=columns):
        for c in batch.to_pylist():
            old[(c["route_id"], c["cell_index"])] = c
    comparisons = Counter()
    fields = Counter()
    examples = []
    for gid, payload in db.execute(
        "SELECT geometry_id,payload FROM sections WHERE kind='route'"
    ):
        r = json.loads(payload)
        rid = r.get("historical_route_id", r["route_id"])
        for (payload,) in db.execute(
            "SELECT payload FROM cells WHERE geometry_id=?", (gid,)
        ):
            c = json.loads(payload)
            before = old.get((rid, c["cell_index"]))
            if not before:
                continue
            if (
                max(
                    abs(c[k] - before[k])
                    for k in ["midpoint_longitude", "midpoint_latitude"]
                )
                > 1e-9
            ):
                comparisons["different_grid_not_numerically_compared"] += 1
                continue
            comparisons["matching_sampling_locations"] += 1
            different = []
            for key in columns[4:]:
                a, b = c[key], before[key]
                equal = (a is None and b is None) or (
                    a is not None and b is not None and abs(float(a) - float(b)) <= 1e-6
                )
                if not equal:
                    fields[key] += 1
                    different.append({"field": key, "historical": b, "current": a})
            if not different:
                comparisons["identical_inputs"] += 1
            elif len(examples) < 20:
                examples.append(
                    {
                        "route_id": rid,
                        "cell_index": c["cell_index"],
                        "differences": different,
                    }
                )
    support = db.execute(
        "SELECT count(*),sum(length_m),sum(length_m*sample_coverage),sum(CASE WHEN ruggedness IS NOT NULL THEN length_m ELSE 0 END),sum(CASE WHEN tunnel=1 THEN length_m ELSE 0 END),sum(CASE WHEN sample_coverage<1 AND tunnel=0 THEN length_m ELSE 0 END) FROM cells"
    ).fetchone()
    result = {
        "checks": checks,
        "score_checks": dict(score_checks),
        "protected_inputs": protected,
        "historical_actual_input_comparison": {
            "counts": dict(comparisons),
            "differing_fields": dict(fields),
            "examples": examples,
            "interpretation": "Same scoring function and policy remain unchanged. Input differences are disclosed; new source snapshot, native-edge recovery and road/tunnel attribution may differ. Different sampling locations are explicitly excluded from equality claims.",
        },
        "cell_support": dict(
            zip(
                [
                    "cells",
                    "sampled_traversal_m",
                    "worldcover_valid_sample_equivalent_m",
                    "native_3x3_supported_m",
                    "supported_tunnel_excluded_m",
                    "partial_worldcover_non_tunnel_m",
                ],
                support,
            )
        ),
        "failures": failures,
        "status": "passed" if not failures else "requires_review",
    }
    # Raw observations survive tunnel exclusions. Acquisition/extraction support
    # is deliberately distinct from eligibility of those inputs for scoring.
    raw = db.execute(
        "SELECT sum(length_m*json_extract(payload,'$.roadside_valid_samples')/6.0),sum(CASE WHEN json_extract(payload,'$.dsm_3x3_valid_samples')=9 THEN length_m ELSE 0 END),sum(CASE WHEN json_extract(payload,'$.dsm_elevation_m') IS NOT NULL THEN length_m ELSE 0 END) FROM cells"
    ).fetchone()
    result["cell_support"]["scored_native_3x3_supported_m"] = result[
        "cell_support"
    ].pop("native_3x3_supported_m")
    result["raw_acquired_and_extracted_support"] = dict(
        zip(
            [
                "worldcover_valid_sample_equivalent_m",
                "native_3x3_supported_m",
                "native_center_elevation_supported_m",
            ],
            raw,
        )
    )
    result["support_scope"] = (
        "Projected sampled geometry traversals, including overlapping registered routes; not unique physical national road kilometers. Raw raster observations are not actual road visibility or permission to score tunnel surface surroundings."
    )
    missing = []
    for gid, wkb, crs, length, payload in db.execute(
        "SELECT geometry_id,geometry,crs,length_m,score_json FROM jobs WHERE status='processed_missing_inputs'"
    ):
        line = transform(projection(crs).transform, from_wkb(wkb))
        score = json.loads(payload)
        if length <= 4 and line.is_closed:
            reason = "Closed source loop shorter than the ±2m tangent span: tangent endpoints coincide, so directional roadside samples are undefined. Native DSM remains separately available; no alternate tangent method or synthetic raster input substituted."
        else:
            reason = "One or more scored component supports are below the policy threshold; inspect cells, native pixel references and acquisition failure queue. Unknown is not observed zero."
        missing.append(
            {"geometry_id": gid, "length_m": length, "scenic": score, "reason": reason}
        )
    result["missing_input_dispositions"] = missing
    save(out, "dataset_validation.json", result)
    text = (
        "# Dataset validation and limitations\n\n"
        + json.dumps(result, ensure_ascii=False, indent=2)
        + "\n\nScoring eligibility is evidence-weight support, not field verification or a recommendation. Source age (2021 rasters), DSM surface objects, discrete sampling, road alignment, unresolved tunnel context, visibility and preference uncertainty remain. Unknown raster support is not observed zero. Failed requests remain retryable; pending jobs are not completed acquisition. The geodesic centerline union is distinct from projected traversal lengths.\n"
    )
    (out / "data_quality_report.md").write_text(text, encoding="utf8")
    return result
