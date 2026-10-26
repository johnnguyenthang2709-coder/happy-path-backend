"""Export the private integrated snapshot and author-generated public metadata."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from happy_path.pretrip.integrated import VERSION, matching_input, readonly
from happy_path.pretrip.raster_scenic import checksum


def finalize(dataset, public_output, *, osm=None):
    dataset, public_output = Path(dataset).resolve(), Path(public_output).resolve()
    report = json.loads((dataset / "release_report.json").read_text("utf8"))
    if osm:
        matching = matching_input(osm)
        if report.get("matching_input") and report["matching_input"] != matching:
            raise ValueError("OSM matching manifest differs from build")
        report["matching_input"] = matching
        (dataset / "release_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
        )
    if report["validation_errors"]:
        raise ValueError("Material dataset validation errors block release")
    db = readonly(dataset / "integrated.sqlite")
    folder = dataset / "portable"
    folder.mkdir(exist_ok=True)
    for table in [
        "entities",
        "sources",
        "claims",
        "routes",
        "scores",
        "cells",
        "road_links",
        "claim_links",
        "place_links",
        "relationships",
        "invalidations",
        "corrections",
        "unresolved",
    ]:
        cursor = db.execute("SELECT * FROM " + table)
        columns = [c[0] for c in cursor.description]
        target_context = (
            (folder / (table + ".jsonl")).open("w", encoding="utf8")
            if table != "cells"
            else nullcontext()
        )
        with target_context as target:
            # Cells are exported as compressed Parquet rather than duplicate JSONL.
            if table == "cells":
                schema = pa.schema(
                    [
                        ("geometry_id", pa.string()),
                        ("cell_index", pa.int64()),
                        ("start_m", pa.float64()),
                        ("end_m", pa.float64()),
                        ("length_m", pa.float64()),
                        ("payload_json", pa.string()),
                    ]
                )
                with pq.ParquetWriter(
                    folder / "cells.parquet", schema, compression="zstd"
                ) as writer:
                    while rows := cursor.fetchmany(1000):
                        writer.write_table(
                            pa.Table.from_pylist(
                                [
                                    dict(
                                        zip(
                                            [
                                                "geometry_id",
                                                "cell_index",
                                                "start_m",
                                                "end_m",
                                                "length_m",
                                                "payload_json",
                                            ],
                                            r,
                                        )
                                    )
                                    for r in rows
                                ],
                                schema=schema,
                            )
                        )
                continue
            while rows := cursor.fetchmany(1000):
                for row in rows:
                    record = dict(zip(columns, row))
                    for key, v in record.items():
                        if isinstance(v, bytes):
                            record[key] = v.hex()
                        elif (
                            key in ["payload", "provenance", "intervals"]
                            and v is not None
                        ):
                            record[key] = json.loads(v)
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics = {
        "public_entities_default": db.execute(
            "SELECT count(*) FROM effective_entities"
        ).fetchone()[0],
        "inferred_entities": db.execute(
            "SELECT count(*) FROM entities WHERE inferred=1"
        ).fetchone()[0],
        "unresolved_entities": db.execute(
            "SELECT count(*) FROM entities WHERE unresolved=1"
        ).fetchone()[0],
        "catalog_roads": db.execute(
            "SELECT count(*) FROM entities WHERE kind='road'"
        ).fetchone()[0],
        "raster_scored_geometries": db.execute(
            "SELECT count(*) FROM scores WHERE json_extract(payload,'$.ranking_eligible')=1"
        ).fetchone()[0],
        "exact_reused_geometries": db.execute(
            "SELECT count(*) FROM scores WHERE json_extract(provenance,'$.method')='identical_registered_geometry_same_crs_and_version'"
        ).fetchone()[0],
        "new_native_geometries": db.execute(
            "SELECT count(*) FROM scores WHERE json_extract(provenance,'$.method')='integrated_derivative_native_extraction_same_policy'"
        ).fetchone()[0],
        "supported_provider_approaches": db.execute(
            "SELECT count(*) FROM effective_place_links WHERE json_extract(payload,'$.provider_routed_approach')=1"
        ).fetchone()[0],
        "compatible_detours": db.execute(
            "SELECT count(*) FROM effective_place_links WHERE json_extract(payload,'$.compatible_detour')=1"
        ).fetchone()[0],
        "default_route_claim_bindings": db.execute(
            "SELECT count(*) FROM effective_claim_links"
        ).fetchone()[0],
        "duplicate_claims_grouped": db.execute(
            "SELECT count(*) FROM claims WHERE claim_id!=canonical_id"
        ).fetchone()[0],
        "default_discovery_relationships": db.execute(
            "SELECT count(*) FROM effective_place_links"
        ).fetchone()[0],
    }
    comparisons = [
        json.loads(r[0])
        for r in db.execute("SELECT payload FROM comparisons ORDER BY case_id")
    ]
    lines = [
        "# Integrated baseline coverage and quality",
        "",
        f"Status: {report['status']}. This is an opt-in research baseline, not navigation or stop-feasibility certification.",
        "",
        "| Measure | Actual count |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in {**report["counts"], **metrics}.items()]
    lines += [
        "",
        "## Baseline comparisons",
        "",
        "All captured priority directed connections remain distinct. Single-candidate cases are not alternative-route choices. Comparative costs apply only within each same-OD case and retain their provider snapshot dates. No winner is forced.",
        "",
        "| Case | Route intent | Scenic | Coverage | Tree points | Water points | Crop points | RMS points | km | Captured min |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in comparisons:
        for r in case["routes"]:
            s = r["scenic"]
            parts = s["component_contributions"]
            get = lambda k, parts=parts: (
                round(parts[k]["conditional_points"], 4)
                if parts[k]["conditional_points"] is not None
                else "unknown"
            )
            lines.append(
                f"| {case['case_id']} | {r['intent']} | {s['score']} | {s['evidence_weight_coverage']:.5f} | {get('tree_cover')} | {get('water')} | {get('cropland')} | {get('ruggedness')} | {r['provider_distance_m'] / 1000:.3f} | {r['provider_duration_s'] / 60:.2f} |"
            )
    lines += [
        "",
        "## Limitations and resolved integration defects",
        "",
        "The source layer is now connected to exact registered raster geometry and newly extracted captured geometry. Same-source duplicate claims retain canonical IDs; local research files are not promoted into public corroboration. Held and superseded relationships remain excluded. Localized La Ngà historical markers and Hải Vân Quan monument context are never whole-road facts. Representative areas are not entrance points.",
        "",
        "Spatial road-context proximity is not field identity. Existing one-way contradictions and uncertain tunnel context remain in raw cells. Raster age (2021), DSM surface objects, native sampling and visibility uncertainty survive narrow missing-input bounds. Historical community conditions remain historical. Access evidence is a provider pin approach; detour evidence is reused only for the same original route snapshot. No parking, opening, public access or operational stop eligibility is invented.",
        "",
        "The initial integration attempt exposed an insert-placeholder defect and was fixed before populated validation. A later worker was restarted after committed raster extraction to replace a quadratic slicing implementation with indexed source-vertex slicing. Initial-attempt acquisition timing/HTTP counters were not saved; they are not claimed as complete measurements. Newly acquired native blocks and their dated receipts remain in the local raster cache. Resume metrics concern that invocation only.",
        "",
        "Full source/provider material is deliberately not published in Git. Private raw originals, the full nationwide raster database and exact input checksums remain in their preserved release folders. The compact private handoff is a separate authorized-transfer artifact, not a public download.",
    ]
    (dataset / "data_quality_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf8"
    )
    (dataset / "README.md").write_text(
        "See docs/phase412_integrated_release.md in the release checkout.\n\nThis local directory contains restricted source/provider data; do not publish its databases, raw exports or private handoffs.\n",
        encoding="utf8",
    )
    db.close()
    public_output.mkdir(parents=True, exist_ok=True)
    # Aggregates and hashes only: no provider geometries, individual metrics,
    # quotations, coordinates, personal identifiers or local absolute paths.
    compact = {
        "version": VERSION,
        "status": report["status"],
        "counts": report["counts"],
        "coverage": metrics,
        "validation_errors": report["validation_errors"],
        "bundled_data": "Author-generated aggregate metadata only; full provider/source snapshots require an authorized private handoff",
        "runtime_default": "unchanged; explicit --enable-research required",
        "policy": {
            "tree": 0.35,
            "water": 0.25,
            "cropland": 0.1,
            "ruggedness": 0.3,
            "rms_saturation_m": 10,
            "minimum_evidence_weight_coverage": 0.8,
        },
        "limitations": [
            "2021 nearby-landscape proxy, not road-visible scenery or safety",
            "Catalog research is incomplete; no forced Scenic/Iconic/Express winner",
            "Inherited holds, ambiguity, inferred positions and source age retained",
        ],
    }
    (public_output / "release_summary.json").write_text(
        json.dumps(compact, indent=2), encoding="utf8"
    )
    (public_output / "input_manifest.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {"name": Path(name.replace("\\", "/")).name, **meta}
                    for name, meta in report["source_inputs"].items()
                ],
                "required_external": [
                    "Phase 4.12 source/validity database",
                    "32-route raster capsule or original raster database",
                    "OSM road matching inventory",
                    "native raster cache for offline extraction, otherwise official public online windows",
                ],
                "osm_matching_inventory": report.get("matching_input"),
                "redistribution": "Input contents not included; provider/source permissions are not inferred from possession of a snapshot",
            },
            indent=2,
        ),
        encoding="utf8",
    )
    paths = [
        dataset / "integrated.sqlite",
        dataset / "release_report.json",
        dataset / "data_quality_report.md",
        *folder.glob("*"),
        *(dataset / "comparisons").glob("*.json"),
    ]
    (dataset / "output_manifest.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "files": [
                    {
                        "path": p.relative_to(dataset).as_posix(),
                        "bytes": p.stat().st_size,
                        "sha256": checksum(p),
                    }
                    for p in paths
                ],
            },
            indent=2,
        ),
        encoding="utf8",
    )
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--osm", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(finalize(args.dataset, args.public_output, osm=args.osm), indent=2)
    )


if __name__ == "__main__":
    main()
