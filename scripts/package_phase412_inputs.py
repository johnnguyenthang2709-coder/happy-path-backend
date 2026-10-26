"""Create a LOCAL PRIVATE input handoff; never publish its provider/source content."""

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from happy_path.pretrip.integrated import readonly
from happy_path.pretrip.raster_scenic import checksum


def include_legacy_crops(phase, output):
    """Bundle exactly the validated crop families the integrated extractor reads."""
    cache = Path(output) / "raster/raster_cache"
    cache.mkdir(exist_ok=True)
    added = 0
    for folder in Path(phase).parent.glob("phase4*/cache/raster"):
        for kind in ["worldcover", "dem"]:
            for meta in (folder / kind).glob("*.meta.json"):
                receipt = json.loads(meta.read_text("utf8"))
                path = meta.with_name(meta.name.replace(".meta.json", ".tif"))
                if not (
                    receipt.get("success")
                    and receipt.get("year") == 2021
                    and receipt.get("crop_method") == "native_pixel_center_partition_v2"
                    and path.exists()
                ):
                    continue
                if checksum(path) != receipt["sha256"]:
                    raise ValueError("Legacy crop checksum mismatch")
                dest = cache / ("legacy_" + receipt["sha256"] + ".tif")
                if not dest.exists():
                    shutil.copyfile(path, dest)
                    shutil.copyfile(meta, dest.with_suffix(".meta.json"))
                    added += 1
    return added


def package(phase, raster, osm, output, *, include_rasters=False, derived_cache=None):
    phase, raster, osm, output = [
        Path(p).resolve() for p in [phase, raster, osm, output]
    ]
    if output.exists():
        raise ValueError(
            "Use a new handoff directory; no existing dataset is overwritten"
        )
    if shutil.disk_usage(output.parent).free < 4 * 1024**3:
        raise ValueError("Insufficient free space for a bounded private handoff")
    (output / "phase").mkdir(parents=True)
    (output / "raster").mkdir()
    shutil.copyfile(
        phase / "national_mapping.sqlite", output / "phase/national_mapping.sqlite"
    )
    source = readonly(raster / "scenic.sqlite")
    target = sqlite3.connect(output / "raster/scenic.sqlite")
    for table in ["jobs", "cells", "raster_assets"]:
        target.execute(
            source.execute(
                "SELECT sql FROM sqlite_master WHERE name=?", (table,)
            ).fetchone()[0]
        )
    ids = [
        r[0]
        for r in source.execute(
            "SELECT DISTINCT geometry_id FROM sections WHERE kind='route'"
        )
    ]
    for table in ["jobs", "cells"]:
        columns = [r[1] for r in source.execute("PRAGMA table_info(" + table + ")")]
        for gid in ids:
            cursor = source.execute(
                "SELECT * FROM " + table + " WHERE geometry_id=?", (gid,)
            )
            while rows := cursor.fetchmany(1000):
                target.executemany(
                    "INSERT INTO "
                    + table
                    + " VALUES ("
                    + ",".join("?" for _ in columns)
                    + ")",
                    rows,
                )
        target.commit()
    target.executemany(
        "INSERT INTO raster_assets VALUES (?,?,?)",
        source.execute("SELECT * FROM raster_assets"),
    )
    target.commit()
    target.close()
    source.close()
    for name in ["route_register.json", "config.json"]:
        shutil.copyfile(raster / name, output / "raster" / name)
    shutil.copyfile(osm, output / "base_roads.parquet")
    if include_rasters:
        cache = output / "raster/raster_cache"
        cache.mkdir()
        for folder in [
            raster / "raster_cache",
            Path(derived_cache) if derived_cache else None,
        ]:
            if not folder or not folder.exists():
                continue
            for p in folder.iterdir():
                if p.is_file() and (
                    p.suffix == ".tif"
                    or p.name.endswith(".meta.json")
                    or p.name == ".happy_path_raster_cache.json"
                ):
                    target = cache / p.name
                    if target.exists() and p.name == ".happy_path_raster_cache.json":
                        if json.loads(target.read_text("utf8"))["purpose"] != json.loads(p.read_text("utf8"))["purpose"]:
                            raise ValueError("Conflicting cache ownership")
                        continue
                    if target.exists() and checksum(target) != checksum(p):
                        raise ValueError("Conflicting cached asset")
                    if not target.exists():
                        shutil.copyfile(p, target)
        include_legacy_crops(phase, output)
    files = [p for p in output.rglob("*") if p.is_file()]
    manifest = {
        "classification": "PRIVATE handoff, not authorized for public GitHub redistribution",
        "scope": "Full Phase 4.12 effective/current source snapshot; only the 32 priority/reference raster geometries, plus the national OSM matching inventory. Nationwide raster cell database is intentionally omitted.",
        "rasters_included": include_rasters,
        "limitations": "No provider credentials included. Queries after build are offline. If native rasters are omitted, building new geometries requires permitted online raster reads. Validated 2021 legacy native crop families used by the extractor are included when available; other historic raw assets and full upstream source captures remain external for byte-level re-audits.",
        "origin_input_sha256": {
            name: checksum(path)
            for name, path in [
                ("phase_database", phase / "national_mapping.sqlite"),
                ("full_raster_database", raster / "scenic.sqlite"),
                ("osm_inventory", osm),
            ]
        },
        "files": [
            {
                "path": p.relative_to(output).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": checksum(p),
            }
            for p in files
        ],
    }
    (output / "PRIVATE_INPUT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf8"
    )
    return {
        "directory": str(output),
        "file_count": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "private": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["phase", "raster", "osm", "output"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--include-rasters", action="store_true")
    parser.add_argument("--derived-cache", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            package(
                args.phase,
                args.raster,
                args.osm,
                args.output,
                include_rasters=args.include_rasters,
                derived_cache=args.derived_cache,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
