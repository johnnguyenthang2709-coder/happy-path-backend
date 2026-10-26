"""Native cached raster lookup with explicit pixel provenance and no imputation."""

import hashlib
import json

import numpy as np
import rasterio
from shapely.geometry import Point, box
from shapely.strtree import STRtree

from happy_path.core.road_sources import RasterPatches

CLASSES = {10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100}


class NativeRasters:
    def __init__(self, source, supplemental=None):
        receipt_list = json.loads(
            (source / "acquisition_receipts.json").read_text(encoding="utf-8")
        )["raster"]
        entries = [
            (source / "cache/raster" / r["source"] / (r["cache_key"] + ".tif"), r)
            for r in receipt_list
            if r.get("success")
        ]
        if supplemental and supplemental.exists():
            for path in sorted(supplemental.glob("*/*.meta.json")):
                receipt = json.loads(path.read_text(encoding="utf-8"))
                if receipt.get("success"):
                    entries.append(
                        (
                            path.with_name(path.name.replace(".meta.json", ".tif")),
                            receipt,
                        )
                    )
        self.patches = {"worldcover": [], "dem": []}
        seen = set()
        for path, receipt in entries:
            key = (receipt["source"], receipt["cache_key"])
            if key in seen:
                continue
            seen.add(key)
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
                raise ValueError("Native raster checksum mismatch")
            with rasterio.open(path) as ds:
                if ds.crs.to_epsg() != 4326:
                    raise ValueError("Unsupported raster alignment")
                data = ds.read(1, masked=True)
                valid = ~np.ma.getmaskarray(data) & np.isfinite(data.data)
                if receipt["source"] == "worldcover":
                    valid &= np.isin(data.data, list(CLASSES))
                self.patches[receipt["source"]].append(
                    dict(
                        bounds=box(*ds.bounds),
                        data=data.data,
                        valid=valid,
                        inverse=~ds.transform,
                        receipt=receipt,
                        file=str(path),
                    )
                )
        self.trees = {
            p: STRtree([r["bounds"] for r in rows]) for p, rows in self.patches.items()
        }

    def lookup(self, product, longitude, latitude):
        for i in sorted(self.trees[product].query(Point(longitude, latitude))):
            patch = self.patches[product][int(i)]
            col, row = patch["inverse"] * (longitude, latitude)
            col, row = int(np.floor(col)), int(np.floor(row))
            if not (
                0 <= row < patch["data"].shape[0]
                and 0 <= col < patch["data"].shape[1]
                and patch["valid"][row, col]
            ):
                continue
            receipt = patch["receipt"]
            observation = dict(
                value=float(patch["data"][row, col]),
                source_cache_key=receipt["cache_key"],
                source_patch_sha256=receipt["sha256"],
                source_url=receipt["url"],
                retrieved_at=receipt["retrieved_at"],
                native_pixel_row=row,
                native_pixel_column=col,
                product=receipt["product"],
                source_year=receipt.get("year"),
                verification_status="native_source_observation_not_field_verified",
            )
            ruggedness = None
            if (
                product == "dem"
                and 0 < row < patch["data"].shape[0] - 1
                and 0 < col < patch["data"].shape[1] - 1
            ):
                window = patch["data"][row - 1 : row + 2, col - 1 : col + 2]
                if patch["valid"][row - 1 : row + 2, col - 1 : col + 2].all():
                    differences = (window.astype(float) - observation["value"]).ravel()
                    ruggedness = float(np.sqrt(np.mean(np.delete(differences, 4) ** 2)))
            return observation, ruggedness
        return None, None


def acquire_missing(source, output, area_wgs):
    """Download only uncached route neighborhoods to a separate derivative cache."""
    native = NativeRasters(source, output / "cache/raster")
    receipts = []
    from shapely.ops import unary_union

    for product in ("worldcover", "dem"):
        footprint = unary_union([x["bounds"] for x in native.patches[product]])
        missing = area_wgs.difference(footprint)
        if missing.is_empty:
            continue
        patches = RasterPatches(output / "cache/raster" / product, product)
        try:
            patches.acquire(missing, 0.05, 160)
            receipts.extend(patches.receipts)
        finally:
            patches.close()
    return receipts
