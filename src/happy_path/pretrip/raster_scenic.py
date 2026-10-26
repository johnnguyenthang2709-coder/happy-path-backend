"""Bounded native raster windows for opt-in Scenic analysis. No production imports."""

import ctypes
import hashlib
import json
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely import points
from shapely.geometry import box
from shapely.strtree import STRtree

from happy_path.core.road_sources import (
    ATTRIBUTIONS,
    atomic,
    json_bytes,
    now,
    raster_url,
)
from happy_path.pretrip.rasters import CLASSES

VERSION = "native-roadside-v2-worldcover2021-dsm2021"


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class StorageBudgetError(RuntimeError):
    pass


class NetworkStats:
    """GDAL's own HTTP byte counters, not an estimate from raster file sizes."""

    def __init__(self):
        self.lib = None
        folders = [
            Path(rasterio.__file__).parent.parent / "rasterio.libs",
            Path(rasterio.__file__).parent.parent / "rasterio.libs",
        ]
        for folder in folders:
            for path in [*folder.glob("*gdal*.dll"), *folder.glob("libgdal*.so*")]:
                try:
                    lib = ctypes.CDLL(str(path))
                    lib.VSINetworkStatsGetAsSerializedJSON.argtypes = [ctypes.c_void_p]
                    lib.VSINetworkStatsGetAsSerializedJSON.restype = ctypes.c_void_p
                    lib.VSIFree.argtypes = [ctypes.c_void_p]
                    lib.VSINetworkStatsReset()
                    self.lib = lib
                    return
                except (OSError, AttributeError):
                    continue

    def snapshot(self):
        if self.lib is None:
            return {
                "available": False,
                "reason": "GDAL network counter symbol unavailable; cache bytes are reported separately",
            }
        ptr = self.lib.VSINetworkStatsGetAsSerializedJSON(None)
        try:
            return json.loads(ctypes.string_at(ptr))
        finally:
            self.lib.VSIFree(ptr)


class RasterStore:
    """Read historical crops first; acquire official COG blocks only for missing pixels.

    Metadata/index is small. Arrays and open handles use fixed-size LRU caches.
    DSM neighbors cross crop/product edges by compatible actual native centers.
    """

    def __init__(
        self,
        cache,
        inherited_roots=(),
        *,
        online=False,
        budget_bytes=8 * 1024**3,
        reserve_bytes=8 * 1024**3,
        ram_bytes=128 * 1024**2,
    ):
        self.cache = Path(cache).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        marker = self.cache / ".happy_path_raster_cache.json"
        if marker.exists():
            if (
                json.loads(marker.read_text("utf8")).get("purpose")
                != "happy_path_raster_scenic_owned_blocks_v1"
            ):
                raise StorageBudgetError(
                    "Cache ownership marker belongs to another dataset"
                )
        elif any(self.cache.glob("*.tif")):
            raise StorageBudgetError(
                "Refusing to adopt a populated cache without a task ownership marker"
            )
        else:
            atomic(
                marker,
                json_bytes(
                    {
                        "purpose": "happy_path_raster_scenic_owned_blocks_v1",
                        "created_at": now(),
                    }
                ),
            )
        self.online = online
        self.budget = budget_bytes
        self.reserve = reserve_bytes
        self.ram_budget = ram_bytes
        self.assets = []
        self.arrays = OrderedDict()
        self.array_bytes = 0
        self.handles = OrderedDict()
        self.failed_urls = {}
        self.failures = []
        self.acquisition_s = 0.0
        self.acquired_bytes = 0
        self.source_bytes = 0
        self.requests = 0
        self.cache_reads = 0
        self.env = rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_TIMEOUT="25",
            GDAL_HTTP_MAX_RETRY="1",
            GDAL_HTTP_RETRY_DELAY="1",
            GDAL_CACHEMAX=32 * 1024**2,
            CPL_VSIL_CURL_CACHE_SIZE=8 * 1024**2,
            CPL_VSIL_NETWORK_STATS_ENABLED="YES",
        )
        self.env.__enter__()
        self.network = NetworkStats()
        self.owned_bytes = sum(p.stat().st_size for p in self.cache.glob("*.tif"))
        seen = set()
        for root in [*map(Path, inherited_roots), self.cache]:
            for meta in sorted(root.glob("*.meta.json")):
                r = json.loads(meta.read_text("utf8"))
                path = meta.with_name(meta.name.replace(".meta.json", ".tif"))
                if (
                    not r.get("success")
                    or r.get("source") not in ["worldcover", "dem"]
                    or not path.exists()
                ):
                    continue
                if r.get("crop_method") not in [
                    "native_pixel_center_partition_v2",
                    "native_cog_block_v1",
                ]:
                    continue
                if r.get("year") != 2021:
                    continue
                if r["sha256"] in seen:
                    continue
                if checksum(path) != r["sha256"]:
                    raise ValueError("Raster checksum mismatch: " + str(path))
                seen.add(r["sha256"])
                self._add(path, r, root == self.cache)
        self._index()

    def _add(self, path, receipt, owned):
        with rasterio.open(path) as ds:
            if ds.crs is None or ds.crs.to_epsg() != 4326 or ds.count != 1:
                raise ValueError("Unexpected source raster CRS/bands")
            if (
                ds.transform.a <= 0
                or ds.transform.e >= 0
                or ds.transform.b
                or ds.transform.d
            ):
                raise ValueError("Unsupported rotated or reversed native grid")
            self.assets = [a for a in self.assets if a["path"] != path]
            self.assets.append(
                {
                    "path": path,
                    "receipt": receipt,
                    "owned": owned,
                    "bounds": tuple(ds.bounds),
                    "transform": ds.transform,
                    "inverse": ~ds.transform,
                    "shape": (ds.height, ds.width),
                    "nodata": ds.nodata,
                    "key": receipt["cache_key"] + ":" + receipt["sha256"],
                }
            )

    def _index(self):
        self.by_kind = {
            k: [
                a
                for a in self.assets
                if a["receipt"]["source"] == k and a["path"].exists()
            ]
            for k in ["worldcover", "dem"]
        }
        self.trees = {
            k: STRtree([box(*a["bounds"]) for a in v]) for k, v in self.by_kind.items()
        }

    def _array(self, asset):
        key = asset["key"]
        if key in self.arrays:
            self.arrays.move_to_end(key)
            return self.arrays[key]
        with rasterio.open(asset["path"]) as ds:
            a = ds.read(1, masked=True)
        data = a.data
        valid = ~np.ma.getmaskarray(a) & np.isfinite(data)
        if asset["receipt"]["source"] == "worldcover":
            valid &= np.isin(data, list(CLASSES))
        size = data.nbytes + valid.nbytes
        while self.arrays and self.array_bytes + size > self.ram_budget:
            _, (_, _, n) = self.arrays.popitem(last=False)
            self.array_bytes -= n
        self.arrays[key] = (data, valid, size)
        self.array_bytes += size
        self.cache_reads += 1
        return self.arrays[key]

    def budget_check(self, required):
        if required > self.budget:
            raise StorageBudgetError(
                "One native block exceeds the configured cache budget"
            )
        # Only this task's reproducible acquired blocks are evictable, never inherited assets.
        candidates = sorted(
            (a for a in self.assets if a["owned"] and a["path"].exists()),
            key=lambda a: a["path"].stat().st_mtime,
        )
        while self.owned_bytes + required > self.budget and candidates:
            a = candidates.pop(0)
            target = a["path"].resolve()
            if target.parent != self.cache:
                raise StorageBudgetError(
                    "Refusing cache eviction outside configured cache directory"
                )
            n = target.stat().st_size
            target.unlink()
            self.owned_bytes -= n
            if a["key"] in self.arrays:
                self.array_bytes -= self.arrays.pop(a["key"])[2]
        if self.owned_bytes + required > self.budget:
            raise StorageBudgetError("Cache budget cannot admit next block")
        if shutil.disk_usage(self.cache).free < self.reserve + required:
            raise StorageBudgetError(
                "D: free-space reserve reached; progress remains committed"
            )

    def _handle(self, url):
        if url in self.handles:
            self.handles.move_to_end(url)
            return self.handles[url]
        ds = rasterio.open(url)
        if ds.crs.to_epsg() != 4326:
            ds.close()
            raise ValueError("Unexpected remote raster CRS")
        while len(self.handles) >= 4:
            self.handles.popitem(last=False)[1].close()
        self.handles[url] = ds
        return ds

    def _acquire(self, kind, coordinates):
        groups = {}
        for lon, lat in coordinates:
            groups.setdefault(raster_url(kind, lat, lon), []).append((lon, lat))
        added = False
        queue = list(groups.items())
        visited = set()
        while queue:
            url, coords = queue.pop(0)
            coords = [
                (lon, lat) for lon, lat in coords if (url, lon, lat) not in visited
            ]
            if not coords:
                continue
            visited.update((url, lon, lat) for lon, lat in coords)
            if not self.online or url in self.failed_urls:
                continue
            start = time.perf_counter()
            try:
                ds = self._handle(url)
                blocks = {}
                for lon, lat in coords:
                    row, col = ds.index(lon, lat)
                    if not (0 <= row < ds.height and 0 <= col < ds.width):
                        # Public DSM COGs have pixel-center aligned outer edges.
                        # Ask the adjacent official product, never fabricate a
                        # border value or treat the wrong tile as missing land.
                        shifted_lon = lon + ds.res[0] * (
                            1 if col >= ds.width else -1 if col < 0 else 0
                        )
                        shifted_lat = lat + ds.res[1] * (
                            1 if row < 0 else -1 if row >= ds.height else 0
                        )
                        neighbor = raster_url(kind, shifted_lat, shifted_lon)
                        if neighbor != url and (neighbor, lon, lat) not in visited:
                            existing = next(
                                (item for item in queue if item[0] == neighbor), None
                            )
                            if existing:
                                existing[1].append((lon, lat))
                            else:
                                queue.append((neighbor, [(lon, lat)]))
                        continue
                    blocks[(row // 1024, col // 1024)] = None
                for br, bc in blocks:
                    key = digest([url, br, bc, "native_cog_block_v1"])
                    path = self.cache / (key + ".tif")
                    meta = path.with_suffix(".meta.json")
                    if path.exists() and meta.exists():
                        continue
                    window = Window(
                        bc * 1024,
                        br * 1024,
                        min(1024, ds.width - bc * 1024),
                        min(1024, ds.height - br * 1024),
                    )
                    if window.width <= 0 or window.height <= 0:
                        continue
                    estimated = (
                        int(
                            window.width
                            * window.height
                            * np.dtype(ds.dtypes[0]).itemsize
                            * 2
                        )
                        + 65536
                    )
                    self.budget_check(estimated)
                    a = ds.read(1, window=window, masked=True)
                    self.requests += 1
                    self.source_bytes += a.nbytes
                    profile = ds.profile.copy()
                    profile.update(
                        driver="GTiff",
                        width=int(window.width),
                        height=int(window.height),
                        transform=ds.window_transform(window),
                        compress="deflate",
                        tiled=True,
                        blockxsize=256,
                        blockysize=256,
                    )
                    temp = path.with_suffix(".tmp.tif")
                    with rasterio.open(temp, "w", **profile) as target:
                        target.write(
                            a.filled(ds.nodata if ds.nodata is not None else 0), 1
                        )
                        target.write_mask(
                            (~np.ma.getmaskarray(a) * 255).astype("uint8")
                        )
                    os.replace(temp, path)
                    receipt = {
                        "source": kind,
                        "url": url,
                        "cache_key": key,
                        "sha256": checksum(path),
                        "success": True,
                        "retrieved_at": now(),
                        "crop_method": "native_cog_block_v1",
                        "native_window": [
                            window.col_off,
                            window.row_off,
                            window.width,
                            window.height,
                        ],
                        "crs": str(ds.crs),
                        "resolution_deg": list(ds.res),
                        "nodata": ds.nodata,
                        "shape": list(a.shape),
                        "bytes": path.stat().st_size,
                        "transform": list(ds.window_transform(window)),
                        **ATTRIBUTIONS[kind],
                    }
                    atomic(meta, json_bytes(receipt))
                    self.owned_bytes += path.stat().st_size
                    self.acquired_bytes += path.stat().st_size
                    self._add(path, receipt, True)
                    added = True
            except StorageBudgetError:
                raise
            except (rasterio.errors.RasterioError, ValueError) as error:
                self.failed_urls[url] = type(error).__name__
                self.failures.append(
                    {
                        "source": kind,
                        "url": url,
                        "retrieved_at": now(),
                        "reason": str(error),
                        "retryable": True,
                    }
                )
            finally:
                self.acquisition_s += time.perf_counter() - start
        if added:
            self._index()

    def sample(self, kind, coords, *, acquire=True):
        coords = np.asarray(coords, dtype=float)
        n = len(coords)
        values = np.full(n, np.nan)
        centers = np.full((n, 2), np.nan)
        res = np.full((n, 2), np.nan)
        refs = [None] * n
        conflicting = np.zeros(n, dtype=bool)
        if not n:
            return values, centers, res, refs

        def local():
            pairs = self.trees[kind].query(points(coords))
            if not pairs.size:
                return
            for ai in np.unique(pairs[1]):
                ix = pairs[0, pairs[1] == ai]
                asset = self.by_kind[kind][int(ai)]
                data, valid, _ = self._array(asset)
                affine = asset["transform"]
                inv = asset["inverse"]
                col = np.floor(inv.a * coords[ix, 0] + inv.c).astype(int)
                row = np.floor(inv.e * coords[ix, 1] + inv.f).astype(int)
                okay = (
                    (row >= 0)
                    & (col >= 0)
                    & (row < data.shape[0])
                    & (col < data.shape[1])
                )
                ii = ix[okay]
                rr = row[okay]
                cc = col[okay]
                v = valid[rr, cc]
                ii = ii[v]
                rr = rr[v]
                cc = cc[v]
                for j, r, c in zip(ii, rr, cc):
                    if conflicting[j]:
                        continue
                    value = float(data[r, c])
                    if np.isfinite(values[j]) and abs(values[j] - value) > 1e-6:
                        values[j] = np.nan
                        refs[j] = None
                        conflicting[j] = True
                        continue
                    if refs[j] is not None:
                        continue
                    values[j] = value
                    centers[j] = affine * (int(c) + 0.5, int(r) + 0.5)
                    res[j] = [affine.a, affine.e]
                    refs[j] = [asset["key"], int(r), int(c)]

        local()
        missing = ~np.isfinite(values)
        if missing.any() and acquire and self.online:
            self._acquire(kind, coords[missing])
            local()
        return values, centers, res, refs

    def terrain(self, coords):
        values, center, res, refs = self.sample("dem", coords)
        offsets = np.array([(dx, dy) for dy in [-1, 0, 1] for dx in [-1, 0, 1]])
        neighbors = (center[:, None, :] + res[:, None, :] * offsets).reshape(-1, 2)
        finite = np.isfinite(neighbors).all(axis=1)
        v = np.full(len(neighbors), np.nan)
        actual = np.full_like(neighbors, np.nan)
        nr = [None] * len(neighbors)
        if finite.any():
            sub, xy, _, r = self.sample("dem", neighbors[finite])
            v[finite] = sub
            actual[finite] = xy
            for i, ref in zip(np.where(finite)[0], r):
                nr[int(i)] = ref
        compatible = (
            np.abs(actual - neighbors) <= np.repeat(np.abs(res) * 0.01, 9, axis=0)
        ).all(axis=1)
        v[~compatible] = np.nan
        v = v.reshape(-1, 9)
        support = np.isfinite(v).sum(axis=1)
        rms = np.sqrt(
            np.nanmean(np.delete(v - values[:, None], 4, axis=1) ** 2, axis=1)
        )
        rms[support != 9] = np.nan
        return values, rms, support, refs, [nr[i : i + 9] for i in range(0, len(nr), 9)]

    def close(self):
        for ds in self.handles.values():
            ds.close()
        self.env.__exit__(None, None, None)
