"""Bounded public-source acquisition with secret-free, hash-validated local caches."""

import hashlib
import json
import math
import os
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import numpy as np
import rasterio
from pyproj import Geod
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, mapping


def now():
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True
    ).encode("utf-8")


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


class SourceCache:
    def __init__(self, directory, max_days=7, offline=False):
        self.directory, self.max_days, self.offline = Path(directory), max_days, offline
        self.receipts = []
        self.client = httpx.Client(
            timeout=85,
            follow_redirects=False,
            headers={
                "User-Agent": "HappyPathRoadResearch/1.0 (bounded acquisition; no OSM edits)"
            },
        )

    def close(self):
        self.client.close()

    def request(
        self,
        source,
        method,
        url,
        *,
        public_params=None,
        secret_params=None,
        data=None,
        license=None,
    ):
        if urlsplit(url).scheme != "https":
            raise ValueError("Acquisition requires HTTPS")
        host = urlsplit(url).hostname
        if secret_params and host not in {"rsapi.goong.io", "graph.mapillary.com"}:
            raise ValueError("Refusing credentials to a non-provider host")
        descriptor = dict(
            source=source, method=method, url=url, params=public_params or {}, data=data
        )
        key = hashlib.sha256(json_bytes(descriptor)).hexdigest()
        path, meta = self.directory / f"{key}.json", self.directory / f"{key}.meta.json"
        receipt = dict(**descriptor, cache_key=key, license=license, attempted_at=now())
        if path.exists() and meta.exists():
            metadata = json.loads(meta.read_text(encoding="utf-8"))
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != metadata["sha256"]:
                raise ValueError(f"Corrupt {source} cache: {key}")
            age_days = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(metadata["retrieved_at"])
            ).total_seconds() / 86400
            if self.offline or age_days <= self.max_days:
                receipt.update(metadata, cache_hit=True, cache_age_days=age_days)
                self.receipts.append(receipt)
                return json.loads(content), receipt
        if self.offline:
            receipt.update(success=False, reason="OFFLINE_CACHE_MISS")
            self.receipts.append(receipt)
            raise RuntimeError(f"{source}: OFFLINE_CACHE_MISS")
        try:
            for attempt in range(3):
                response = self.client.request(
                    method,
                    url,
                    params={**(public_params or {}), **(secret_params or {})},
                    data=data,
                )
                if response.status_code not in (429, 502, 503, 504) or attempt == 2:
                    break
                retry = response.headers.get("Retry-After", "")
                time.sleep(
                    min(60, int(retry)) if retry.isdigit() else 5 * (attempt + 1)
                )
            receipt.update(
                http_status=response.status_code, retrieved_at=now(), cache_hit=False
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP_{response.status_code}")
            payload = response.json()
            if isinstance(payload, dict) and payload.get("remark"):
                raise RuntimeError("PROVIDER_PARTIAL_RESPONSE_REMARK")
            content = json_bytes(payload)
            # Provider error URLs containing credentials must never reach stored exception strings.
            for value in (secret_params or {}).values():
                if value and str(value).encode() in content:
                    raise ValueError(
                        "Provider echoed a credential; response not retained"
                    )
            receipt.update(success=True, sha256=hashlib.sha256(content).hexdigest())
            atomic(path, content)
            atomic(
                meta,
                json_bytes(
                    {
                        k: receipt[k]
                        for k in ("success", "sha256", "retrieved_at", "http_status")
                    }
                ),
            )
            self.receipts.append(receipt)
            return payload, receipt
        except (httpx.HTTPError, ValueError, RuntimeError) as error:
            receipt.update(
                success=False,
                reason=str(error)
                if isinstance(error, RuntimeError)
                else type(error).__name__,
            )
            self.receipts.append(receipt)
            raise RuntimeError(f"{source}: {receipt['reason']}") from None


ATTRIBUTIONS = {
    "worldcover": {
        "product": "ESA WorldCover 2021 v200, 10m",
        "year": 2021,
        "license": "CC-BY-4.0",
        "attribution": "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium",
        "documentation": "https://esa-worldcover.org/en/data-access",
    },
    "dem": {
        "product": "Copernicus DEM GLO-30 Public DSM, 2021 release",
        "year": 2021,
        "license": "Copernicus DEM public license; see source documentation",
        "attribution": "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved",
        "vertical_reference": "EGM2008 geoid, EPSG:3855; surface heights in meters",
        "documentation": "https://registry.opendata.aws/copernicus-dem/",
    },
}


def raster_url(kind, lat, lng):
    if kind == "worldcover":
        y, x = math.floor(lat / 3) * 3, math.floor(lng / 3) * 3
        tile = (
            f"{'N' if y >= 0 else 'S'}{abs(y):02d}{'E' if x >= 0 else 'W'}{abs(x):03d}"
        )
        return f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    y, x = math.floor(lat), math.floor(lng)
    tile = f"Copernicus_DSM_COG_10_{'N' if y >= 0 else 'S'}{abs(y):02d}_00_{'E' if x >= 0 else 'W'}{abs(x):03d}_00_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif"


def native_window(bounds, dataset):
    """Disjoint native crops selected by pixel centers, including half-pixel grids."""
    window = from_bounds(*bounds, dataset.transform)
    col = math.ceil(window.col_off - 0.5 - 1e-6)
    row = math.ceil(window.row_off - 0.5 - 1e-6)
    right = math.ceil(window.col_off + window.width - 0.5 - 1e-6)
    bottom = math.ceil(window.row_off + window.height - 0.5 - 1e-6)
    return Window(col, row, right - col, bottom - row).intersection(
        Window(0, 0, dataset.width, dataset.height)
    )


class RasterPatches:
    def __init__(self, directory, kind, offline=False):
        self.directory, self.kind, self.offline = Path(directory), kind, offline
        self.patches, self.receipts, self.handles = [], [], ExitStack()

    def close(self):
        self.handles.close()

    def acquire(self, area_wgs, grid_deg, limit):
        xmin, ymin, xmax, ymax = area_wgs.bounds
        cells = []
        for y in range(math.floor(ymin / grid_deg), math.ceil(ymax / grid_deg)):
            for x in range(math.floor(xmin / grid_deg), math.ceil(xmax / grid_deg)):
                bounds = (
                    x * grid_deg,
                    y * grid_deg,
                    (x + 1) * grid_deg,
                    (y + 1) * grid_deg,
                )
                if area_wgs.intersects(box(*bounds)):
                    cells.append(bounds)
        if len(cells) > limit:
            raise ValueError(
                f"{self.kind}: {len(cells)} cells exceeds configured {limit}; increase limit explicitly"
            )
        remotes, failures = {}, {}
        settings = dict(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
            GDAL_HTTP_TIMEOUT="25",
            GDAL_HTTP_MAX_RETRY="1",
            GDAL_HTTP_RETRY_DELAY="2",
            GDAL_CACHEMAX=256 * 1024 * 1024,
        )
        with rasterio.Env(**settings), ExitStack() as remote_handles:
            for i, bounds in enumerate(cells):
                url = raster_url(
                    self.kind, (bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2
                )
                identity = hashlib.sha256(
                    json_bytes([url, bounds, "native_pixel_center_partition_v2"])
                ).hexdigest()
                path = self.directory / f"{identity}.tif"
                meta = path.with_suffix(".meta.json")
                receipt = dict(
                    source=self.kind,
                    url=url,
                    cache_key=identity,
                    crop_method="native_pixel_center_partition_v2",
                    requested_bounds=list(bounds),
                    **ATTRIBUTIONS[self.kind],
                )
                try:
                    if path.exists() and meta.exists():
                        old = json.loads(meta.read_text(encoding="utf-8"))
                        if (
                            hashlib.sha256(path.read_bytes()).hexdigest()
                            != old["sha256"]
                        ):
                            raise ValueError("Raster cache integrity failure")
                        receipt.update(old, cache_hit=True)
                    else:
                        if self.offline:
                            raise RuntimeError("OFFLINE_RASTER_CACHE_MISS")
                        if url in failures:
                            raise RuntimeError(failures[url])
                        if url not in remotes:
                            try:
                                remotes[url] = remote_handles.enter_context(
                                    rasterio.open(url)
                                )
                            except rasterio.errors.RasterioError as e:
                                failures[url] = type(e).__name__
                                raise
                        dataset = remotes[url]
                        if dataset.crs is None or dataset.crs.to_epsg() != 4326:
                            raise ValueError("Unexpected raster CRS")
                        win = native_window(bounds, dataset)
                        array = dataset.read(1, window=win, masked=True)
                        transform = dataset.window_transform(win)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temp = path.with_suffix(".tmp.tif")
                        profile = dict(
                            driver="GTiff",
                            width=array.shape[1],
                            height=array.shape[0],
                            count=1,
                            dtype=dataset.dtypes[0],
                            crs=dataset.crs,
                            transform=transform,
                            nodata=dataset.nodata,
                            compress="deflate",
                        )
                        with rasterio.open(temp, "w", **profile) as output:
                            output.write(
                                array.filled(
                                    dataset.nodata if dataset.nodata is not None else 0
                                ),
                                1,
                            )
                        os.replace(temp, path)
                        receipt.update(
                            success=True,
                            retrieved_at=now(),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            cache_hit=False,
                            crs="EPSG:4326",
                            shape=list(array.shape),
                            nodata=dataset.nodata,
                            pixel_size_deg=list(dataset.res),
                        )
                        atomic(meta, json_bytes(receipt))
                    ds = self.handles.enter_context(rasterio.open(path))
                    self.patches.append((box(*ds.bounds), ds, receipt))
                except (
                    rasterio.errors.RasterioError,
                    ValueError,
                    RuntimeError,
                ) as error:
                    receipt.update(
                        success=False,
                        retrieved_at=now(),
                        reason=str(error)
                        if isinstance(error, RuntimeError)
                        else type(error).__name__,
                    )
                self.receipts.append(receipt)
                if (i + 1) % 10 == 0 or i + 1 == len(cells):
                    print(
                        json.dumps(
                            {
                                "source": self.kind,
                                "patches_done": i + 1,
                                "patches_total": len(cells),
                                "successes": len(self.patches),
                            }
                        ),
                        flush=True,
                    )

    def sample(self, lon, lat):
        for bounds, ds, _ in self.patches:
            if (
                ds.bounds.left <= lon < ds.bounds.right
                and ds.bounds.bottom < lat <= ds.bounds.top
            ):
                row, col = ds.index(lon, lat)
                if not (0 <= row < ds.height and 0 <= col < ds.width):
                    continue  # A shared right/bottom boundary belongs to the adjacent crop.
                value = next(ds.sample([(lon, lat)], masked=True))[0]
                if (
                    not np.ma.is_masked(value)
                    and np.isfinite(value)
                    and (
                        self.kind != "worldcover"
                        or value in (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
                    )
                ):
                    return float(value)
        return None

    def values_in(self, polygon_wgs):
        """Native pixel-center intersection, geodesic pixel-area weights; no area extrapolation."""
        values, weights, slope_values = [], [], []
        sources = set()
        geod = Geod(ellps="WGS84")
        for bounds, ds, receipt in self.patches:
            if not bounds.intersects(polygon_wgs):
                continue
            intersection = bounds.intersection(polygon_wgs)
            if intersection.is_empty or intersection.area <= 0:
                continue
            window = from_bounds(*intersection.bounds, ds.transform)
            window = Window(
                math.floor(window.col_off),
                math.floor(window.row_off),
                math.ceil(window.width) + 2,
                math.ceil(window.height) + 2,
            )
            try:
                window = window.intersection(Window(0, 0, ds.width, ds.height))
            except rasterio.errors.WindowError:
                continue
            data = ds.read(1, window=window, masked=True)
            affine = ds.window_transform(window)
            inside = geometry_mask(
                [mapping(polygon_wgs)],
                out_shape=data.shape,
                transform=affine,
                invert=True,
            )
            valid = inside & ~np.ma.getmaskarray(data) & np.isfinite(data.data)
            if self.kind == "worldcover":
                valid &= np.isin(
                    data.data, [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
                )
            if not valid.any():
                continue
            area_rows = []
            for r in range(data.shape[0]):
                x, y = affine * (0, r)
                area_rows.append(
                    abs(
                        geod.polygon_area_perimeter(
                            [x, x + affine.a, x + affine.a, x],
                            [y, y, y + affine.e, y + affine.e],
                        )[0]
                    )
                )
            area = np.broadcast_to(np.array(area_rows)[:, None], data.shape)
            values.append(data.data[valid].astype(float))
            weights.append(area[valid])
            sources.add(receipt["sha256"])
            if self.kind == "dem" and min(data.shape) > 1:
                # DSM surface slope proxy, not road engineering grade or bare-earth terrain.
                lat = (intersection.bounds[1] + intersection.bounds[3]) / 2
                dx = geod.inv(0, lat, abs(affine.a), lat)[2]
                dy = geod.inv(0, lat, 0, lat + abs(affine.e))[2]
                z = data.filled(np.nan).astype(float)
                gy, gx = np.gradient(z, dy, dx)
                slope = np.degrees(np.arctan(np.hypot(gx, gy)))
                slope_values.append(slope[valid & np.isfinite(slope)])
        return (
            np.concatenate(values) if values else np.array([]),
            np.concatenate(weights) if weights else np.array([]),
            np.concatenate(slope_values) if slope_values else np.array([]),
            sorted(sources),
        )
