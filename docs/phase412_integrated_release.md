# Integrated Phase 4.12 research baseline

This release adds an explicitly enabled research CLI and SQLite query layer. The production app, solver, trip sessions, existing API defaults and numerical scoring policy are unchanged. The separate AI memory/reminder repository is neither modified nor recreated. Mapillary and image scoring are outside this release.

The local build connects public-source evidence and its validity decisions, captured Goong geometries, native-raster Scenic features, nearby places, and inherited provider access/detour evidence. Scenic, documented significance and provider travel costs stay separate. There is no new recommendation model, forced profile winner or automatic stop scheduling.

## Install in a separate research environment

Use CPython 3.12. Linux is required for the existing backend's `fcntl`-based full runtime tests. The research queries themselves also run on Windows.

```sh
python3.12 -m venv .research-venv
.research-venv/bin/python -m pip install . -r requirements/research-phase412.txt
```

On Windows use `.research-venv/Scripts/python.exe`. This uses the committed base project and a separate research requirements file; it does not require unrelated local edits to `pyproject.toml` or `uv.lock`. The wheel includes the integration SQL schema through `MANIFEST.in`.

## Inputs, licensing and portability

**Bundled in Git:** implementation, SQL schema, tests, research dependency pins, documentation and compact author-generated release metadata. No Goong coordinates, provider response bodies, full article text, private community posts, credentials, SQLite databases, Parquet acquisition collections, raster blocks or images are included in the public release artifacts.

**Required externally to reproduce the full existing snapshot:**

| Input | Relative location in the original research workspace | Role |
|---|---|---|
| Phase 4.12 database | `exports/route_intelligence/phase4_12/national_mapping.sqlite` | Original/current identities, sources, scoped claims, invalidations, captures and access comparisons |
| Raster database | `exports/route_intelligence/raster_scenic_nationwide/scenic.sqlite` | Existing exact registered geometry/cell/score evidence |
| Raster register and policy | `route_register.json`, `config.json` in that raster directory | Captured routes, request snapshots and unchanged Phase 4 policy |
| OSM matching inventory | `exports/road_experience/enriched/nationwide/base_roads.parquet` | Direction-aware matches and supported tunnel masks for newly sampled geometry |
| Native blocks | `raster_cache/` under the raster directory and the integrated derivative | Optional cache for bounded offline native extraction |

The original snapshots contain provider-restricted records and source content without blanket redistribution permission. Obtain the private handoff directly from the dataset holder through an authorized private transfer. There is no public download URL for these snapshots and **the GitHub checkout alone cannot recreate them offline**. A compact input manifest records sizes and checksums, without publishing their contents.

Create a smaller **local private** handoff from the original workspace:

```sh
python scripts/package_phase412_inputs.py \
  --phase exports/route_intelligence/phase4_12 \
  --raster exports/route_intelligence/raster_scenic_nationwide \
  --osm exports/road_experience/enriched/nationwide/base_roads.parquet \
  --derived-cache exports/route_intelligence/phase4_12_integrated/raster_cache \
  --output /path/to/new-private-handoff --include-rasters
```

It retains the full source/validity database and the 32 priority/reference raster geometries, rather than the 15 GB nationwide raster database. With `--include-rasters`, it also includes cached native blocks and checksum-validated 2021 legacy crop families used by this integration. Never upload this handoff to the public PR. Original input hashes and per-file checksums remain in `PRIVATE_INPUT_MANIFEST.json`. Other historical acquisition assets remain external. The validated private bundle reproduces the integrated scores offline; the public Git checkout does not contain that bundle.

Public-source acquisition references: [Geofabrik Vietnam](https://download.geofabrik.de/asia/vietnam.html), [ESA WorldCover data access](https://esa-worldcover.org/en/data-access), [Copernicus public DSM](https://registry.opendata.aws/copernicus-dem/). Preserve the recorded 2026-10-22 OSM snapshot, WorldCover 2021 v200 and Copernicus public DSM 2021 product. Newer snapshots must be separate releases. The OSM inventory's ten major road classes and access exclusions are documented in the raster manifest; it does not contain every Vietnamese street. Re-acquiring Goong responses needs your own permitted credentials and may return different routes/ETAs. Public pages can change or disappear. Neither a current download nor new provider response is promised to reproduce a historical snapshot byte-for-byte.

## Build and query

After installing the required private snapshots at their original relative locations:

```sh
python scripts/integrate_phase412.py build --enable-research
```

Or build from the portable handoff, with explicit paths:

```sh
python scripts/integrate_phase412.py build --enable-research \
  --phase /private-handoff/phase --raster /private-handoff/raster \
  --osm /private-handoff/base_roads.parquet --output /new-integrated-release
```

Add `--online` only when missing native raster windows should be acquired from the official public products. This builder performs no new Goong calls. Identical canonical geometry, CRS and extraction version reuse existing cell-level results. All other captured geometry is sampled through the native extractor; proximity to a parallel road never licenses copying its score. Builds commit score checkpoints and may resume. Changed inputs require a new output directory. Existing source releases are read-only.

An existing score row can contain partial inputs even when its conditional score is non-null. To retry those gaps, use `--retry-missing` explicitly on a **working derivative**, keeping the same declared inputs, OSM snapshot, policy and extraction version:

```sh
# First create your working derivative using the private snapshots.
python scripts/integrate_phase412.py build --enable-research \
  --phase /private-handoff/phase --raster /private-handoff/raster \
  --osm /private-handoff/base_roads.parquet --output /working-derivative
# Resume after enabling permitted official raster access (or omit --online
# if additional native blocks have already been added to the cache).
python scripts/integrate_phase412.py build --enable-research --online --retry-missing \
  --phase /private-handoff/phase --raster /private-handoff/raster \
  --osm /private-handoff/base_roads.parquet --output /working-derivative
```

Do not use a published release as the writable output. Without `--retry-missing`, existing scores keep their previous resume behavior. Explicit retry selects only cells with missing native inputs or newly available cached evidence; fully observed cells, supported tunnel exclusions, degenerate tangents and covered native NoData/conflicting pixels are not repeatedly extracted. Covered NoData remains unknown, not zero landscape value. Successful partial retries preserve earlier known samples and recompute scores and comparisons using unchanged weights. Previous evidence and attempts remain in `score_retry_history`; failed retries preserve the prior consistent score/cell checkpoint. Changed geometry, extraction version or contradictory samples are rejected. A score change invalidates its cached comparisons in the same transaction, and a completed build regenerates comparisons; an interrupted build cannot serve those stale comparisons.

All queries require `--enable-research`; omitting it returns `disabled` without touching files. The output directory defaults to `exports/route_intelligence/phase4_12_integrated`; pass `--output` for a portable location.

```sh
python scripts/integrate_phase412.py inventory --enable-research
python scripts/integrate_phase412.py route --enable-research --route-id route:5dc7cfdc7b7e718e5df5
python scripts/integrate_phase412.py compare --enable-research --case-id hcm_dalat
python scripts/integrate_phase412.py evidence --enable-research --route-id route:bfa45056fe2ad4d8e1fa
python scripts/integrate_phase412.py places --enable-research --route-id route:5dc7cfdc7b7e718e5df5
python scripts/integrate_phase412.py coverage --enable-research --route-id route:5dc7cfdc7b7e718e5df5
python scripts/integrate_phase412.py search --enable-research --text "Hải Vân"
python scripts/integrate_phase412.py search --enable-research --text "Vĩnh Hy" --include-inferred --include-unresolved
python scripts/integrate_phase412.py gaps --enable-research
```

Nationwide catalog records remain searchable independently of baseline routes. Inferred/unresolved discoveries are explicit opt-ins. They do not become provider-routed access, verified entrances or operational stop eligibility. Inherited access and compatible detours apply only to their original route ID/snapshot and coordinate evidence; proximity-generated opportunities never inherit a detour from a different candidate.

`--include-inferred` and `--include-unresolved` widen only geographic-status filters. Entity, route, claim/source, road-context and relationship invalidations remain mandatory, including when inspecting older databases. Connection-local views enforce these rules without migrating or writing the input snapshot. Cached comparisons whose current dependencies no longer qualify return `stale_comparison`, with no usable cached comparison, until a derivative is rebuilt. Inventory counts explicitly include retained history; `gaps` is labeled an unresolved/history queue and does not confer current eligibility. No flag exposes quarantined or superseded records as current usable data.

## Geographic scope and score explanations

Route context uses bounded OSM components and preserves source-explicit capture links. Spatial overlap is a proximity relationship, not field-verified carriageway identity. General corridor claims, localized historical context and nearby place context remain distinct. Pending localized extents do not acquire invented route intervals. Historical La Ngà markers are not modern geocoding or whole-QL20 events. Hải Vân Quan monument claims remain nearby monument context, not pass-wide significance. The original Bảo Lộc audit route is excluded from default usable candidates.

Claims from the same canonical source family are grouped; repeated captures are not independent corroboration. Private/local research-file references remain traceable but cannot masquerade as public independent evidence. Default views enforce invalidations, current relationships and inferred/unresolved status.

Scenic retains weights `.35 tree / .25 water / .10 cropland / .30 ruggedness`, RMS saturation at 10 m, support/length weighting, and the 80% evidence gate. The six samples are discrete roadside fractions, not buffer area. Supported tunnel surroundings retain raw observations but are excluded from scoring. Missing inputs retain conditional values and bounds. Bounds do not cover raster/source error, sampling error, unresolved road identity, actual visibility, current surface conditions, safety or uncalibrated user preferences.

Same-OD comparisons show provider distance and captured ETA, Scenic components/contributions, shared versus differentiating landscapes, public claims and place opportunities. Shared geometry uses exact line topology as a conservative lower bound; parallel carriageways are not collapsed. A partially shared cell apportions its existing observation by actual overlap length, and shared plus differentiating support equals route length. Whole-section score averages are never substituted. Express uses the existing relative provider cost function only within the same OD case. Documented Iconic evidence remains incomplete; ordinary identity records are not significance evidence and no three distinct winners are forced.

## Tests and release review

The supported full suite is `python -m pytest tests -q` on a Linux disposable seeded database. The repository's original runtime tests are included. The public checkout intentionally omits historical, unrelated phase test modules that require unreleased acquisition trees. The final release test report distinguishes the full workspace suite (excluding the out-of-scope Mapillary module), the clean public checkout suite, and actual-input integration tests. Skips without private inputs are explicit, not passes.

Production CI remains Python 3.10 with `uv==0.12.21`, `make setup`, frozen production dependencies, `uv lock --check`, `make test` (unittest discovery), and the OpenAPI diff check. Research pytest modules live in `tests/research/`, deliberately without `__init__.py`, so production unittest discovery does not import optional geospatial/pytest dependencies. A separate Python 3.12 CI job installs the base project plus `requirements/research-phase412.txt` and runs `python -m pytest tests/research -q -rs`; private-data and acquisition-manifest skips appear explicitly in its log. The production runtime isolation unittest remains in the production test tree.

```sh
export HAPPY_PATH_DB_PATH=/tmp/happy-path-release-tests.db
python scripts/seed_db.py --db-path "$HAPPY_PATH_DB_PATH" --if-missing
python -m pytest tests -q
export HAPPY_PATH_INTEGRATED_TEST_DATA=/path/to/integrated-release
python -m pytest tests/research/test_integrated_release.py -q
```

See `releases/phase4_12/` for the actual results, input inventory and compact release metadata. The local full dataset contains record-level corrections, unresolved claims, raw/canonical source IDs, original source payloads, score/cell provenance and comparison exports. It is a research baseline, not an import into production.
