# PR #2 review-fix validation

This round fixes the reviewed `0c6887727fe0d5a962c9848ff76fae6773c2e11e` implementation. The original release results in `test_results.md` remain historical results, not new executions.

## Behavior changes

- Production CI remains Python 3.10, `uv==0.12.21`, frozen production dependencies, unittest discovery and the OpenAPI diff check. Pytest/geospatial modules moved to `tests/research/` without an `__init__.py`, so production unittest discovery does not import them. The separate Python 3.12 research job installs the committed research requirements and executes pytest with explicit skip reporting.
- Current entity, search, route, evidence, road-context and place/access queries apply connection-local validity/dependency views, including against the original release database. Inferred/unresolved flags widen geographic status only. Invalidations, audit exclusion, public-source validity and relationship dependencies remain mandatory. Stale cached comparisons abstain instead of returning invalidated dependencies. Raw inventory counts and the unresolved/history queue are explicitly labeled.
- `build --retry-missing` retries absent native inputs and newly available cached evidence, including partial inputs behind a non-null score. Completed observations, supported tunnels, degenerate tangents and covered native NoData/conflicts are terminal for acquisition. Previous samples and retry attempts remain traceable; changed sampling/version or contradictory evidence cannot replace the checkpoint. Pixel receipts, cells, scores and cached-comparison invalidation commit atomically. A complete build regenerates comparisons with unchanged scoring weights.

See [the release guide](../../docs/phase412_integrated_release.md) for working-derivative and online retry commands. No published dataset was rebuilt, no live acquisition was needed for these fixes, and no production API or external memory/reminder implementation changed. Acquisition behavior is tested using real raster fixtures with a controlled acquisition boundary; those fixtures are not delivered geographic observations.

## Validation scope

Clean-checkout tests use the exact staged Git blobs with LF line endings and committed `pyproject.toml`, `uv.lock` and research requirements. The production environment contains neither pytest nor numpy/Shapely. Research uses a separate Python 3.12 virtual environment without system-site-packages. Official PyPI wheel files were cached locally to avoid slow repeated downloads; large wheel SHA-256 values were verified against PyPI metadata. The broader historical suite additionally requires `osmium==4.3.2`, installed only in its separate dependency environment.

The initial temporary Linux checkout used Windows-converted CRLF files, causing a byte-level OpenAPI diff despite identical schemas. This was a validation-harness error, corrected by exporting exact Git blobs. The committed OpenAPI file and exporter are unchanged. The complete production job was rerun after that correction, rather than substituting an isolated retry. An initial broader attempt stopped at collection with two missing-osmium errors and one dependency skip; the subsequent broader run includes that dependency and reruns the full selected suite.

Actual-input checks use the existing private integrated dataset through read-only SQLite connections. All 22 stored baseline comparison queries remain usable. The integrated and nationwide raster database SHA-256 checksums still match the transfer manifest. Unrelated working edits to `README.md`, `pyproject.toml` and `uv.lock` were preserved and excluded from the commit. Public staged changes were checked for secrets, local dependencies and large/private artifacts.

Final executed results are recorded in `review_fixes_results.json`. GitHub Actions status is reported separately after pushing; local execution is not a claim that hosted checks passed. Historical timing failures in the original release report remain recorded. No timing assertion was relaxed and no tests were removed.

| Final run | Actual result |
|---|---|
| Exact Git checkout, complete production CI commands, CPython 3.10.21 / uv 0.12.21 | **23 unittest tests passed**, 8.481 s; frozen install, lock check, optional-dependency absence assertions and OpenAPI diff check passed |
| Clean research dependency closure, CPython 3.12.15, `pytest tests/research -q -rs` | **54 passed, 5 skipped**, 0.57 s; `pip check` passed |
| Actual existing private inputs, CPython 3.12.10, all `tests/research` | **59 passed**, 16.23 s; four actual-input checks and protected acquisition-manifest check executed |
| Full broader suite on original read-only repository mount, disposable seeded DB, CPython 3.12.15; `pytest tests -q -rs --ignore=tests/test_mapillary_visual_codex.py` | **359 passed, 1 failed, 4 skipped**, 417 subtests passed, 292.20 s |

The broader-suite failure is `tests/test_corridors_e2e.py::TestHappyPathFreezeSpecV3E2E::test_ticket_08_sub350ms_parallel_goong_clock_propagation_and_null_points`: **0.41340238799966755 s**, required **<0.34 s**. The full-suite failure remains a limitation; the separate clean production job is not a substitute for that outcome, and no isolated timing retry was run. The four broader skips require the explicitly configured private integrated input, already exercised in the separate actual-data run. Clean research has those four skips plus the unbundled nationwide acquisition-manifest check. Mapillary is explicitly outside this release scope.

The earlier copied-checkout full run returned **355 passed, 5 failed, 4 skipped**, 424 subtests passed in 176.91 s. All five failures were manifest path-escape guards encountering the harness's symlinked source directories. The subsequent run used the original repository mount and reran the entire suite; the guards and tests were not weakened or removed. Both outcomes are retained in the machine-readable results.
