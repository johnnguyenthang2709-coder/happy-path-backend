# Release validation

These are the original release-round results. See [PR #2 review-fix validation](review_fixes_validation.md) for the separately executed fix-round results and current test locations.

Executed during this release round, 2026-10-26. No production database was used.

| Run | Actual result |
|---|---|
| Full existing research workspace, Linux, disposable seeded DB; `pytest tests -q --ignore=tests/test_mapillary_visual_codex.py` | **326 passed, 1 failed**, 417 subtests passed, 731.82 s |
| Clean public checkout, Linux, installed wheel, disposable seeded DB; `pytest tests -q` | **42 passed, 5 skipped**, 323 subtests passed, 12.70 s |
| Final actual-input integration and raster tests; `pytest tests/test_integrated_release.py tests/test_raster_scenic.py -q` with `HAPPY_PATH_INTEGRATED_TEST_DATA` configured | **24 passed**, 12.05 s |
| Separate private-handoff offline build | 32 exact geometry reuses and 265 native geometry extractions; all **297** numerical score records match the original integration after packaging repair |
| Representative populated queries | inventory, route, compare, evidence, places, coverage, nationwide search and unresolved gaps all returned `ok` |

The full workspace failure is `TestHappyPathFreezeSpecV3E2E::test_ticket_08_sub350ms_parallel_goong_clock_propagation_and_null_points` in `tests/test_corridors_e2e.py`. Its elapsed value was 0.3819942379999972 s against a 0.34 s assertion. The timing threshold and production implementation were not changed. The clean-checkout pass is a distinct full checkout run, **not** a replacement for or an isolated retry of that failed full workspace run. Historical research tests outside the released dependency closure are not bundled. Mapillary tests were explicitly outside scope.

The five clean-checkout skips are four tests requiring the unbundled private integrated snapshot and one test requiring the unbundled nationwide acquisition manifest. The final actual-input run executes all integration tests; the original workspace also retains the nationwide acquisition manifest. Deprecation warnings remain in upstream Pydantic, raster affine operations and Shapely APIs.

## Packaging and reproduction checks

- A clean Windows research environment installed the base project and pinned research requirements. Isolated Python imported the installed package, rather than the workspace source, and executed populated queries.
- A universal wheel was built from the staged public checkout. The integration SQL schema is bundled and readable from the installed package. Linux installed that wheel in a separate environment using the already installed dependency runtime; `pip check` passed. This Linux environment reused dependency packages and is not a claim of an independently downloaded dependency closure.
- The private handoff's offline build exposed one geometry's missing legacy WorldCover crop. The packager now includes checksum-validated legacy crop families; 136 crops were added to the tested bundle. The repaired derivative was rebuilt offline and all 297 score, conditional-score, coverage, component and bound records match. No Goong acquisition or network raster requests occurred in the offline builds.
- Source databases are read-only; unrelated `README.md`, `pyproject.toml` and `uv.lock` working changes were retained. Default research-disabled queries do not touch files. Runtime isolation tests verify no route intelligence import or failure affects existing itinerary runtime.
- Targeted tests cover localized La Ngà/Hải Vân scope, quarantine exclusion, original audit isolation, access-relationship preservation, shared-length accounting, unknown bounds, tunnel masks, exact reverse reuse and parallel-road separation.

Public release reproduction requires the inputs described in [the release guide](../../docs/phase412_integrated_release.md). No source payloads, provider geometries or individual provider ETA/distance records are in this Git release. Complete comparisons and their numerical explanations are in the local/private integrated exports.
