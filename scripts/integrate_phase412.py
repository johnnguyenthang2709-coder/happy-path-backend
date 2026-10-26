"""Build and query the explicit opt-in Phase 4.12 integrated research release."""

import argparse
import json
from pathlib import Path

from happy_path.pretrip.integrated import IntegratedBaseline, build


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=[
            "build",
            "inventory",
            "search",
            "route",
            "compare",
            "places",
            "evidence",
            "coverage",
            "gaps",
            "entity",
        ],
    )
    parser.add_argument(
        "--enable-research",
        action="store_true",
        help="Required for every query or build; never enables production APIs",
    )
    parser.add_argument(
        "--phase", type=Path, default=root / "exports/route_intelligence/phase4_12"
    )
    parser.add_argument(
        "--raster",
        type=Path,
        default=root / "exports/route_intelligence/raster_scenic_nationwide",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "exports/route_intelligence/phase4_12_integrated",
    )
    parser.add_argument(
        "--osm",
        type=Path,
        default=root / "exports/road_experience/enriched/nationwide/base_roads.parquet",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow official native raster windows for missing inputs; no new Goong calls",
    )
    parser.add_argument("--route-id")
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="Retry missing native raster inputs on this derivative; completed cells and terminal exclusions stay unchanged",
    )
    parser.add_argument("--entity-id")
    parser.add_argument("--case-id")
    parser.add_argument("--text", default="")
    parser.add_argument("--include-inferred", action="store_true")
    parser.add_argument("--include-unresolved", action="store_true")
    args = parser.parse_args()
    if not args.enable_research:
        result = {"status": "disabled", "records": []}
    elif args.operation == "build":
        result = build(
            root,
            args.output,
            args.phase,
            args.raster,
            online=args.online,
            osm=args.osm,
            retry_missing=args.retry_missing,
        )
    else:
        result = IntegratedBaseline(args.output, enabled=True).query(
            args.operation,
            route_id=args.route_id,
            case_id=args.case_id,
            text=args.text,
            include_inferred=args.include_inferred,
            include_unresolved=args.include_unresolved,
            entity_id=args.entity_id,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
