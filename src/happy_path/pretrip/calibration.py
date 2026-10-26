"""Opt-in Phase 4.1 validation; no runtime registration or geographic imputation.

Preference stress scenarios are sensitivity experiments, never confidence levels.
The original Phase 4 policy remains available unchanged for reproducibility.
"""

import itertools
import json
import math

from happy_path.pretrip.scoring import express_scores, iconic_score, scenic_score

COMPONENTS = ("tree_cover", "water", "cropland", "ruggedness")


def validated_scenic(cells, policy):
    if set(policy["scenic_weights"]) != set(COMPONENTS):
        raise ValueError("Exactly four documented scenic components are required")
    if (
        not math.isfinite(policy["ruggedness_saturation_m"])
        or policy["ruggedness_saturation_m"] <= 0
    ):
        raise ValueError("Ruggedness saturation must be positive")
    if not 0 < policy["minimum_landscape_coverage"] <= 1:
        raise ValueError("Coverage threshold must be in (0,1]")
    for cell in cells:
        if not math.isfinite(cell["cell_length_m"]) or cell["cell_length_m"] <= 0:
            raise ValueError("Cell length must be finite and positive")
        if (
            not math.isfinite(cell["roadside_sample_coverage"])
            or not 0 <= cell["roadside_sample_coverage"] <= 1
        ):
            raise ValueError("Invalid sample coverage")
        ruggedness = cell["dsm_ruggedness_rms_m"]
        if ruggedness is not None and (not math.isfinite(ruggedness) or ruggedness < 0):
            raise ValueError("Invalid measured ruggedness")
        landscape = [
            cell["roadside_" + key + "_fraction"]
            for key in ("tree", "water", "cropland")
        ]
        if any(v is None for v in landscape) != all(v is None for v in landscape):
            raise ValueError("Landscape classes must share their observation support")
        if (landscape[0] is None) != (cell["roadside_sample_coverage"] == 0):
            raise ValueError("Missing landscape must have zero support")
        if landscape[0] is not None and sum(landscape) > 1 + 1e-9:
            raise ValueError("Exclusive landscape fractions exceed one")
    return scenic_score(cells, policy)


def validated_express(routes, policy):
    """Avoid eager fallback evaluation when only the new eligibility key exists."""
    safe = [
        dict(r, route_ranking_eligible=r.get("route_ranking_eligible", False))
        for r in routes
    ]
    return express_scores(safe, policy)


def validated_iconic(known_road_m, total_m, road_entities, nearby_entities, policy):
    saturation = policy["iconic_entity_saturation"]
    if (
        not all(math.isfinite(v) for v in (known_road_m, total_m, saturation))
        or saturation <= 0
    ):
        raise ValueError("Invalid Iconic normalization or exposure")
    result = iconic_score(known_road_m, total_m, road_entities, nearby_entities, policy)
    # A policy completion interval is not a defensible bound on total Iconic
    # value when there is no measured catalog completeness or preference model.
    result["possible_total_value_bounds"] = None
    result["bounds_reason"] = (
        "Incomplete research cannot calibrate total iconic value; positive evidence index only"
    )
    return result


def isolate_tunnels(cells):
    """Source-supported tunnels: surface surroundings remain factual, score unknown.

    No inference is made about tunnel extent in unmatched or ambiguous cells.
    """
    result = []
    for original in cells:
        cell = dict(original)
        tags = json.loads(cell.get("source_road_tags_json") or "{}")
        if cell.get(
            "road_match_status"
        ) == "supported_geometry_not_field_verified" and tags.get("tunnel") in (
            "yes",
            "building_passage",
        ):
            cell.update(
                roadside_tree_fraction=None,
                roadside_water_fraction=None,
                roadside_cropland_fraction=None,
                roadside_sample_coverage=0,
                dsm_ruggedness_rms_m=None,
                scenic_surface_context_unavailable=True,
            )
        result.append(cell)
    return result


def compare_scenic(a, b):
    """A score or ordering cannot bypass its evidence eligibility check."""
    if not all(
        r.get("provider_comparison_eligible", False) and r["scenic"]["ranking_eligible"]
        for r in (a, b)
    ):
        return "withheld_insufficient_evidence_or_invalid_candidate"
    x, y = (
        a["scenic"]["missing_evidence_bounds"],
        b["scenic"]["missing_evidence_bounds"],
    )
    if x[0] > y[1]:
        return "a_above_b_under_missing_input_bounds_only"
    if y[0] > x[1]:
        return "b_above_a_under_missing_input_bounds_only"
    return "unresolved_under_missing_input_bounds"


def pareto(a, b, scenic=False, robust=False):
    """Dominance in declared measured dimensions; incomplete Iconic is excluded."""
    if not all(r.get("provider_comparison_eligible", False) for r in (a, b)):
        return None
    advantages = [
        b["provider_distance_m"] - a["provider_distance_m"],
        b["provider_duration_s"] - a["provider_duration_s"],
    ]
    if scenic:
        if not all(r["scenic"]["ranking_eligible"] for r in (a, b)):
            return None
        advantages.append(
            a["scenic"]["missing_evidence_bounds"][0]
            - b["scenic"]["missing_evidence_bounds"][1]
            if robust
            else a["scenic"]["score"] - b["scenic"]["score"]
        )
    return all(v >= -1e-9 for v in advantages) and any(v > 1e-9 for v in advantages)


def elevation_support(cells):
    """Distance between native sample midpoints, never full terminal-cell length."""
    return sum(
        ((b["start_m"] + b["end_m"]) - (a["start_m"] + a["end_m"])) / 2
        for a, b in itertools.pairwise(cells)
        if a["dsm_elevation_m"] is not None and b["dsm_elevation_m"] is not None
    )


def policy_scenarios(policy):
    """Deterministic declared preference grids, no winner-fitted weights."""
    for spread in (0.25, 0.5):
        for multipliers in itertools.product((1 - spread, 1, 1 + spread), repeat=4):
            raw = {
                k: policy["scenic_weights"][k] * m
                for k, m in zip(COMPONENTS, multipliers)
            }
            total = sum(raw.values())
            for saturation in (5, 10, 20):
                yield (
                    dict(
                        policy,
                        scenic_weights={k: v / total for k, v in raw.items()},
                        ruggedness_saturation_m=saturation,
                    ),
                    {
                        "family": f"weights_plus_minus_{int(spread * 100)}pct",
                        "multipliers": list(multipliers),
                        "saturation_m": saturation,
                    },
                )
    for omitted in COMPONENTS:
        raw = dict(policy["scenic_weights"], **{omitted: 0})
        total = sum(raw.values())
        yield (
            dict(policy, scenic_weights={k: v / total for k, v in raw.items()}),
            {"family": "leave_component_out", "omitted": omitted},
        )


def ranking_summary(experiments, route_ids):
    summary = {
        r: {
            "first_count": 0,
            "unique_first_count": 0,
            "eligible_count": 0,
            "min_rank": None,
            "max_rank": None,
        }
        for r in route_ids
    }
    for scores in experiments:
        known = {r: v for r, v in scores.items() if v is not None}
        for r, value in known.items():
            rank = 1 + sum(v > value + 1e-9 for v in known.values())
            row = summary[r]
            row["first_count"] += int(rank == 1)
            row["unique_first_count"] += int(
                rank == 1 and sum(abs(v - value) <= 1e-9 for v in known.values()) == 1
            )
            row["eligible_count"] += 1
            row["min_rank"] = min(row["min_rank"] or rank, rank)
            row["max_rank"] = max(row["max_rank"] or rank, rank)
    return {
        "experiments": len(experiments),
        "routes": summary,
        "interpretation": "Counts describe this declared stress grid, not probabilities or preference ground truth",
    }


def profile_decision(routes, sensitivity, profile):
    eligible = [r for r in routes if r.get("provider_comparison_eligible", False)]
    if profile == "iconic":
        return {
            "comparison_leader": None,
            "navigation_recommendation": None,
            "status": "withheld_incomplete_research_catalog",
        }
    candidates = [
        r
        for r in eligible
        if r[profile]["ranking_eligible"] and r[profile]["score"] is not None
    ]
    if not candidates:
        return {
            "comparison_leader": None,
            "navigation_recommendation": None,
            "status": "withheld_insufficient_evidence",
        }
    best = max(candidates, key=lambda r: r[profile]["score"])
    # A sole candidate is not differentiation. All perturbation scenarios must
    # include all candidates, otherwise missing support cannot imply stability.
    stable = (
        len(candidates) > 1
        and sensitivity["experiments"] > 0
        and all(
            sensitivity["routes"][r["route_id"]]["eligible_count"]
            == sensitivity["experiments"]
            for r in eligible
        )
        and sensitivity["routes"][best["route_id"]]["unique_first_count"]
        == sensitivity["experiments"]
    )
    separated = profile == "express" or all(
        compare_scenic(best, r) == "a_above_b_under_missing_input_bounds_only"
        for r in candidates
        if r is not best
    )
    return {
        "comparison_leader": best["route_id"],
        "navigation_recommendation": None,
        "status": "stable_comparison_leader_only"
        if stable and separated
        else "provisional_leader_not_robust",
        "stable_on_declared_grid": stable,
        "separated_under_missing_inputs": separated,
        "navigation_reason": "Research comparisons are not navigation certification; alignment/restrictions require independent resolution",
    }
