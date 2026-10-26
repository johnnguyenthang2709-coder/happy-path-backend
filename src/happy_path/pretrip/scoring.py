"""Explicit preference indices and missing-evidence bounds, not scenic ground truth."""

import math


def scenic_score(cells, policy):
    total = sum(c["cell_length_m"] for c in cells)
    weights = policy["scenic_weights"]
    if (
        total <= 0
        or any(not math.isfinite(v) or v < 0 for v in weights.values())
        or abs(sum(weights.values()) - 1) > 1e-9
    ):
        raise ValueError("Invalid scoring weights or route length")
    numerators = dict.fromkeys(weights, 0.0)
    supports = dict.fromkeys(weights, 0.0)
    for cell in cells:
        values = dict(
            tree_cover=cell["roadside_tree_fraction"],
            water=cell["roadside_water_fraction"],
            cropland=cell["roadside_cropland_fraction"],
            ruggedness=cell["dsm_ruggedness_rms_m"],
        )
        for key, value in values.items():
            if value is None:
                continue
            if not math.isfinite(value):
                raise ValueError("Nonfinite measured scenic input")
            if key == "ruggedness":
                normalized = min(1, max(0, value / policy["ruggedness_saturation_m"]))
                support = cell["cell_length_m"]
            else:
                if (
                    not 0 <= value <= 1
                    or not 0 <= cell["roadside_sample_coverage"] <= 1
                ):
                    raise ValueError("Invalid sampled landscape fraction")
                normalized = value
                support = cell["cell_length_m"] * cell["roadside_sample_coverage"]
            numerators[key] += support * normalized
            supports[key] += support
    known_value = sum(weights[k] * numerators[k] for k in weights)
    known_support = sum(weights[k] * supports[k] for k in weights)
    coverage = known_support / total
    conditional = 100 * known_value / known_support if known_support > 0 else None
    lower = 100 * known_value / total
    upper = min(100.0, lower + 100 * (1 - coverage))
    eligible = coverage >= policy["minimum_landscape_coverage"]
    return dict(
        score=conditional if eligible else None,
        observed_score=conditional,
        missing_evidence_bounds=[lower, upper],
        evidence_weight_coverage=coverage,
        ranking_eligible=eligible,
        status="preliminary_nearby_landscape_preference"
        if eligible
        else "insufficient_evidence",
        component_means={
            k: numerators[k] / supports[k] if supports[k] else None for k in weights
        },
        component_coverage={k: supports[k] / total for k in weights},
        weights=weights,
        interpretation="Six discrete roadside transect samples every <=50m and native DSM ruggedness; not area fractions, confirmed visibility or safety",
        bounds_scope="Bounds concern missing scored inputs only, not source error, sampling error, subjective preferences or actual views",
    )


SIGNIFICANCE_PREDICATES = {
    "documented_historical_event",
    "historical_supply_route",
    "documented_construction_start",
    "recognition_basis",
    "tourism_area_designation",
}


def iconic_score(known_road_m, total_m, road_entities, nearby_entities, policy):
    """Positive documented evidence index; no claim that unresearched roads lack value."""
    if total_m <= 0 or known_road_m < 0 or known_road_m > total_m + 1e-5:
        raise ValueError("Invalid documented road exposure")
    identities = sorted(set(road_entities) | set(nearby_entities))
    evidence_index = 100 * (
        0.7 * min(1, known_road_m / total_m)
        + 0.3
        * min(
            1,
            (
                len(set(road_entities))
                + 0.5 * len(set(nearby_entities) - set(road_entities))
            )
            / policy["iconic_entity_saturation"],
        )
    )
    return dict(
        score=evidence_index if identities else None,
        score_kind="positive_documented_significance_index_not_total_iconic_value",
        status="partial_research_catalog",
        ranking_eligible=False,
        documented_significant_road_fraction=known_road_m / total_m,
        documented_road_entities=sorted(set(road_entities)),
        nearby_only_entities=sorted(set(nearby_entities)),
        research_completeness=None,
        possible_total_value_bounds=[evidence_index, 100.0],
        interpretation="Historical or official-recognition claims only; notices and incidents confer no iconic status. Context belongs to road/entity identity, not an event site on every segment. Unresearched roads remain unknown.",
    )


def express_scores(routes, policy):
    if (
        any(
            not math.isfinite(policy[k]) or policy[k] < 0
            for k in ("express_eta_weight", "express_distance_weight")
        )
        or abs(policy["express_eta_weight"] + policy["express_distance_weight"] - 1)
        > 1e-9
    ):
        raise ValueError("Invalid Express weights")
    eligible = [
        r
        for r in routes
        if r.get("provider_comparison_eligible", r["route_ranking_eligible"])
        and all(
            isinstance(r[k], (int, float)) and math.isfinite(r[k]) and r[k] > 0
            for k in ("provider_duration_s", "provider_distance_m")
        )
    ]
    if not eligible:
        return {
            r["route_id"]: dict(
                score=None,
                ranking_eligible=False,
                status="no_eligible_provider_metrics",
            )
            for r in routes
        }
    fastest = min(r["provider_duration_s"] for r in eligible)
    shortest = min(r["provider_distance_m"] for r in eligible)
    results = {}
    for route in routes:
        if route not in eligible:
            results[route["route_id"]] = dict(
                score=None,
                ranking_eligible=False,
                status="route_requires_review_or_missing_metrics",
            )
            continue
        eta_component = fastest / route["provider_duration_s"]
        distance_component = shortest / route["provider_distance_m"]
        results[route["route_id"]] = dict(
            score=100
            * (
                policy["express_eta_weight"] * eta_component
                + policy["express_distance_weight"] * distance_component
            ),
            ranking_eligible=True,
            status="relative_provider_cost_preference",
            eta_component=eta_component,
            distance_component=distance_component,
            fastest_candidate_duration_s=fastest,
            shortest_candidate_distance_m=shortest,
            interpretation="Relative to these eligible candidates only; provider ETA is an estimate, not guaranteed live traffic or arrival time",
        )
    return results


def compare(a, b):
    x, y = a["scenic"], b["scenic"]
    if not a.get(
        "provider_comparison_eligible", a["route_ranking_eligible"]
    ) or not b.get("provider_comparison_eligible", b["route_ranking_eligible"]):
        ordering = "withheld_route_alignment_review"
    elif x["missing_evidence_bounds"][0] > y["missing_evidence_bounds"][1]:
        ordering = "a_above_b_under_missing_input_bounds"
    elif y["missing_evidence_bounds"][0] > x["missing_evidence_bounds"][1]:
        ordering = "b_above_a_under_missing_input_bounds"
    else:
        ordering = "not_resolved_by_missing_evidence_bounds"
    return dict(
        route_a=a["route_id"],
        route_b=b["route_id"],
        distance_delta_m=a["provider_distance_m"] - b["provider_distance_m"],
        eta_delta_s=a["provider_duration_s"] - b["provider_duration_s"],
        scenic_comparison=ordering,
        iconic_comparison="positive_evidence_indices_only; catalog incompleteness prevents a total iconic ranking",
        unique_documented_entities_a=sorted(
            set(
                a["iconic"]["documented_road_entities"]
                + a["iconic"]["nearby_only_entities"]
            )
            - set(
                b["iconic"]["documented_road_entities"]
                + b["iconic"]["nearby_only_entities"]
            )
        ),
        landscape_coverage_delta=a["landscape_sample_coverage"]
        - b["landscape_sample_coverage"],
    )
