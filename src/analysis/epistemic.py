"""
Decision Support System - Epistemic Uncertainty Propagation Engine
Analyzes the impact of epistemic uncertainty (E) realization on deterministic MCDM methods
without mutating source evaluations or baseline runs on disk.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from src.analysis.dispatcher import AnalysisDispatcher, load_rating_config_direct
from src.factors_manager import load_factors_config


def generate_discrete_grid(r: float, E: int, Ke: float = 0.5) -> List[float]:
    """Generates discrete test ratings for r ± (Ke * E) clipped to [0, 10]."""
    r_val = float(r)
    e_val = int(E)
    ke_val = float(Ke)

    if e_val <= 0:
        return [round(float(np.clip(r_val, 0.0, 10.0)), 2)]

    if abs(ke_val - 0.5) < 1e-5:
        step = 0.5 if (e_val % 2 != 0) else 1.0
    else:
        step = ke_val if (e_val % 2 != 0) else (2.0 * ke_val)

    max_delta = ke_val * e_val
    deltas = np.arange(-max_delta, max_delta + (step / 2.0), step)
    deltas = np.unique(np.round(np.append(deltas, 0.0), 4))
    deltas.sort()

    grid = [round(float(np.clip(r_val + d, 0.0, 10.0)), 2) for d in deltas]
    return sorted(list(set(grid)))


class EpistemicEngine:
    """Orchestrates Level 1, Level 2, and Level 3 Epistemic Propagation Sensitivity."""

    @classmethod
    def analyze_individual_criteria(
        cls,
        baseline_run_id: str,
        method_name: str,
        material_shift_threshold: float = 0.03
    ) -> Dict[str, Any]:
        baseline_run = AnalysisDispatcher.load_baseline_run(baseline_run_id)
        ctx = AnalysisDispatcher.build_in_memory_context(baseline_run_id)
        rating_cfg = load_rating_config_direct()
        ke = float(rating_cfg.get("coefficients", {}).get("Ke", 0.5))

        countries = ctx["countries"]
        factors = ctx["factors"]
        c_ids = ctx["criteria_ids"]
        base_mat = np.copy(ctx["matrix"])
        base_weights = np.array([ctx["global_weights"].get(f["id"], 0.0) for f in factors], dtype=float)
        if np.sum(base_weights) > 0:
            base_weights /= np.sum(base_weights)

        base_res = AnalysisDispatcher.execute_method_iteration(
            method_name=method_name,
            matrix=base_mat,
            fuzzy_matrix=ctx["fuzzy_matrix"],
            weights=base_weights,
            types=ctx["types"],
            parameters=baseline_run.get("parameters", {}),
            countries=countries
        )
        base_scores = base_res.get("scores", [0.0] * len(countries))
        base_winner = base_res.get("winner", countries[0])
        base_adv = round(base_scores[0] - base_scores[1], 5) if len(countries) >= 2 else 0.0

        domain_map = {d["id"]: d["name"] for d in ctx["domains"]}
        criteria_results = []

        for j, fid in enumerate(c_ids):
            factor = next(f for f in factors if f["id"] == fid)
            cat_name = domain_map.get(factor["domain_id"], "Unknown")
            gw = float(base_weights[j])

            eval_map = {}
            has_uncertainty = False
            for i, c in enumerate(countries):
                ev = next((e for e in ctx["evaluations"] if e["criterion_id"] == fid and e["country"] == c), None)
                r_val = ev["rating"] if ev else base_mat[i, j]
                e_val = ev["uncertainty"] if ev else 0
                grid = generate_discrete_grid(r_val, e_val, ke)
                if e_val > 0:
                    has_uncertainty = True
                eval_map[c] = {"r": r_val, "E": e_val, "grid": grid}

            if not has_uncertainty:
                continue

            trajectories = []
            adv_values = []
            flips = []

            for test_r0 in eval_map[countries[0]]["grid"]:
                test_mat = np.copy(base_mat)
                test_mat[0, j] = test_r0
                iter_res = AnalysisDispatcher.execute_method_iteration(
                    method_name=method_name, matrix=test_mat, fuzzy_matrix=ctx["fuzzy_matrix"],
                    weights=base_weights, types=ctx["types"], parameters=baseline_run.get("parameters", {}),
                    countries=countries
                )
                sc = iter_res.get("scores", [0.0, 0.0])
                adv = round(sc[0] - sc[1], 5)
                adv_values.append(adv)
                win = iter_res.get("winner", "")
                if win != base_winner:
                    flips.append({"perturbed_country": countries[0], "rating": test_r0, "new_winner": win, "advantage": adv})
                trajectories.append({"perturbed_country": countries[0], "tested_rating": test_r0, "scores": sc, "advantage": adv, "winner": win})

            if len(countries) >= 2:
                for test_r1 in eval_map[countries[1]]["grid"]:
                    test_mat = np.copy(base_mat)
                    test_mat[1, j] = test_r1
                    iter_res = AnalysisDispatcher.execute_method_iteration(
                        method_name=method_name, matrix=test_mat, fuzzy_matrix=ctx["fuzzy_matrix"],
                        weights=base_weights, types=ctx["types"], parameters=baseline_run.get("parameters", {}),
                        countries=countries
                    )
                    sc = iter_res.get("scores", [0.0, 0.0])
                    adv = round(sc[0] - sc[1], 5)
                    adv_values.append(adv)
                    win = iter_res.get("winner", "")
                    if win != base_winner:
                        flips.append({"perturbed_country": countries[1], "rating": test_r1, "new_winner": win, "advantage": adv})
                    trajectories.append({"perturbed_country": countries[1], "tested_rating": test_r1, "scores": sc, "advantage": adv, "winner": win})

            min_r0, max_r0 = eval_map[countries[0]]["grid"][0], eval_map[countries[0]]["grid"][-1]
            min_r1, max_r1 = eval_map[countries[1]]["grid"][0], eval_map[countries[1]]["grid"][-1]

            for (r0_ext, r1_ext, desc) in [(min_r0, max_r1, "Extreme Disadvantage C1"), (max_r0, min_r1, "Extreme Advantage C1")]:
                test_mat = np.copy(base_mat)
                test_mat[0, j] = r0_ext
                test_mat[1, j] = r1_ext
                iter_res = AnalysisDispatcher.execute_method_iteration(
                    method_name=method_name, matrix=test_mat, fuzzy_matrix=ctx["fuzzy_matrix"],
                    weights=base_weights, types=ctx["types"], parameters=baseline_run.get("parameters", {}),
                    countries=countries
                )
                sc = iter_res.get("scores", [0.0, 0.0])
                adv = round(sc[0] - sc[1], 5)
                adv_values.append(adv)
                win = iter_res.get("winner", "")
                if win != base_winner:
                    flips.append({"perturbed_country": "Both (Antagonistic)", "rating": f"{r0_ext} vs {r1_ext}", "new_winner": win, "advantage": adv})

            min_adv = min(adv_values) if adv_values else base_adv
            max_adv = max(adv_values) if adv_values else base_adv
            max_shift = round(max(abs(min_adv - base_adv), abs(max_adv - base_adv)), 5)
            has_flip = len(flips) > 0

            if has_flip:
                classification = "Winner Flip"
            elif max_shift >= material_shift_threshold:
                classification = "Material Shift"
            else:
                classification = "Negligible"

            criteria_results.append({
                "criterion_id": fid,
                "criterion_name": factor["name"],
                "short_name": factor.get("short_name", fid),
                "category_id": factor["domain_id"],
                "category_name": cat_name,
                "global_weight": round(gw, 5),
                "evaluations": eval_map,
                "baseline_advantage": base_adv,
                "min_advantage": min_adv,
                "max_advantage": max_adv,
                "max_abs_advantage_shift": max_shift,
                "winner_flipped": has_flip,
                "classification": classification,
                "flip_realizations": flips,
                "trajectories": trajectories
            })

        criteria_results.sort(key=lambda x: (x["winner_flipped"], x["max_abs_advantage_shift"]), reverse=True)

        return {
            "metadata": {
                "analysis_type": "epistemic_individual_oat",
                "baseline_run_id": baseline_run_id,
                "baseline_run_name": baseline_run.get("name", ""),
                "method_name": method_name,
                "Ke": ke,
                "countries": countries,
                "baseline_winner": base_winner,
                "baseline_advantage": base_adv,
                "total_uncertain_criteria": len(criteria_results)
            },
            "criteria_results": criteria_results
        }

    @classmethod
    def select_category_representatives(
        cls,
        strategy: str = "highest_weight_x_e"
    ) -> List[Dict[str, Any]]:
        ctx = AnalysisDispatcher.build_in_memory_context()
        factors = ctx["factors"]
        domains = ctx["domains"]
        global_w = ctx["global_weights"]
        evals = ctx["evaluations"]

        representatives = []
        for domain in domains:
            d_id = domain["id"]
            d_factors = [f for f in factors if f["domain_id"] == d_id]
            if not d_factors:
                continue

            candidates = []
            for f in d_factors:
                fid = f["id"]
                gw = global_w.get(fid, 0.0)
                f_evals = [e for e in evals if e["criterion_id"] == fid]
                max_e = max([e.get("uncertainty", 0) for e in f_evals]) if f_evals else 0
                candidates.append({
                    "factor": f,
                    "global_weight": gw,
                    "max_e": max_e,
                    "weight_x_e": gw * max_e
                })

            if strategy == "highest_weight":
                chosen = max(candidates, key=lambda x: x["global_weight"])
            elif strategy == "median_weight":
                sorted_c = sorted(candidates, key=lambda x: x["global_weight"])
                chosen = sorted_c[len(sorted_c) // 2]
            else:
                chosen = max(candidates, key=lambda x: (x["weight_x_e"], x["global_weight"]))

            f_obj = chosen["factor"]
            representatives.append({
                "domain_id": d_id,
                "domain_name": domain["name"],
                "criterion_id": f_obj["id"],
                "criterion_name": f_obj["name"],
                "short_name": f_obj.get("short_name", f_obj["id"]),
                "global_weight": round(chosen["global_weight"], 5),
                "max_uncertainty_E": chosen["max_e"],
                "weight_x_e": round(chosen["weight_x_e"], 5)
            })

        return representatives

    @classmethod
    def analyze_combined_scenarios(
        cls,
        baseline_run_id: str,
        method_name: str,
        representatives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        baseline_run = AnalysisDispatcher.load_baseline_run(baseline_run_id)
        ctx = AnalysisDispatcher.build_in_memory_context(baseline_run_id)
        rating_cfg = load_rating_config_direct()
        ke = float(rating_cfg.get("coefficients", {}).get("Ke", 0.5))

        countries = ctx["countries"]
        factors = ctx["factors"]
        c_ids = ctx["criteria_ids"]
        base_mat = np.copy(ctx["matrix"])
        base_weights = np.array([ctx["global_weights"].get(f["id"], 0.0) for f in factors], dtype=float)
        if np.sum(base_weights) > 0:
            base_weights /= np.sum(base_weights)

        base_res = AnalysisDispatcher.execute_method_iteration(
            method_name=method_name, matrix=base_mat, fuzzy_matrix=ctx["fuzzy_matrix"],
            weights=base_weights, types=ctx["types"], parameters=baseline_run.get("parameters", {}),
            countries=countries
        )
        base_scores = base_res.get("scores", [0.0] * len(countries))
        base_winner = base_res.get("winner", countries[0])
        base_adv = round(base_scores[0] - base_scores[1], 5) if len(countries) >= 2 else 0.0

        scenarios_def = [
            ("Baseline Realization", "All criteria evaluated at center point r", "base"),
            ("All Lower Bounds", "All category representatives realize at r - (Ke * E)", "all_low"),
            ("All Upper Bounds", "All category representatives realize at r + (Ke * E)", "all_high"),
            ("Mixed Scenario A", "Even categories realize Lower, Odd categories realize Upper", "mixed_a"),
            ("Mixed Scenario B", "Even categories realize Upper, Odd categories realize Lower", "mixed_b"),
            ("Adversarial Realization A", f"{countries[0]} realizes Lower (-Ke*E), {countries[1] if len(countries)>1 else 'C2'} realizes Upper (+Ke*E)", "adv_a"),
            ("Adversarial Realization B", f"{countries[0]} realizes Upper (+Ke*E), {countries[1] if len(countries)>1 else 'C2'} realizes Lower (-Ke*E)", "adv_b"),
        ]

        scenario_results = []

        for name, desc, code in scenarios_def:
            mat_scen = np.copy(base_mat)

            for idx, r_info in enumerate(representatives):
                fid = r_info["criterion_id"]
                col_j = c_ids.index(fid)

                for row_i, country in enumerate(countries):
                    ev = next((e for e in ctx["evaluations"] if e["criterion_id"] == fid and e["country"] == country), None)
                    r_val = ev["rating"] if ev else base_mat[row_i, col_j]
                    e_val = ev["uncertainty"] if ev else 0
                    delta = ke * e_val

                    if code == "base": target_r = r_val
                    elif code == "all_low": target_r = r_val - delta
                    elif code == "all_high": target_r = r_val + delta
                    elif code == "mixed_a": target_r = (r_val - delta) if (idx % 2 == 0) else (r_val + delta)
                    elif code == "mixed_b": target_r = (r_val + delta) if (idx % 2 == 0) else (r_val - delta)
                    elif code == "adv_a": target_r = (r_val - delta) if row_i == 0 else (r_val + delta)
                    elif code == "adv_b": target_r = (r_val + delta) if row_i == 0 else (r_val - delta)
                    else: target_r = r_val

                    mat_scen[row_i, col_j] = np.clip(target_r, 0.0, 10.0)

            iter_res = AnalysisDispatcher.execute_method_iteration(
                method_name=method_name, matrix=mat_scen, fuzzy_matrix=ctx["fuzzy_matrix"],
                weights=base_weights, types=ctx["types"], parameters=baseline_run.get("parameters", {}),
                countries=countries
            )

            scores = iter_res.get("scores", [0.0] * len(countries))
            winner = iter_res.get("winner", countries[0])
            adv = round(scores[0] - scores[1], 5) if len(countries) >= 2 else 0.0

            scenario_results.append({
                "scenario_name": name,
                "description": desc,
                "code": code,
                "winner": winner,
                "winner_flipped": (winner != base_winner),
                "decision_advantage": adv,
                "delta_advantage_from_baseline": round(adv - base_adv, 5),
                "scores": {countries[i]: round(float(scores[i]), 5) for i in range(len(countries))},
                "ranking": {countries[i]: int(iter_res.get("ranking", [1, 2])[i]) for i in range(len(countries))}
            })

        return {
            "metadata": {
                "analysis_type": "epistemic_combined_scenarios",
                "baseline_run_id": baseline_run_id,
                "method_name": method_name,
                "Ke": ke,
                "countries": countries,
                "baseline_winner": base_winner,
                "baseline_advantage": base_adv,
                "representatives_used": representatives
            },
            "scenarios": scenario_results
        }