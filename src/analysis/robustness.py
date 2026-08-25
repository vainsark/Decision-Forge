"""
Decision Support System - Robustness Engine
Analyzes decision stability, critical switching thresholds, ranking inversions,
and victory margins across perturbed MCDM configurations.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.analysis.dispatcher import AnalysisDispatcher
from src.analysis.sensitivity import SensitivityEngine


class RobustnessEngine:
    """Evaluates decision boundaries and rank stability from sensitivity experiment data."""

    @classmethod
    def find_criterion_weight_threshold(
        cls,
        baseline_run_id: str,
        target_criterion_id: str,
        method_name: str,
        resolution: int = 41
    ) -> Dict[str, Any]:
        """Performs a dense 0% -> 100% sweep on a single criterion weight to find switching boundaries (Single Flat Mode)."""
        test_weights = np.linspace(0.0, 1.0, resolution).tolist()
        
        sens_res = SensitivityEngine.analyze_criteria_weights(
            baseline_run_id=baseline_run_id,
            target_criterion_id=target_criterion_id,
            test_weights=test_weights,
            methods_to_run=[method_name]
        )
        
        base_val = sens_res["metadata"]["baseline_value"]
        iterations = sens_res.get("iterations", [])
        base_winner = sens_res.get("baseline_summary", {}).get(method_name, {}).get("winner", "")
        
        lower_flip = None
        upper_flip = None
        stable_range_min = 0.0
        stable_range_max = 1.0

        for it in iterations:
            w = it["param_value"]
            res = it.get("method_results", {}).get(method_name, {})
            if res.get("status") != "success":
                continue
                
            cur_winner = res.get("winner", "")
            
            if w < base_val:
                if cur_winner != base_winner:
                    lower_flip = w
            elif w > base_val:
                if cur_winner != base_winner and upper_flip is None:
                    upper_flip = w

        if lower_flip is not None:
            stable_range_min = lower_flip
        if upper_flip is not None:
            stable_range_max = upper_flip

        return {
            "method": method_name,
            "target_domain_id": target_criterion_id,
            "target_domain_name": sens_res["metadata"]["target_name"],
            "baseline_weight": base_val,
            "baseline_winner": base_winner,
            "is_strictly_stable_across_entire_range": (lower_flip is None and upper_flip is None),
            "lower_switch_boundary": lower_flip,
            "upper_switch_boundary": upper_flip,
            "safe_stability_range": [stable_range_min, stable_range_max],
            "max_tolerable_decrease": round(base_val - stable_range_min, 4) if lower_flip else round(base_val, 4),
            "max_tolerable_increase": round(stable_range_max - base_val, 4) if upper_flip else round(1.0 - base_val, 4)
        }


    @classmethod
    def evaluate_sensitivity_robustness(cls, sensitivity_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the structured output of a SensitivityEngine run to extract
        winner stability flags, critical switching points, and margin statistics.
        """
        metadata = sensitivity_result.get("metadata", {})
        baseline_summary = sensitivity_result.get("baseline_summary", {})
        iterations = sensitivity_result.get("iterations", [])
        
        # Robust method key lookup with fallback to first iteration
        methods = metadata.get("methods_evaluated", metadata.get("methods_executed", []))
        if not methods and iterations:
            methods = list(iterations[0].get("method_results", {}).keys())
            
        countries = metadata.get("countries", [])

        if not iterations:
            return {"status": "error", "message": "No iteration data found in sensitivity result."}

        method_robustness = {}

        for m_name in methods:
            base = baseline_summary.get(m_name, {})
            base_winner = base.get("winner", "")
            
            winner_retained_count = 0
            flips = []
            margins = []
            rank_inversions_count = 0
            
            for iter_idx, it in enumerate(iterations):
                param_val = it.get("param_value")
                res = it.get("method_results", {}).get(m_name, {})
                
                if res.get("status") != "success":
                    continue
                    
                cur_winner = res.get("winner", "")
                scores = res.get("scores", {})
                ranks = res.get("ranking", {})
                
                # Check winner stability
                if cur_winner == base_winner:
                    winner_retained_count += 1
                else:
                    flips.append({
                        "param_value": param_val,
                        "iteration_index": iter_idx,
                        "baseline_winner": base_winner,
                        "new_winner": cur_winner,
                        "new_scores": scores
                    })
                
                # Calculate margin between 1st and 2nd place
                sorted_scores = sorted(scores.values(), reverse=True)
                if len(sorted_scores) >= 2:
                    margin = round(sorted_scores[0] - sorted_scores[1], 5)
                    margins.append(margin)
                    
                # Track rank delta inversions across all alternatives
                rank_deltas = res.get("rank_deltas", {})
                if any(delta != 0 for delta in rank_deltas.values()):
                    rank_inversions_count += 1

            total_valid_iters = len(margins)
            stability_pct = (winner_retained_count / total_valid_iters * 100.0) if total_valid_iters > 0 else 0.0
            
            method_robustness[m_name] = {
                "baseline_winner": base_winner,
                "is_winner_strictly_stable": (len(flips) == 0),
                "winner_stability_pct": round(stability_pct, 2),
                "flip_count": len(flips),
                "critical_flip_points": flips,
                "rank_inversion_iterations": rank_inversions_count,
                "margin_stats": {
                    "min_margin": min(margins) if margins else 0.0,
                    "max_margin": max(margins) if margins else 0.0,
                    "avg_margin": round(float(np.mean(margins)), 5) if margins else 0.0
                }
            }

        # Multi-method consensus check
        all_strictly_stable = all(mr["is_winner_strictly_stable"] for mr in method_robustness.values()) if method_robustness else True
        overall_stability_pct = (
            round(float(np.mean([mr["winner_stability_pct"] for mr in method_robustness.values()])), 2)
            if method_robustness else 0.0
        )

        return {
            "metadata": {
                "dimension": metadata.get("dimension", ""),
                "baseline_run_id": metadata.get("baseline_run_id", ""),
                "target_id": metadata.get("target_id", ""),
                "target_name": metadata.get("target_name", ""),
                "methods_analyzed": methods
            },
            "overall_summary": {
                "is_strictly_stable_all_methods": all_strictly_stable,
                "mean_stability_pct": overall_stability_pct,
                "total_iterations_tested": len(iterations)
            },
            "method_details": method_robustness
        }

    # =========================================================================
    # HIGH-RESOLUTION CATEGORY WEIGHT CRITICAL BOUNDARY FINDER
    # =========================================================================
    @classmethod
    def find_category_weight_threshold(
        cls,
        baseline_run_id: str,
        target_domain_id: str,
        method_name: str,
        resolution: int = 41
    ) -> Dict[str, Any]:
        """
        Performs a dense 0% -> 100% sweep on a single category weight to find
        the exact critical lower and upper boundary percentages where the winner flips.
        """
        test_weights = np.linspace(0.0, 1.0, resolution).tolist()
        
        sens_res = SensitivityEngine.analyze_category_weights(
            baseline_run_id=baseline_run_id,
            target_domain_id=target_domain_id,
            test_weights=test_weights,
            methods_to_run=[method_name]
        )
        
        base_cat_val = sens_res["metadata"]["baseline_value"]
        iterations = sens_res.get("iterations", [])
        base_winner = sens_res.get("baseline_summary", {}).get(method_name, {}).get("winner", "")
        
        lower_flip = None
        upper_flip = None
        stable_range_min = 0.0
        stable_range_max = 1.0

        for it in iterations:
            w = it["param_value"]
            res = it.get("method_results", {}).get(method_name, {})
            if res.get("status") != "success":
                continue
                
            cur_winner = res.get("winner", "")
            
            if w < base_cat_val:
                if cur_winner != base_winner:
                    lower_flip = w
            elif w > base_cat_val:
                if cur_winner != base_winner and upper_flip is None:
                    upper_flip = w

        if lower_flip is not None:
            stable_range_min = lower_flip
        if upper_flip is not None:
            stable_range_max = upper_flip

        return {
            "method": method_name,
            "target_domain_id": target_domain_id,
            "target_domain_name": sens_res["metadata"]["target_name"],
            "baseline_weight": base_cat_val,
            "baseline_winner": base_winner,
            "is_strictly_stable_across_entire_range": (lower_flip is None and upper_flip is None),
            "lower_switch_boundary": lower_flip,
            "upper_switch_boundary": upper_flip,
            "safe_stability_range": [stable_range_min, stable_range_max],
            "max_tolerable_decrease": round(base_cat_val - stable_range_min, 4) if lower_flip else round(base_cat_val, 4),
            "max_tolerable_increase": round(stable_range_max - base_cat_val, 4) if upper_flip else round(1.0 - base_cat_val, 4)
        }