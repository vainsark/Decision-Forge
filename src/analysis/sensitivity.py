"""
Decision Support System - Sensitivity Engine
Executes controlled parameter sweeps across the core sensitivity scopes and 
all-category tornado leverage without modifying source data or saved baseline files on disk.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.analysis.dispatcher import AnalysisDispatcher


# Standard predefined trapezoid defuzzification schemes
DEFUZZ_WEIGHT_SCHEMES = {
    "Baseline / GMIR (1/6, 2/6, 2/6, 1/6)": [0.1667, 0.3333, 0.3333, 0.1667],
    "Core-Focused (0.05, 0.45, 0.45, 0.05)": [0.05, 0.45, 0.45, 0.05],
    "Equal Weights (0.25, 0.25, 0.25, 0.25)": [0.25, 0.25, 0.25, 0.25],
    "Tail-Focused (0.35, 0.15, 0.15, 0.35)": [0.35, 0.15, 0.15, 0.35],
    "Asymmetric Optimistic (0.15, 0.30, 0.30, 0.25)": [0.15, 0.30, 0.30, 0.25],
    "Asymmetric Pessimistic (0.25, 0.30, 0.30, 0.15)": [0.25, 0.30, 0.30, 0.15]
}


class SensitivityEngine:
    """Orchestrates sensitivity sweeps across MCDM parameters and records score/ranking shifts."""

    @classmethod
    def _extract_baseline_state(cls, baseline_run: Dict[str, Any], methods: List[str]) -> Dict[str, Any]:
        """Extracts baseline winners, scores, and rankings for comparison."""
        countries = baseline_run.get("countries", [])
        baseline_summary = {}

        for m in methods:
            m_res = baseline_run.get("results", {}).get(m, {})
            if m_res.get("status") == "success":
                ranks = m_res.get("ranking", [])
                scores = m_res.get("scores", [])
                winner_idx = int(np.argmin(ranks)) if ranks else 0
                baseline_summary[m] = {
                    "scores": {countries[i]: float(scores[i]) for i in range(len(countries))},
                    "ranking": {countries[i]: int(ranks[i]) for i in range(len(countries))},
                    "winner": countries[winner_idx] if countries else "N/A"
                }
        return baseline_summary

    @classmethod
    def _evaluate_iteration(
        cls,
        methods: List[str],
        matrix: Any,
        fuzzy_matrix: Any,
        weights: np.ndarray,
        types: np.ndarray,
        parameters: Dict[str, Any],
        countries: List[str],
        baseline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes all chosen methods for a single parameter value and computes deltas."""
        iteration_results = {}

        for m_name in methods:
            res = AnalysisDispatcher.execute_method_iteration(
                method_name=m_name,
                matrix=matrix,
                fuzzy_matrix=fuzzy_matrix,
                weights=weights,
                types=types,
                parameters=parameters,
                countries=countries
            )

            if res.get("status") == "success":
                scores = res.get("scores", [])
                ranks = res.get("ranking", [])
                winner_idx = int(np.argmin(ranks))
                winner = countries[winner_idx]

                base = baseline_state.get(m_name, {})
                base_scores = base.get("scores", {})
                base_ranks = base.get("ranking", {})
                base_winner = base.get("winner", "")

                score_deltas = {}
                rank_deltas = {}
                for i, c in enumerate(countries):
                    cur_score = float(scores[i])
                    cur_rank = int(ranks[i])
                    score_deltas[c] = round(cur_score - base_scores.get(c, cur_score), 5)
                    rank_deltas[c] = cur_rank - base_ranks.get(c, cur_rank)

                iteration_results[m_name] = {
                    "status": "success",
                    "scores": {countries[i]: round(float(scores[i]), 5) for i in range(len(countries))},
                    "ranking": {countries[i]: int(ranks[i]) for i in range(len(countries))},
                    "winner": winner,
                    "winner_changed": (winner != base_winner) if base_winner else False,
                    "score_deltas": score_deltas,
                    "rank_deltas": rank_deltas
                }
            else:
                iteration_results[m_name] = {
                    "status": "error",
                    "message": res.get("message", "Execution failed")
                }

        return iteration_results

    # =========================================================================
    # 1. CATEGORY WEIGHT SENSITIVITY
    # =========================================================================
    @classmethod
    def analyze_category_weights(
        cls,
        baseline_run_id: str,
        target_domain_id: str,
        test_weights: List[float],
        methods_to_run: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        baseline_run = AnalysisDispatcher.load_baseline_run(baseline_run_id)
        ctx = AnalysisDispatcher.build_in_memory_context()
        
        methods = methods_to_run or baseline_run.get("methods_executed", [])
        baseline_state = cls._extract_baseline_state(baseline_run, methods)
        
        domain_name = next(
            (d["name"] for d in ctx["domains"] if d["id"] == target_domain_id), 
            target_domain_id
        )
        base_cat_weights = ctx["category_weights"]
        base_val = float(base_cat_weights.get(target_domain_id, 0.0))

        iterations = []
        for w in test_weights:
            new_cat_weights, derived_global_w = AnalysisDispatcher.rebalance_category_weights(
                target_domain_id=target_domain_id,
                new_weight=w,
                baseline_cat_weights=base_cat_weights,
                local_weights=ctx["local_weights"],
                factors_config=ctx["factors_config"]
            )

            res = cls._evaluate_iteration(
                methods=methods,
                matrix=ctx["matrix"],
                fuzzy_matrix=ctx["fuzzy_matrix"],
                weights=derived_global_w,
                types=ctx["types"],
                parameters=baseline_run.get("parameters", {}),
                countries=ctx["countries"],
                baseline_state=baseline_state
            )

            iterations.append({
                "param_value": round(float(w), 4),
                "category_weights_snapshot": {k: round(v, 4) for k, v in new_cat_weights.items()},
                "method_results": res
            })

        return {
            "metadata": {
                "baseline_run_id": baseline_run_id,
                "baseline_run_name": baseline_run.get("name", ""),
                "dimension": "category_weights",
                "target_id": target_domain_id,
                "target_name": domain_name,
                "baseline_value": round(base_val, 4),
                "countries": ctx["countries"],
                "methods_evaluated": methods
            },
            "baseline_summary": baseline_state,
            "iterations": iterations
        }

    # =========================================================================
    # 1.B CRITERIA WEIGHT SENSITIVITY (SINGLE FLAT MODE)
    # =========================================================================
    @classmethod
    def analyze_criteria_weights(
        cls,
        baseline_run_id: str,
        target_criterion_id: str,
        test_weights: List[float],
        methods_to_run: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        baseline_run = AnalysisDispatcher.load_baseline_run(baseline_run_id)
        ctx = AnalysisDispatcher.build_in_memory_context()
        
        methods = methods_to_run or baseline_run.get("methods_executed", [])
        baseline_state = cls._extract_baseline_state(baseline_run, methods)
        
        factor_obj = next((f for f in ctx["factors"] if f["id"] == target_criterion_id), None)
        factor_name = factor_obj["name"] if factor_obj else target_criterion_id
        
        base_global_weights = ctx["global_weights"]
        base_val = float(base_global_weights.get(target_criterion_id, 0.0))

        iterations = []
        for w in test_weights:
            derived_global_w = AnalysisDispatcher.rebalance_criteria_weights(
                target_criterion_id=target_criterion_id,
                new_weight=w,
                baseline_global_weights=base_global_weights,
                factors_config=ctx["factors_config"]
            )

            res = cls._evaluate_iteration(
                methods=methods,
                matrix=ctx["matrix"],
                fuzzy_matrix=ctx["fuzzy_matrix"],
                weights=derived_global_w,
                types=ctx["types"],
                parameters=baseline_run.get("parameters", {}),
                countries=ctx["countries"],
                baseline_state=baseline_state
            )

            iterations.append({
                "param_value": round(float(w), 4),
                "method_results": res
            })

        return {
            "metadata": {
                "baseline_run_id": baseline_run_id,
                "baseline_run_name": baseline_run.get("name", ""),
                "dimension": "criteria_weights",
                "target_id": target_criterion_id,
                "target_name": factor_name,
                "baseline_value": round(base_val, 4),
                "countries": ctx["countries"],
                "methods_evaluated": methods
            },
            "baseline_summary": baseline_state,
            "iterations": iterations
        }

    @classmethod
    def analyze_criteria_tornado(
        cls,
        baseline_run_id: str,
        method_name: str,
        perturbation_fraction: float = 0.50
    ) -> Dict[str, Any]:
        baseline_run = AnalysisDispatcher.load_baseline_run(baseline_run_id)
        ctx = AnalysisDispatcher.build_in_memory_context()
        
        base_summary = cls._extract_baseline_state(baseline_run, [method_name])
        base_method_data = base_summary.get(method_name, {})
        base_winner = base_method_data.get("winner", "")
        base_scores = base_method_data.get("scores", {})
        base_winner_score = base_scores.get(base_winner, 0.0)

        base_global_weights = ctx["global_weights"]
        tornado_entries = []

        for factor in ctx["factors"]:
            f_id = factor["id"]
            f_name = factor["name"]
            w0 = float(base_global_weights.get(f_id, 0.0))

            if w0 <= 0.0:
                continue

            w_low = max(0.0, w0 * (1.0 - perturbation_fraction))
            w_high = min(1.0, w0 * (1.0 + perturbation_fraction))

            glob_w_low = AnalysisDispatcher.rebalance_criteria_weights(
                target_criterion_id=f_id,
                new_weight=w_low,
                baseline_global_weights=base_global_weights,
                factors_config=ctx["factors_config"]
            )
            res_low = AnalysisDispatcher.execute_method_iteration(
                method_name=method_name,
                matrix=ctx["matrix"],
                fuzzy_matrix=ctx["fuzzy_matrix"],
                weights=glob_w_low,
                types=ctx["types"],
                parameters=baseline_run.get("parameters", {}),
                countries=ctx["countries"]
            )

            glob_w_high = AnalysisDispatcher.rebalance_criteria_weights(
                target_criterion_id=f_id,
                new_weight=w_high,
                baseline_global_weights=base_global_weights,
                factors_config=ctx["factors_config"]
            )
            res_high = AnalysisDispatcher.execute_method_iteration(
                method_name=method_name,
                matrix=ctx["matrix"],
                fuzzy_matrix=ctx["fuzzy_matrix"],
                weights=glob_w_high,
                types=ctx["types"],
                parameters=baseline_run.get("parameters", {}),
                countries=ctx["countries"]
            )

            s_low = res_low["scores"][ctx["countries"].index(base_winner)] if res_low.get("status") == "success" else base_winner_score
            s_high = res_high["scores"][ctx["countries"].index(base_winner)] if res_high.get("status") == "success" else base_winner_score

            delta_low = round(s_low - base_winner_score, 5)
            delta_high = round(s_high - base_winner_score, 5)
            swing = round(abs(s_high - s_low), 5)

            tornado_entries.append({
                "domain_id": f_id,
                "domain_name": f_name,
                "baseline_weight": round(w0, 4),
                "w_low": round(w_low, 4),
                "w_high": round(w_high, 4),
                "score_low": round(s_low, 5),
                "score_high": round(s_high, 5),
                "delta_low": delta_low,
                "delta_high": delta_high,
                "total_swing": swing
            })

        tornado_entries.sort(key=lambda x: x["total_swing"], reverse=True)

        return {
            "metadata": {
                "baseline_run_id": baseline_run_id,
                "method_name": method_name,
                "baseline_winner": base_winner,
                "baseline_winner_score": round(base_winner_score, 5),
                "perturbation_fraction": perturbation_fraction
            },
            "tornado_entries": tornado_entries
        }


    # =========================================================================
    # 2. FUZZY TRAPEZOID COMPONENT WEIGHTS
    # =========================================================================
    @classmethod
    def analyze_fuzzy_component_weights(
        cls,
        baseline_run_id: str,
        schemes: Optional[Dict[str, List[float]]] = None,
        methods_to_run: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        baseline_run = AnalysisDispatcher.load_baseline_run(baseline_run_id)
        ctx = AnalysisDispatcher.build_in_memory_context()
        
        test_schemes = schemes or DEFUZZ_WEIGHT_SCHEMES
        available_methods = methods_to_run or baseline_run.get("methods_executed", [])
        methods = [m for m in available_methods if "fuzzy" in m.lower() or m == "Fuzzy PROMETHEE"]
        if not methods:
            methods = ["Fuzzy PROMETHEE"]

        baseline_state = cls._extract_baseline_state(baseline_run, methods)
        global_w = np.array([ctx["global_weights"].get(f["id"], 0.0) for f in ctx["factors"]])
        if np.sum(global_w) > 0:
            global_w /= np.sum(global_w)

        iterations = []
        for scheme_name, weights_4 in test_schemes.items():
            tot = sum(weights_4)
            norm_w = [round(x / tot, 4) for x in weights_4] if tot > 0 else [0.25, 0.25, 0.25, 0.25]
            
            iter_params = dict(baseline_run.get("parameters", {}))
            iter_params["defuzz_weights"] = norm_w

            res = cls._evaluate_iteration(
                methods=methods,
                matrix=ctx["matrix"],
                fuzzy_matrix=ctx["fuzzy_matrix"],
                weights=global_w,
                types=ctx["types"],
                parameters=iter_params,
                countries=ctx["countries"],
                baseline_state=baseline_state
            )

            iterations.append({
                "param_value": scheme_name,
                "weights_vector": norm_w,
                "method_results": res
            })

        return {
            "metadata": {
                "baseline_run_id": baseline_run_id,
                "baseline_run_name": baseline_run.get("name", ""),
                "dimension": "fuzzy_component_weights",
                "countries": ctx["countries"],
                "methods_evaluated": methods
            },
            "baseline_summary": baseline_state,
            "iterations": iterations
        }

    # =========================================================================
    # 3. FUZZY PROMETHEE q / p PARAMETERS
    # =========================================================================
    @classmethod
    def analyze_promethee_qp(
        cls,
        baseline_run_id: str,
        qp_pairs: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        baseline_run = AnalysisDispatcher.load_baseline_run(baseline_run_id)
        ctx = AnalysisDispatcher.build_in_memory_context()
        methods = ["Fuzzy PROMETHEE"]
        baseline_state = cls._extract_baseline_state(baseline_run, methods)

        global_w = np.array([ctx["global_weights"].get(f["id"], 0.0) for f in ctx["factors"]])
        if np.sum(global_w) > 0:
            global_w /= np.sum(global_w)

        iterations = []
        for q_val, p_val in qp_pairs:
            q_clean = max(0.0, float(q_val))
            p_clean = max(q_clean + 0.01, float(p_val))

            iter_params = dict(baseline_run.get("parameters", {}))
            iter_params["promethee_q"] = q_clean
            iter_params["promethee_p"] = p_clean

            res = cls._evaluate_iteration(
                methods=methods,
                matrix=ctx["matrix"],
                fuzzy_matrix=ctx["fuzzy_matrix"],
                weights=global_w,
                types=ctx["types"],
                parameters=iter_params,
                countries=ctx["countries"],
                baseline_state=baseline_state
            )

            iterations.append({
                "param_value": f"q={q_clean:.2f}, p={p_clean:.2f}",
                "q": q_clean,
                "p": p_clean,
                "method_results": res
            })

        return {
            "metadata": {
                "baseline_run_id": baseline_run_id,
                "baseline_run_name": baseline_run.get("name", ""),
                "dimension": "fuzzy_promethee_qp",
                "countries": ctx["countries"],
                "methods_evaluated": methods
            },
            "baseline_summary": baseline_state,
            "iterations": iterations
        }

    # =========================================================================
    # 4. WASPAS LAMBDA
    # =========================================================================
    @classmethod
    def analyze_waspas_lambda(
        cls,
        baseline_run_id: str,
        lambda_values: List[float]
    ) -> Dict[str, Any]:
        baseline_run = AnalysisDispatcher.load_baseline_run(baseline_run_id)
        ctx = AnalysisDispatcher.build_in_memory_context()
        methods = ["WASPAS"]
        baseline_state = cls._extract_baseline_state(baseline_run, methods)

        global_w = np.array([ctx["global_weights"].get(f["id"], 0.0) for f in ctx["factors"]])
        if np.sum(global_w) > 0:
            global_w /= np.sum(global_w)

        iterations = []
        for lmbd in lambda_values:
            l_clean = float(np.clip(lmbd, 0.0, 1.0))
            iter_params = dict(baseline_run.get("parameters", {}))
            iter_params["WASPAS_lambda"] = l_clean

            res = cls._evaluate_iteration(
                methods=methods,
                matrix=ctx["matrix"],
                fuzzy_matrix=ctx["fuzzy_matrix"],
                weights=global_w,
                types=ctx["types"],
                parameters=iter_params,
                countries=ctx["countries"],
                baseline_state=baseline_state
            )

            iterations.append({
                "param_value": round(l_clean, 3),
                "method_results": res
            })

        return {
            "metadata": {
                "baseline_run_id": baseline_run_id,
                "baseline_run_name": baseline_run.get("name", ""),
                "dimension": "waspas_lambda",
                "countries": ctx["countries"],
                "methods_evaluated": methods
            },
            "baseline_summary": baseline_state,
            "iterations": iterations
        }

    # =========================================================================
    # 5. KV / KE MULTIPLIERS
    # =========================================================================
    @classmethod
    def analyze_kv_ke(
        cls,
        baseline_run_id: str,
        kv_ke_pairs: List[Tuple[float, float]],
        methods_to_run: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        baseline_run = AnalysisDispatcher.load_baseline_run(baseline_run_id)
        ctx = AnalysisDispatcher.build_in_memory_context()
        
        available_methods = methods_to_run or baseline_run.get("methods_executed", [])
        methods = [m for m in available_methods if "fuzzy" in m.lower() or m == "Fuzzy PROMETHEE"]
        if not methods:
            methods = ["Fuzzy PROMETHEE"]

        baseline_state = cls._extract_baseline_state(baseline_run, methods)
        global_w = np.array([ctx["global_weights"].get(f["id"], 0.0) for f in ctx["factors"]])
        if np.sum(global_w) > 0:
            global_w /= np.sum(global_w)

        iterations = []
        for kv, ke in kv_ke_pairs:
            coeffs = {"Kv": float(kv), "Ke": float(ke), "Kb": 1.0}
            in_memory_fuzzy_mat = AnalysisDispatcher.regenerate_fuzzy_matrix(
                evaluations=ctx["evaluations"],
                factors=ctx["factors"],
                countries=ctx["countries"],
                coeffs=coeffs
            )

            res = cls._evaluate_iteration(
                methods=methods,
                matrix=ctx["matrix"],
                fuzzy_matrix=in_memory_fuzzy_mat,
                weights=global_w,
                types=ctx["types"],
                parameters=baseline_run.get("parameters", {}),
                countries=ctx["countries"],
                baseline_state=baseline_state
            )

            iterations.append({
                "param_value": f"Kv={kv:.1f}, Ke={ke:.1f}",
                "Kv": float(kv),
                "Ke": float(ke),
                "method_results": res
            })

        return {
            "metadata": {
                "baseline_run_id": baseline_run_id,
                "baseline_run_name": baseline_run.get("name", ""),
                "dimension": "kv_ke",
                "countries": ctx["countries"],
                "methods_evaluated": methods
            },
            "baseline_summary": baseline_state,
            "iterations": iterations
        }

    # =========================================================================
    # 6. BIAS COEFFICIENT (Kb)
    # =========================================================================
    @classmethod
    def analyze_bias_coefficient(
        cls,
        baseline_run_id: str,
        kb_values: List[float],
        methods_to_run: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        baseline_run = AnalysisDispatcher.load_baseline_run(baseline_run_id)
        ctx = AnalysisDispatcher.build_in_memory_context()
        
        available_methods = methods_to_run or baseline_run.get("methods_executed", [])
        methods = [m for m in available_methods if "fuzzy" in m.lower() or m == "Fuzzy PROMETHEE"]
        if not methods:
            methods = ["Fuzzy PROMETHEE"]

        baseline_state = cls._extract_baseline_state(baseline_run, methods)
        global_w = np.array([ctx["global_weights"].get(f["id"], 0.0) for f in ctx["factors"]])
        if np.sum(global_w) > 0:
            global_w /= np.sum(global_w)

        iterations = []
        for kb in kb_values:
            coeffs = {"Kv": 0.5, "Ke": 0.5, "Kb": float(kb)}
            in_memory_fuzzy_mat = AnalysisDispatcher.regenerate_fuzzy_matrix(
                evaluations=ctx["evaluations"],
                factors=ctx["factors"],
                countries=ctx["countries"],
                coeffs=coeffs
            )

            res = cls._evaluate_iteration(
                methods=methods,
                matrix=ctx["matrix"],
                fuzzy_matrix=in_memory_fuzzy_mat,
                weights=global_w,
                types=ctx["types"],
                parameters=baseline_run.get("parameters", {}),
                countries=ctx["countries"],
                baseline_state=baseline_state
            )

            iterations.append({
                "param_value": round(float(kb), 2),
                "Kb": float(kb),
                "method_results": res
            })

        return {
            "metadata": {
                "baseline_run_id": baseline_run_id,
                "baseline_run_name": baseline_run.get("name", ""),
                "dimension": "bias_coefficient",
                "countries": ctx["countries"],
                "methods_evaluated": methods
            },
            "baseline_summary": baseline_state,
            "iterations": iterations
        }

    # =========================================================================
    # 7. ALL-CATEGORY TORNADO LEVERAGE ANALYSIS
    # =========================================================================
    @classmethod
    def analyze_category_tornado(
        cls,
        baseline_run_id: str,
        method_name: str,
        perturbation_fraction: float = 0.50
    ) -> Dict[str, Any]:
        baseline_run = AnalysisDispatcher.load_baseline_run(baseline_run_id)
        ctx = AnalysisDispatcher.build_in_memory_context()
        
        base_summary = cls._extract_baseline_state(baseline_run, [method_name])
        base_method_data = base_summary.get(method_name, {})
        base_winner = base_method_data.get("winner", "")
        base_scores = base_method_data.get("scores", {})
        base_winner_score = base_scores.get(base_winner, 0.0)

        base_cat_weights = ctx["category_weights"]
        tornado_entries = []

        for domain in ctx["domains"]:
            d_id = domain["id"]
            d_name = domain["name"]
            w0 = float(base_cat_weights.get(d_id, 0.0))

            if w0 <= 0.0:
                continue

            w_low = max(0.0, w0 * (1.0 - perturbation_fraction))
            w_high = min(1.0, w0 * (1.0 + perturbation_fraction))

            _, glob_w_low = AnalysisDispatcher.rebalance_category_weights(
                target_domain_id=d_id,
                new_weight=w_low,
                baseline_cat_weights=base_cat_weights,
                local_weights=ctx["local_weights"],
                factors_config=ctx["factors_config"]
            )
            res_low = AnalysisDispatcher.execute_method_iteration(
                method_name=method_name,
                matrix=ctx["matrix"],
                fuzzy_matrix=ctx["fuzzy_matrix"],
                weights=glob_w_low,
                types=ctx["types"],
                parameters=baseline_run.get("parameters", {}),
                countries=ctx["countries"]
            )

            _, glob_w_high = AnalysisDispatcher.rebalance_category_weights(
                target_domain_id=d_id,
                new_weight=w_high,
                baseline_cat_weights=base_cat_weights,
                local_weights=ctx["local_weights"],
                factors_config=ctx["factors_config"]
            )
            res_high = AnalysisDispatcher.execute_method_iteration(
                method_name=method_name,
                matrix=ctx["matrix"],
                fuzzy_matrix=ctx["fuzzy_matrix"],
                weights=glob_w_high,
                types=ctx["types"],
                parameters=baseline_run.get("parameters", {}),
                countries=ctx["countries"]
            )

            s_low = res_low["scores"][ctx["countries"].index(base_winner)] if res_low.get("status") == "success" else base_winner_score
            s_high = res_high["scores"][ctx["countries"].index(base_winner)] if res_high.get("status") == "success" else base_winner_score

            delta_low = round(s_low - base_winner_score, 5)
            delta_high = round(s_high - base_winner_score, 5)
            swing = round(abs(s_high - s_low), 5)

            tornado_entries.append({
                "domain_id": d_id,
                "domain_name": d_name,
                "baseline_weight": round(w0, 4),
                "w_low": round(w_low, 4),
                "w_high": round(w_high, 4),
                "score_low": round(s_low, 5),
                "score_high": round(s_high, 5),
                "delta_low": delta_low,
                "delta_high": delta_high,
                "total_swing": swing
            })

        tornado_entries.sort(key=lambda x: x["total_swing"], reverse=True)

        return {
            "metadata": {
                "baseline_run_id": baseline_run_id,
                "method_name": method_name,
                "baseline_winner": base_winner,
                "baseline_winner_score": round(base_winner_score, 5),
                "perturbation_fraction": perturbation_fraction
            },
            "tornado_entries": tornado_entries
        }