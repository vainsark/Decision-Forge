"""
Decision Support System - Monte Carlo Uncertainty Engine
Executes vectorized probabilistic simulations over epistemic (E) and combined (V + E)
uncertainty distributions across deterministic MCDM models without mutating disk files.
Supports exact Trapezoidal, Triangular, and Uniform (Rectangular) sampling based on fuzzy geometry.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional

from src.analysis.dispatcher import AnalysisDispatcher, load_rating_config_direct


class MonteCarloEngine:
    """Runs stochastic decision simulations across uncertainty ranges for deterministic MCDM methods."""

    @classmethod
    def run_simulation(
        cls,
        baseline_run_id: str,
        method_names: List[str],
        num_iterations: int = 10000,
        mode: str = "full_uncertainty",
        distribution: str = "trapezoidal",
        coverage_pct: float = 75.0,
        discrete_step: Optional[float] = 0.5
    ) -> Dict[str, Any]:
        baseline_run = AnalysisDispatcher.load_baseline_run(baseline_run_id)
        ctx = AnalysisDispatcher.build_in_memory_context(baseline_run_id)
        rating_cfg = load_rating_config_direct()
        coeffs = rating_cfg.get("coefficients", {})
        ke = float(coeffs.get("Ke", 0.5))
        kv = float(coeffs.get("Kv", 0.5))
        kb = float(coeffs.get("Kb", 1.0))

        countries = ctx["countries"]
        factors = ctx["factors"]
        c_ids = ctx["criteria_ids"]
        num_countries = len(countries)
        num_factors = len(factors)

        base_mat = np.copy(ctx["matrix"])
        base_weights = np.array([ctx["global_weights"].get(f["id"], 0.0) for f in factors], dtype=float)
        if np.sum(base_weights) > 0:
            base_weights /= np.sum(base_weights)

        # Precompute trapezoids [a, b, c, d] for every country (i) and criterion (j)
        trapezoids = np.zeros((num_countries, num_factors, 4))

        for j, fid in enumerate(c_ids):
            for i, c in enumerate(countries):
                ev = next((e for e in ctx["evaluations"] if e["criterion_id"] == fid and e["country"] == c), None)
                if ev:
                    r = float(ev.get("rating", base_mat[i, j]))
                    e_val = float(ev.get("uncertainty", 0))
                    v_val = float(ev.get("volatility", 0))
                    bias = ev.get("bias", "neu")
                    
                    if mode == "epistemic_only":
                        v_val = 0.0  # Epistemic only -> rectangle/uniform over epistemic spread
                        
                    v_spread = v_val * kv
                    e_spread = e_val * ke
                    total_spread = v_spread + e_spread
                    
                    half_e = e_spread * 0.5
                    
                    bias_shift = 0.0
                    if bias == "pes":
                        bias_shift = -0.2 * total_spread * kb
                    elif bias == "opt":
                        bias_shift = 0.2 * total_spread * kb
                        
                    b = r - half_e + bias_shift
                    c = r + half_e + bias_shift
                    a = r - total_spread + bias_shift
                    d = r + total_spread + bias_shift
                    
                    # Ensure valid trapezoid bounds
                    a = max(0.0, a)
                    b = max(a, b)
                    c = max(b, c)
                    d = max(c, d)
                    
                    trapezoids[i, j] = [a, b, c, d]
                else:
                    r = base_mat[i, j]
                    trapezoids[i, j] = [r, r, r, r]

        sampled_matrices = np.zeros((num_iterations, num_countries, num_factors))

        if distribution == "normal":
            deltas = np.zeros((num_countries, num_factors))
            for j in range(num_factors):
                for i in range(num_countries):
                    a, _, _, d = trapezoids[i, j]
                    deltas[i, j] = (d - a) / 4.0
            p_val = float(np.clip(coverage_pct / 100.0, 0.50, 0.999))
            z_score = float(stats.norm.ppf((1.0 + p_val) / 2.0))
            z_score = max(z_score, 0.01)
            sigma = np.where(deltas > 0, deltas / z_score, 0.0)
            sampled_matrices = np.random.normal(
                loc=np.tile(base_mat, (num_iterations, 1, 1)),
                scale=np.tile(sigma, (num_iterations, 1, 1))
            )
        elif distribution == "uniform":
            lows = trapezoids[:, :, 0]
            highs = trapezoids[:, :, 3]
            sampled_matrices = np.random.uniform(
                low=np.tile(lows, (num_iterations, 1, 1)),
                high=np.tile(highs, (num_iterations, 1, 1))
            )
        else:
            # True Trapezoidal / Triangular / Rectangular sampling matching fuzzy geometry
            for i in range(num_countries):
                for j in range(num_factors):
                    a, b, c, d = trapezoids[i, j]
                    if np.isclose(a, d):
                        sampled_matrices[:, i, j] = a
                    elif np.isclose(a, b) and np.isclose(c, d):
                        sampled_matrices[:, i, j] = np.random.uniform(a, d, num_iterations)
                    else:
                        sampled_matrices[:, i, j] = cls._sample_trapezoidal_vectorized(a, b, c, d, num_iterations)

        if discrete_step and discrete_step > 0:
            sampled_matrices = np.round(sampled_matrices / discrete_step) * discrete_step

        sampled_matrices = np.clip(sampled_matrices, 0.0001, 10.0)

        model_results = {}
        for m_name in method_names:
            win_counts = {c: 0 for c in countries}
            score_samples = {c: np.zeros(num_iterations) for c in countries}
            rank_samples = {c: np.zeros(num_iterations, dtype=int) for c in countries}
            advantage_samples = np.zeros(num_iterations)

            params = baseline_run.get("parameters", {})

            for it in range(num_iterations):
                mat_iter = sampled_matrices[it]
                iter_res = AnalysisDispatcher.execute_method_iteration(
                    method_name=m_name,
                    matrix=mat_iter,
                    fuzzy_matrix=ctx["fuzzy_matrix"],
                    weights=base_weights,
                    types=ctx["types"],
                    parameters=params,
                    countries=countries
                )

                if iter_res.get("status") == "success":
                    sc = iter_res.get("scores", [0.0] * num_countries)
                    rnk = iter_res.get("ranking", [1] * num_countries)
                    winner = iter_res.get("winner", countries[0])

                    win_counts[winner] += 1
                    for idx, c in enumerate(countries):
                        score_samples[c][it] = sc[idx]
                        rank_samples[c][it] = rnk[idx]

                    if num_countries >= 2:
                        advantage_samples[it] = sc[0] - sc[1]

            win_pcts = {c: round((win_counts[c] / num_iterations) * 100.0, 2) for c in countries}
            country_stats = {}
            for c in countries:
                s_arr = score_samples[c]
                r_arr = rank_samples[c]
                country_stats[c] = {
                    "win_percentage": win_pcts[c],
                    "total_wins": win_counts[c],
                    "mean_score": round(float(np.mean(s_arr)), 5),
                    "score_std": round(float(np.std(s_arr)), 5),
                    "ci_95_score": [round(float(np.percentile(s_arr, 2.5)), 5), round(float(np.percentile(s_arr, 97.5)), 5)],
                    "mean_rank": round(float(np.mean(r_arr)), 2),
                    "rank_1_pct": win_pcts[c]
                }

            top_winner = max(win_pcts.keys(), key=lambda k: win_pcts[k]) if win_pcts else (countries[0] if countries else "N/A")
            subsample_idx = np.linspace(0, num_iterations - 1, min(500, num_iterations), dtype=int)
            adv_subsample = [round(float(advantage_samples[i]), 5) for i in subsample_idx]

            model_results[m_name] = {
                "top_probabilistic_winner": top_winner,
                "top_winner_pct": win_pcts[top_winner],
                "country_stats": country_stats,
                "advantage_stats": {
                    "mean_advantage": round(float(np.mean(advantage_samples)), 5),
                    "std_advantage": round(float(np.std(advantage_samples)), 5),
                    "ci_95_advantage": [round(float(np.percentile(advantage_samples, 2.5)), 5), round(float(np.percentile(advantage_samples, 97.5)), 5)],
                    "pct_advantage_positive": round(float(np.mean(advantage_samples > 0) * 100.0), 2)
                },
                "advantage_subsample": adv_subsample
            }

        return {
            "metadata": {
                "analysis_type": "monte_carlo",
                "dimension": "monte_carlo_simulation",
                "baseline_run_id": baseline_run_id,
                "baseline_run_name": baseline_run.get("name", ""),
                "models_evaluated": method_names,
                "num_iterations": num_iterations,
                "uncertainty_mode": mode,
                "distribution": distribution,
                "coverage_pct": coverage_pct if distribution == "normal" else 100.0,
                "discrete_step": discrete_step,
                "countries": countries,
                "coefficients": {"Ke": ke, "Kv": kv, "Kb": kb}
            },
            "model_results": model_results
        }

    @staticmethod
    def _sample_trapezoidal_vectorized(a: float, b: float, c: float, d: float, num_samples: int) -> np.ndarray:
        """Vectorized inverse transform sampling for a trapezoidal/triangular distribution [a, b, c, d]."""
        u = np.random.uniform(0, 1, num_samples)
        denom = d + c - b - a
        if np.isclose(denom, 0.0) or np.isclose(a, d):
            return np.full(num_samples, a)

        h = 2.0 / denom
        area1 = 0.5 * (b - a) * h if not np.isclose(a, b) else 0.0
        area2 = area1 + (c - b) * h if not np.isclose(b, c) else area1

        samples = np.empty(num_samples)

        mask1 = u < area1
        if np.any(mask1) and not np.isclose(a, b):
            samples[mask1] = a + np.sqrt(u[mask1] * (b - a) * denom)

        mask2 = (u >= area1) & (u < area2)
        if np.any(mask2):
            if np.isclose(b, c):
                samples[mask2] = b
            else:
                samples[mask2] = b + (u[mask2] - area1) / h

        mask3 = u >= area2
        if np.any(mask3) and not np.isclose(c, d):
            rem = 1.0 - u[mask3]
            samples[mask3] = d - np.sqrt(rem * (d - c) * denom)

        return samples