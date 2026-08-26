"""
Decision Support System - In-Memory Analysis Dispatcher
Extracts baseline runs, executes non-destructive in-memory transformations,
and restores evaluation/weight contexts from historical snapshots without external module dependencies.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from src.mcdm_methods import METHOD_REGISTRY
from src.factors_manager import load_factors_config
from src.project_manager import get_active_project_dir

# =========================================================================
# DYNAMIC PATH RESOLUTION FOR MULTI-PROJECT WORKSPACES
# =========================================================================
def _get_project_data_dir() -> str:
    """Returns the active project directory, asserting it is not None."""
    proj_dir = get_active_project_dir()
    assert proj_dir is not None, "Active project directory is required."
    return proj_dir

def get_runs_dir() -> str:
    """Returns the runs directory path inside the active project folder."""
    return os.path.join(_get_project_data_dir(), 'runs')

def get_analysis_runs_dir() -> str:
    """Returns the analysis runs directory path inside the active project folder."""
    return os.path.join(_get_project_data_dir(), 'analysis_runs')

def get_evaluations_filepath() -> str:
    """Returns the path to the evaluations file for the active project."""
    return os.path.join(_get_project_data_dir(), 'evaluations.json')

def get_rating_config_filepath() -> str:
    """Returns the path to the rating configuration file for the active project."""
    return os.path.join(_get_project_data_dir(), 'rating_config.json')

def get_weights_filepath() -> str:
    """Returns the path to the weights file for the active project."""
    return os.path.join(_get_project_data_dir(), 'weights.json')

def get_factors_filepath() -> str:
    """Returns the path to the factors configuration file for the active project."""
    return os.path.join(_get_project_data_dir(), 'factors_config.json')


# =========================================================================
# DIRECT SAFE LOADERS & HELPERS (WITH COMPATIBILITY ALIASES)
# =========================================================================
def parse_factor_type(t_val: Any) -> int:
    """Safely converts criterion type representation (int or str) to 1 (benefit) or -1 (cost)."""
    if isinstance(t_val, (int, float)):
        return 1 if t_val >= 0 else -1
    if isinstance(t_val, str):
        return 1 if t_val.strip().lower() in ["benefit", "1", "max", "true"] else -1
    return 1


def load_weights_direct() -> Dict[str, Any]:
    """Loads weights data directly from the active project's weights file."""
    weights_file = get_weights_filepath()
    if os.path.exists(weights_file):
        with open(weights_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

_load_weights_safely = load_weights_direct


def load_evaluations_direct() -> List[Dict[str, Any]]:
    """Loads evaluations directly from the active project's evaluations file."""
    evals_file = get_evaluations_filepath()
    if os.path.exists(evals_file):
        with open(evals_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

_load_evaluations_safely = load_evaluations_direct


def load_rating_config_direct() -> Dict[str, Any]:
    """Loads rating configuration from the active project with default coefficient fallbacks."""
    rating_file = get_rating_config_filepath()
    if os.path.exists(rating_file):
        with open(rating_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"coefficients": {"Kv": 0.5, "Ke": 0.5, "Kb": 1.0}}

_load_rating_config_safely = load_rating_config_direct


def calculate_trapezoid_direct(
    r: float, 
    V: int = 0, 
    E: int = 0, 
    bias: str = "neutral", 
    coeffs: Optional[Dict[str, float]] = None
) -> List[float]:
    """Calculates trapezoidal fuzzy number [a, b, c, d] supporting 'opt'/'optimistic' and 'pes'/'pessimistic'."""
    c = coeffs or {"Kv": 0.5, "Ke": 0.5, "Kb": 1.0}
    kv = float(c.get("Kv", 0.5))
    ke = float(c.get("Ke", 0.5))
    kb = float(c.get("Kb", 1.0))

    delta_v = kv * float(V)
    delta_e = ke * float(E)

    a = max(0.0, r - delta_v - delta_e)
    b = r
    c_pt = r
    d = min(10.0, r + delta_v + delta_e)

    b_str = str(bias).strip().lower()
    if b_str in ["opt", "optimistic"]:
        b = min(10.0, r + (delta_v * 0.5 * kb))
    elif b_str in ["pes", "pessimistic"]:
        c_pt = max(0.0, r - (delta_v * 0.5 * kb))

    # Enforce trapezoid ordering invariant a <= b <= c <= d
    b = max(a, min(b, d))
    c_pt = max(b, min(c_pt, d))

    return [round(float(x), 4) for x in [a, b, c_pt, d]]


# =========================================================================
# DISPATCHER CLASS
# =========================================================================
class AnalysisDispatcher:
    """Non-destructive in-memory baseline loader and MCDM calculation dispatcher."""
    @staticmethod
    def rebalance_criteria_weights(
        target_criterion_id: str,
        new_weight: float,
        baseline_global_weights: Dict[str, float],
        factors_config: Dict[str, Any]
    ) -> np.ndarray:
        """Proportionally scales remaining criteria weights and normalizes them to sum to 1.0 (Single Flat Mode)."""
        new_w = float(np.clip(new_weight, 0.0, 1.0))
        factors = factors_config.get("factors", [])
        
        other_fids = [f["id"] for f in factors if f["id"] != target_criterion_id]
        sum_other_base = sum(float(baseline_global_weights.get(fid, 0.0)) for fid in other_fids)
        
        new_global_weights = {target_criterion_id: new_w}
        remaining_mass = 1.0 - new_w
        
        for fid in other_fids:
            if sum_other_base > 0:
                base_w = float(baseline_global_weights.get(fid, 0.0))
                new_global_weights[fid] = (base_w / sum_other_base) * remaining_mass
            else:
                new_global_weights[fid] = remaining_mass / len(other_fids) if other_fids else 0.0
                
        derived_global_w = np.zeros(len(factors), dtype=float)
        for j, f in enumerate(factors):
            derived_global_w[j] = new_global_weights.get(f["id"], 0.0)
            
        tot = np.sum(derived_global_w)
        if tot > 0:
            derived_global_w /= tot

        return derived_global_w
    
    @staticmethod
    def _ensure_analysis_dir():
        """Ensures that the active project's analysis runs directory exists."""
        analysis_runs_dir = get_analysis_runs_dir()
        if not os.path.exists(analysis_runs_dir):
            os.makedirs(analysis_runs_dir)

    @staticmethod
    def list_saved_runs() -> List[Dict[str, Any]]:
        """Lists all saved MCDM baseline runs from the active project's runs directory."""
        runs_dir = get_runs_dir()
        if not os.path.exists(runs_dir):
            return []
        runs = []
        for file in os.listdir(runs_dir):
            if file.endswith('.json'):
                path = os.path.join(runs_dir, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        runs.append({
                            "run_id": data.get("run_id", file.replace(".json", "")),
                            "name": data.get("name", "Unnamed Run"),
                            "timestamp": data.get("timestamp", ""),
                            "countries": data.get("countries", []),
                            "has_snapshot": "snapshot" in data,
                            "file_path": path
                        })
                except (json.JSONDecodeError, IOError):
                    continue
        return sorted(runs, key=lambda x: x["timestamp"], reverse=True)

    @staticmethod
    def load_baseline_run(run_id: str) -> Dict[str, Any]:
        """Loads a saved MCDM run JSON file from the active project directory."""
        filepath = os.path.join(get_runs_dir(), f"{run_id}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Saved run with ID '{run_id}' not found in active project.")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def build_in_memory_context(baseline_run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Builds in-memory data structures. Prioritizes historical snapshots when available,
        falling back directly to active project JSON files.
        """
        baseline_run = None
        if baseline_run_id:
            try:
                baseline_run = AnalysisDispatcher.load_baseline_run(baseline_run_id)
            except FileNotFoundError:
                baseline_run = None

        snapshot = baseline_run.get("snapshot") if isinstance(baseline_run, dict) else None

        if snapshot:
            factors_cfg = snapshot.get("factors_config", {})
            weights_data = snapshot.get("weights", {})
            evaluations = snapshot.get("evaluations", [])
            countries = (baseline_run.get("countries", []) if isinstance(baseline_run, dict) else []) or []
            types_list = [parse_factor_type(t) for t in snapshot.get("types", [])]
        else:
            factors_cfg = load_factors_config()
            weights_data = load_weights_direct()
            evaluations = load_evaluations_direct()
            countries = (baseline_run.get("countries", []) if isinstance(baseline_run, dict) else []) or sorted(list({e['country'] for e in evaluations}))
            types_list = []

        domains = factors_cfg.get("domains", [])
        factors = factors_cfg.get("factors", [])
        criteria_ids = [f["id"] for f in factors]

        if not types_list:
            types_list = [parse_factor_type(f.get("type", 1)) for f in factors]
        types_arr = np.array(types_list, dtype=int)

        raw_cat_weights = weights_data.get("category_weights", {})
        category_weights = {}
        for d in domains:
            d_id = d["id"]
            d_name = d["name"]
            if d_id in raw_cat_weights:
                category_weights[d_id] = float(raw_cat_weights[d_id])
            elif d_name in raw_cat_weights:
                category_weights[d_id] = float(raw_cat_weights[d_name])
            else:
                category_weights[d_id] = 1.0 / len(domains) if domains else 1.0

        local_weights = weights_data.get("local_weights", {})
        global_weights = weights_data.get("global_weights", {})

        if not global_weights:
            global_weights = {}
            for f in factors:
                f_id = f["id"]
                d_id = f["domain_id"]
                c_w = category_weights.get(d_id, 0.0)
                l_w = local_weights.get(f_id, 0.0)
                global_weights[f_id] = c_w * l_w

        num_c = len(countries)
        num_f = len(factors)
        matrix = np.zeros((num_c, num_f), dtype=float)
        fuzzy_matrix = np.zeros((num_c, num_f, 4), dtype=float)

        for i, country in enumerate(countries):
            for j, f_id in enumerate(criteria_ids):
                ev = next((e for e in evaluations if e["country"] == country and e["criterion_id"] == f_id), None)
                if ev:
                    matrix[i, j] = float(ev.get("rating", 5.0))
                    trap = ev.get("trapezoid", [matrix[i, j]] * 4)
                    fuzzy_matrix[i, j, :] = [float(x) for x in trap]
                else:
                    matrix[i, j] = 5.0
                    fuzzy_matrix[i, j, :] = [5.0, 5.0, 5.0, 5.0]

        return {
            "countries": countries,
            "domains": domains,
            "factors": factors,
            "criteria_ids": criteria_ids,
            "category_weights": category_weights,
            "local_weights": local_weights,
            "global_weights": global_weights,
            "matrix": matrix,
            "fuzzy_matrix": fuzzy_matrix,
            "types": types_arr,
            "factors_config": factors_cfg,
            "evaluations": evaluations,
            "is_from_snapshot": snapshot is not None
        }

    @staticmethod
    def rebalance_category_weights(
        target_domain_id: str,
        new_weight: float,
        baseline_cat_weights: Dict[str, float],
        local_weights: Dict[str, float],
        factors_config: Dict[str, Any]
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Proportionally scales remaining category weights and recalculates global criteria weights."""
        new_w = float(np.clip(new_weight, 0.0, 1.0))
        domains = factors_config.get("domains", [])
        factors = factors_config.get("factors", [])

        other_domains = [d["id"] for d in domains if d["id"] != target_domain_id]
        sum_other_base = sum(float(baseline_cat_weights.get(d_id, 0.0)) for d_id in other_domains)

        rebalanced_cat_weights = {target_domain_id: new_w}
        remaining_mass = 1.0 - new_w

        for d_id in other_domains:
            if sum_other_base > 0:
                base_w = float(baseline_cat_weights.get(d_id, 0.0))
                rebalanced_cat_weights[d_id] = (base_w / sum_other_base) * remaining_mass
            else:
                rebalanced_cat_weights[d_id] = remaining_mass / len(other_domains) if other_domains else 0.0

        derived_global_w = np.zeros(len(factors), dtype=float)
        for j, f in enumerate(factors):
            f_id = f["id"]
            d_id = f["domain_id"]
            c_weight = rebalanced_cat_weights.get(d_id, 0.0)
            l_weight = float(local_weights.get(f_id, 0.0))
            derived_global_w[j] = c_weight * l_weight

        tot = np.sum(derived_global_w)
        if tot > 0:
            derived_global_w /= tot

        return rebalanced_cat_weights, derived_global_w

    @staticmethod
    def regenerate_fuzzy_matrix(
        evaluations: List[Dict[str, Any]],
        factors: List[Dict[str, Any]],
        countries: List[str],
        coeffs: Dict[str, float]
    ) -> np.ndarray:
        """Regenerates the in-memory fuzzy matrix under modified Kv, Ke, or Kb multipliers."""
        num_c = len(countries)
        num_f = len(factors)
        fuzzy_mat = np.zeros((num_c, num_f, 4), dtype=float)
        criteria_ids = [f["id"] for f in factors]

        for i, country in enumerate(countries):
            for j, f_id in enumerate(criteria_ids):
                ev = next((e for e in evaluations if e["country"] == country and e["criterion_id"] == f_id), None)
                if ev:
                    r = float(ev.get("rating", 5.0))
                    v = int(ev.get("volatility", ev.get("V", 0)))
                    e_val = int(ev.get("uncertainty", ev.get("E", 0)))
                    bias = str(ev.get("bias", "neutral"))
                    trap = calculate_trapezoid_direct(r=r, V=v, E=e_val, bias=bias, coeffs=coeffs)
                    fuzzy_mat[i, j, :] = trap
                else:
                    fuzzy_mat[i, j, :] = [5.0, 5.0, 5.0, 5.0]

        return fuzzy_mat

    @staticmethod
    def execute_method_iteration(
        method_name: str,
        matrix: np.ndarray,
        fuzzy_matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        parameters: Dict[str, Any],
        countries: List[str]
    ) -> Dict[str, Any]:
        """Executes a single in-memory MCDM method iteration on METHOD_REGISTRY class instances."""
        if method_name not in METHOD_REGISTRY:
            return {"status": "error", "message": f"Method '{method_name}' not found in registry."}

        method = METHOD_REGISTRY[method_name]
        method_type = getattr(method, "method_type", "deterministic")
        is_fuzzy = (method_type == "fuzzy")

        try:
            if is_fuzzy:
                # Format 3D numpy fuzzy matrix into list of lists of 4-tuples for fuzzy methods
                if isinstance(fuzzy_matrix, np.ndarray) and fuzzy_matrix.ndim == 3:
                    fuz_mat_input = [
                        [tuple(fuzzy_matrix[i, j, :].tolist()) for j in range(fuzzy_matrix.shape[1])]
                        for i in range(fuzzy_matrix.shape[0])
                    ]
                else:
                    fuz_mat_input = fuzzy_matrix

                res = method.execute(
                    matrix=fuz_mat_input,
                    weights=np.copy(weights),
                    types=np.copy(types),
                    parameters=dict(parameters)
                )
            else:
                res = method.execute(
                    matrix=np.copy(matrix),
                    weights=np.copy(weights),
                    types=np.copy(types),
                    parameters=dict(parameters)
                )

            if not isinstance(res, dict):
                return {"status": "error", "message": f"Method {method_name} returned non-dict result."}

            if res.get("status") in ["error", "not_implemented"]:
                return res

            scores = res.get("scores", [])
            ranks = res.get("ranking", [])
            winner_idx = int(np.argmin(ranks)) if len(ranks) > 0 else 0
            winner = countries[winner_idx] if countries and len(countries) > winner_idx else "N/A"

            return {
                "status": "success",
                "scores": [float(s) for s in scores],
                "ranking": [int(r) for r in ranks],
                "winner": winner,
                "winner_index": winner_idx
            }
        except Exception as ex:
            return {
                "status": "error",
                "message": str(ex)
            }

    @staticmethod
    def save_analysis_experiment(experiment_data: Dict[str, Any], custom_name: str = "") -> str:
        """Saves a sensitivity, epistemic, or Monte Carlo experiment into the active project's analysis runs directory."""
        AnalysisDispatcher._ensure_analysis_dir()
        import uuid
        from datetime import datetime

        timestamp = datetime.now().isoformat()
        exp_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        experiment_data["analysis_id"] = exp_id
        experiment_data["saved_name"] = custom_name.strip() if custom_name else f"Analysis_{exp_id[:16]}"
        experiment_data["saved_timestamp"] = timestamp

        save_path = os.path.join(get_analysis_runs_dir(), f"{exp_id}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=4)
            
        return exp_id

    @staticmethod
    def list_saved_analysis_experiments(filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists saved experiments from the active project. filter_type can be 'sensitivity', 'epistemic', 'monte_carlo', or None."""
        AnalysisDispatcher._ensure_analysis_dir()
        analysis_runs_dir = get_analysis_runs_dir()
        experiments = []
        if not os.path.exists(analysis_runs_dir):
            return experiments

        for file in os.listdir(analysis_runs_dir):
            if file.endswith('.json'):
                path = os.path.join(analysis_runs_dir, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        meta = data.get("metadata", {})
                        dim = meta.get("dimension", "")
                        analysis_type = meta.get("analysis_type", "")
                        
                        is_mc = ("monte_carlo" in analysis_type.lower() or "monte_carlo" in dim.lower() or "model_results" in data)
                        is_epistemic = ("epistemic" in analysis_type.lower() or "epistemic" in dim.lower() or "level1" in data or "level3" in data) and not is_mc
                        is_sensitivity = not is_mc and not is_epistemic

                        if filter_type == "sensitivity" and not is_sensitivity:
                            continue
                        if filter_type == "epistemic" and not is_epistemic:
                            continue
                        if filter_type == "monte_carlo" and not is_mc:
                            continue

                        experiments.append({
                            "analysis_id": data.get("analysis_id", file.replace(".json", "")),
                            "saved_name": data.get("saved_name", "Unnamed Analysis"),
                            "dimension": "Monte Carlo Simulation" if is_mc else (dim or analysis_type),
                            "baseline_run_name": meta.get("baseline_run_name", "Unknown"),
                            "baseline_run_id": meta.get("baseline_run_id", ""),
                            "saved_timestamp": data.get("saved_timestamp", ""),
                            "file_path": path
                        })
                except (json.JSONDecodeError, IOError):
                    continue
        return sorted(experiments, key=lambda x: x["saved_timestamp"], reverse=True)

    @staticmethod
    def load_saved_analysis_experiment(analysis_id: str) -> Dict[str, Any]:
        """Loads a saved analysis experiment JSON from the active project directory."""
        filepath = os.path.join(get_analysis_runs_dir(), f"{analysis_id}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Analysis experiment '{analysis_id}' not found in active project.")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def delete_saved_analysis_experiment(analysis_id: str):
        """Deletes a saved analysis experiment file from the active project directory."""
        filepath = os.path.join(get_analysis_runs_dir(), f"{analysis_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)