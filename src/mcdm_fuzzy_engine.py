"""
Isolated Decision Support System - Fuzzy MCDM Orchestrator
Handles validation, matrix building, and execution STRICTLY for fuzzy methods.
"""

import os
import json
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

from src.factors_manager import load_factors_config
from src.evaluations import get_evaluations_filepath, load_rating_config
from src.mcdm_methods import METHOD_REGISTRY
from src.project_manager import get_active_project_dir

# Dynamic path resolution functions for active project isolation
# Dynamic path resolution functions for active project workspace isolation
def _get_project_data_dir() -> str:
    """Returns the active project directory, asserting it is not None."""
    proj_dir = get_active_project_dir()
    assert proj_dir is not None, "Active project directory is required."
    return proj_dir

def get_runs_dir() -> str:
    return os.path.join(_get_project_data_dir(), 'runs')

def get_weights_filepath() -> str:
    return os.path.join(_get_project_data_dir(), 'weights.json')

def _ensure_dirs():
    """Ensures that the runs directory exists within the active project folder."""
    runs_dir = get_runs_dir()
    if not os.path.exists(runs_dir): os.makedirs(runs_dir)

def _build_fuzzy_matrices() -> Dict[str, Any]:
    """Builds fuzzy decision matrices from active project factors, weights, and evaluations."""
    factors_config = load_factors_config()
    factors = factors_config.get("factors", [])
    if not factors: raise ValueError("No factors defined.")
    
    weights_file = get_weights_filepath()
    with open(weights_file, 'r', encoding='utf-8') as f:
        weights_data = json.load(f)
    global_weights = weights_data.get("global_weights", {})
    
    domain_map = {d["id"]: d["name"] for d in factors_config.get("domains", [])}
    cat_weights_display = {domain_map.get(d_id, d_id): w * 100 for d_id, w in weights_data.get("category_weights", {}).items()}
    
    eval_path = get_evaluations_filepath()
    with open(eval_path, 'r', encoding='utf-8') as f:
        evaluations = json.load(f)
        
    countries = list(set([e["country"] for e in evaluations]))
    countries.sort()
    
    c_ids = [f["id"] for f in factors]
    
    fuzzy_matrix = np.empty((len(countries), len(factors)), dtype=object)
    weights_arr = np.zeros(len(factors))
    types_arr = np.zeros(len(factors))
    
    for j, fid in enumerate(c_ids):
        weights_arr[j] = global_weights.get(fid, 0.0)
        types_arr[j] = next((f["type"] for f in factors if f["id"] == fid), 1)
        
        for i, country in enumerate(countries):
            ev = next((e for e in evaluations if e["criterion_id"] == fid and e["country"] == country), None)
            if not ev: raise ValueError(f"Missing evaluation for {country} on {fid}.")
            fuzzy_matrix[i, j] = tuple(ev.get("trapezoid", [ev.get("rating", 0)]*4))
            
    w_sum = np.sum(weights_arr)
    if w_sum > 0: weights_arr = weights_arr / w_sum
            
    return {
        "countries": countries,
        "matrix": fuzzy_matrix.tolist(),
        "weights": weights_arr,
        "types": types_arr,
        "cat_weights_display": cat_weights_display,
        "raw_weights": weights_data,
        "raw_evaluations": evaluations,
        "raw_factors_config": factors_config,
        "criteria_ids": c_ids
    }

def execute_fuzzy_run(method_names: List[str], run_name: str):
    """Executes selected fuzzy MCDM methods and persists the snapshot to the active project's run directory."""
    try:
        data = _build_fuzzy_matrices()
    except ValueError as e:
        print(f"Fuzzy Validation Failed: {e}")
        return None
        
    rating_config = load_rating_config()
    parameters = {
        "defuzz_weights": rating_config.get("defuzz_weights", [0.15, 0.35, 0.35, 0.15]),
        "promethee_q": rating_config.get("promethee_q", 0.5),
        "promethee_p": rating_config.get("promethee_p", 3.5)
    }
    
    run_id = f"run_fuzzy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    results = {}
    
    for name in method_names:
        if name not in METHOD_REGISTRY: continue
        method = METHOD_REGISTRY[name]
        
        # Guardrail: Only run actual fuzzy methods here
        if method.method_type != "fuzzy": continue
        
        try:
            res = method.execute(
                matrix=data["matrix"], # Method receives trapezoids seamlessly
                weights=data["weights"], 
                types=data["types"], 
                parameters=parameters
            )
            res["method_type"] = "fuzzy"
            results[name] = res
        except Exception as e:
            results[name] = {"status": "error", "warnings": [str(e)], "method_type": "fuzzy"}

    # Filter evaluations to only the countries included in this run
    active_countries = data["countries"]
    filtered_evals = [e for e in data["raw_evaluations"] if e.get("country") in active_countries]

    run_snapshot = {
        "run_id": run_id,
        "name": run_name,
        "timestamp": datetime.now().isoformat(),
        "countries": active_countries,
        "category_weights": data["cat_weights_display"],
        "methods_executed": method_names,
        "parameters": parameters,
        "results": results,
        # --- Complete Historical Snapshot ---
        "snapshot": {
            "weights": data["raw_weights"],
            "evaluations": filtered_evals,
            "factors_config": data["raw_factors_config"],
            "criteria_ids": data["criteria_ids"],
            "types": [int(t) for t in data["types"]]
        }
    }
    
    _ensure_dirs()
    save_path = os.path.join(get_runs_dir(), f"{run_id}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(run_snapshot, f, indent=4)
        
    return run_snapshot