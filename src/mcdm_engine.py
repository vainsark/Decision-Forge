"""
Decision Support System - MCDM Orchestrator
Handles validation, execution, saving/loading runs, snapshot generation, and result viewing/comparing.
"""

import os
import json
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

from src.factors_manager import load_factors_config
from src.evaluations import get_evaluations_filepath
from src.mcdm_methods import METHOD_REGISTRY
from src.project_manager import get_active_project_dir

# ==========================================
# DYNAMIC FILE PATHS & CONSTANTS
# ==========================================
# Dynamic path resolution functions for active project workspace isolation
def _get_project_data_dir() -> str:
    """Returns the active project directory, asserting it is not None."""
    proj_dir = get_active_project_dir()
    assert proj_dir is not None, "Active project directory is required."
    return proj_dir

def get_runs_dir() -> str:
    """Returns the runs directory path inside the active project folder."""
    return os.path.join(_get_project_data_dir(), 'runs')

def get_rating_config_filepath() -> str:
    """Returns the path to the rating configuration file for the active project."""
    return os.path.join(_get_project_data_dir(), 'rating_config.json')

def get_weights_filepath() -> str:
    """Returns the path to the weights file for the active project."""
    return os.path.join(_get_project_data_dir(), 'weights.json')

# ==========================================
# CONFIG & STATE MANAGEMENT
# ==========================================
def _ensure_dirs():
    """Ensures that the project directories and runs subfolder exist."""
    proj_dir = _get_project_data_dir()
    runs_dir = get_runs_dir()
    if not os.path.exists(proj_dir): os.makedirs(proj_dir)
    if not os.path.exists(runs_dir): os.makedirs(runs_dir)

def load_engine_config() -> Dict[str, Any]:
    """Loads engine configuration parameters from the active project's rating config."""
    _ensure_dirs()
    rating_config_file = get_rating_config_filepath()
    default_config = {
        "waspas_lambda": 0.5,
        "promethee_q": 0.5,
        "promethee_p": 3.5,
        "promethee_pref_func": "vshape_2",
        "normalization_mode": "default",
        "normalization_ceiling": 10.0
    }
    if os.path.exists(rating_config_file):
        with open(rating_config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            default_config.update(data)
            return default_config
            
    with open(rating_config_file, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=4)
    return default_config

def save_engine_config(config: Dict[str, Any]):
    """Saves engine configuration parameters to the active project's rating config."""
    _ensure_dirs()
    rating_config_file = get_rating_config_filepath()
    if os.path.exists(rating_config_file):
        with open(rating_config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}
    data.update(config)
    with open(rating_config_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# ==========================================
# ENGINE DATA BUILDER & VALIDATION
# ==========================================
def _validate_and_build_matrices() -> Dict[str, Any]:
    """Validates factors, weights, and evaluations for the active project, building matrices for execution."""
    factors_config = load_factors_config()
    factors = factors_config.get("factors", [])
    if not factors: raise ValueError("No factors defined in factors_config.json")
    
    weights_file = get_weights_filepath()
    if not os.path.exists(weights_file):
        raise ValueError("Weights data missing. Please complete the Weight Engine.")
    with open(weights_file, 'r', encoding='utf-8') as f:
        weights_data = json.load(f)
    global_weights = weights_data.get("global_weights", {})
    
    # Extract domain percentages for the visual chart
    domain_map = {d["id"]: d["name"] for d in factors_config.get("domains", [])}
    cat_weights_display = {}
    for d_id, w in weights_data.get("category_weights", {}).items():
        if d_id in domain_map:
            cat_weights_display[domain_map[d_id]] = w * 100
        elif d_id in domain_map.values():
            cat_weights_display[d_id] = w * 100
    
    eval_path = get_evaluations_filepath()
    if not os.path.exists(eval_path):
        raise ValueError("Evaluations missing. Please complete the Rating System.")
    with open(eval_path, 'r', encoding='utf-8') as f:
        evaluations = json.load(f)
        
    alternatives = list(set([e.get("alternative", e.get("country", "")) for e in evaluations if e.get("alternative") or e.get("country")]))
    alternatives.sort()
    if len(alternatives) < 2:
        raise ValueError("Need evaluations for at least two alternatives to run MCDM.")

    c_ids = [f["id"] for f in factors]
    matrix = np.zeros((len(alternatives), len(factors)))
    weights_arr = np.zeros(len(factors))
    types_arr = np.zeros(len(factors))
    fuzzy_data = {alt: {} for alt in alternatives} 
    
    for j, fid in enumerate(c_ids):
        weights_arr[j] = global_weights.get(fid, 0.0)
        types_arr[j] = next((f["type"] for f in factors if f["id"] == fid), 1)
        
        for i, alt in enumerate(alternatives):
            ev = next((e for e in evaluations if e["criterion_id"] == fid and (e.get("alternative") == alt or e.get("country") == alt)), None)
            if not ev:
                raise ValueError(f"Missing evaluation for {alt} on criterion {fid}.")
            
            val = ev["rating"]
            matrix[i, j] = val if val > 0 else 0.0001
            fuzzy_data[alt][fid] = ev.get("trapezoid", [val]*4)
            
    w_sum = np.sum(weights_arr)
    if w_sum <= 0:
        raise ValueError("Weights sum to 0. Please recalculate weights.")
    weights_arr = weights_arr / w_sum
            
    return {
        "alternatives": alternatives,
        "countries": alternatives,  # Backwards compatibility
        "criteria_ids": c_ids,
        "matrix": matrix,
        "weights": weights_arr,
        "types": types_arr,
        "fuzzy_data": fuzzy_data,
        "cat_weights_display": cat_weights_display,
        "raw_factors_config": factors_config,
        "raw_weights": weights_data,
        "raw_evaluations": evaluations
    }

# ==========================================
# EXECUTION (WITH FULL SNAPSHOT PERSISTENCE)
# ==========================================
def execute_run(method_names: List[str], run_name: str):
    """Executes selected deterministic MCDM methods and saves the run snapshot to the active project folder."""
    data = _validate_and_build_matrices()
        
    config = load_engine_config()
    parameters = {
        "waspas_lambda": config.get("waspas_lambda", config.get("WASPAS_lambda", 0.5)),
        "WASPAS_lambda": config.get("waspas_lambda", config.get("WASPAS_lambda", 0.5)),
        "promethee_q": float(config.get("promethee_q", 0.1)),
        "promethee_p": float(config.get("promethee_p", 6.0)),
        "promethee_pref_func": config.get("promethee_pref_func", "vshape_2")
    }
    
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    results = {}
    for name in method_names:
        if name not in METHOD_REGISTRY: continue
        method = METHOD_REGISTRY[name]
        
        try:
            res = method.execute(
                matrix=data["matrix"], 
                weights=data["weights"], 
                types=data["types"], 
                parameters=parameters
            )
            res["method_type"] = method.method_type
            results[name] = res
        except Exception as e:
            results[name] = {"status": "error", "warnings": [str(e)], "method_type": method.method_type}

    active_alts = data["alternatives"]
    filtered_evals = [e for e in data["raw_evaluations"] if e.get("alternative") in active_alts or e.get("country") in active_alts]

    run_snapshot = {
        "run_id": run_id,
        "name": run_name,
        "timestamp": datetime.now().isoformat(),
        "alternatives": active_alts,
        "countries": active_alts,  # Backwards compatibility
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