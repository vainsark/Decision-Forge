"""
Decision Support System - Hybrid Weight Engine
Uses PyMCDM native AHP and Normalization tools to handle weights.
Includes PyMCDM Float Sanitization and a Consistency Finder.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Tuple, List, Any

from pymcdm.weights.subjective import AHP
from pymcdm.normalizations import sum_normalization

# --- NEW PATH FIX: Ensure Python knows where the project root is ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.factors_manager import load_factors_config
from src.project_manager import get_active_project_dir

# ==========================================
# DYNAMIC FILE PATHS & PYMCDM CONSTANTS
# ==========================================
# Dynamic path resolution functions for active project workspace isolation
def _get_project_data_dir() -> str:
    """Returns the active project directory, asserting it is not None."""
    proj_dir = get_active_project_dir()
    assert proj_dir is not None, "Active project directory is required."
    return proj_dir

def get_ahp_filepath() -> str:
    """Returns the path to the AHP matrix file for the active project."""
    return os.path.join(_get_project_data_dir(), 'ahp_matrices.json')

def get_ratings_filepath() -> str:
    """Returns the path to the criteria ratings file for the active project."""
    return os.path.join(_get_project_data_dir(), 'criteria_ratings.json')

def get_weights_filepath() -> str:
    """Returns the path to the weights calculation bundle for the active project."""
    return os.path.join(_get_project_data_dir(), 'weights.json')

# PyMCDM strictly enforces these exact floats for fractions
# Create the exact mathematical floats PyMCDM demands (e.g. 0.3333333333333333 instead of 0.33)
EXACT_AHP_VALUES = [float(x) for x in range(1, 10)] + [1.0 / x for x in range(2, 10)]

def _ensure_data_dir():
    """Ensures that the active project directory exists on the file system."""
    proj_dir = _get_project_data_dir()
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)

def _sanitize_ahp_value(val: float) -> float:
    """Snaps any float to the exact PyMCDM mathematical AHP floats to prevent ValueError crashes."""
    return min(EXACT_AHP_VALUES, key=lambda x: abs(x - val))

# ==========================================
# INITIALIZATION & LOADING
# ==========================================
def load_ahp_matrix(domains: List[Dict]) -> np.ndarray:
    """Loads AHP matrix or returns identity matrix if missing."""
    n = len(domains)
    _ensure_data_dir()
    ahp_file = get_ahp_filepath()
    
    # Check if the active project already has a saved AHP matrix file
    if os.path.exists(ahp_file):
        with open(ahp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "domain_matrix" in data:
                mat = np.array(data["domain_matrix"])
                v_sanitize = np.vectorize(_sanitize_ahp_value)
                mat = v_sanitize(mat)
                if mat.shape == (n, n):
                    return mat
                    
    # Default fallback (identity matrix)
    mat = np.eye(n)
    with open(ahp_file, 'w', encoding='utf-8') as f:
         json.dump({"domain_matrix": mat.tolist()}, f, indent=4)
         
    return mat

def load_criteria_ratings(config: Dict[str, Any]) -> Dict[str, float]:
    """Loads 1-10 criteria ratings. Injects defaults for missing new factors or missing file."""
    _ensure_data_dir()
    factors = config.get("factors", [])
    expected_fids = [f["id"] for f in factors]
    ratings_file = get_ratings_filepath()
    
    # Check if criteria ratings file exists for this project
    if os.path.exists(ratings_file):
        with open(ratings_file, 'r', encoding='utf-8') as f:
            ratings = json.load(f)
            
        # Check if any new factors were added to config but are missing from saved ratings
        missing_keys = [fid for fid in expected_fids if fid not in ratings]
        if missing_keys:
            for fid in missing_keys:
                ratings[fid] = 5.0  # Assign a safe default rating of 5.0 for new factors
            # Resave the repaired file
            with open(ratings_file, 'w', encoding='utf-8') as f:
                json.dump(ratings, f, indent=4)
                
        return ratings
            
    # Default fallback (5.0 for all factors)
    ratings = {f["id"]: 5.0 for f in factors}
    with open(ratings_file, 'w', encoding='utf-8') as f:
        json.dump(ratings, f, indent=4)
        
    return ratings

def save_state(matrix: np.ndarray, ratings: Dict[str, float], weights: Dict[str, Any]):
    """Persists the AHP matrix, criteria ratings, and weight bundles to project files."""
    _ensure_data_dir()
    with open(get_ahp_filepath(), 'w', encoding='utf-8') as f:
        json.dump({"domain_matrix": matrix.tolist()}, f, indent=4)
    with open(get_ratings_filepath(), 'w', encoding='utf-8') as f:
        json.dump(ratings, f, indent=4)
    with open(get_weights_filepath(), 'w', encoding='utf-8') as f:
        json.dump(weights, f, indent=4)

# ==========================================
# PYMCDM MATHEMATICAL ENGINE
# ==========================================
def calculate_ahp_weights(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """Uses pymcdm's AHP module to get weights and Consistency Ratio."""
    # Step 1: Snap to PyMCDM's exact allowed floats
    v_sanitize = np.vectorize(_sanitize_ahp_value)
    safe_matrix = v_sanitize(matrix)
    
    # Step 2: Mathematically force perfect reciprocals so PyMCDM's validator doesn't crash
    n = safe_matrix.shape[0]
    for i in range(n):
        safe_matrix[i, i] = 1.0  # Diagonal must be exactly 1.0
        for j in range(i + 1, n):
            safe_matrix[j, i] = 1.0 / safe_matrix[i, j]
            
    # Step 3: Pass perfectly sanitized matrix to PyMCDM
    ahp_model = AHP(matrix=safe_matrix)
    weights = ahp_model()
    try:
        cr = ahp_model.get_cr()
    except ValueError:
        cr = 0.0  # Fallback just in case N <= 2
    return weights, float(cr)

def get_worst_inconsistency(matrix: np.ndarray, weights: np.ndarray) -> Tuple[int, int, float, float]:
    """Manual function to find the exact cell that contradicts the final weights the most."""
    n = len(weights)
    max_dev = 0.0
    worst_i, worst_j = 0, 1
    actual_val, expected_ratio = 1.0, 1.0
    
    # Compare every pair in the matrix to find highest deviation from expected ratio
    for i in range(n):
        for j in range(i+1, n):
            expected = weights[i] / weights[j]
            actual = matrix[i, j]
            dev = max(expected/actual, actual/expected)
            if dev > max_dev:
                max_dev = dev
                worst_i, worst_j = i, j
                actual_val, expected_ratio = actual, expected
                
    return worst_i, worst_j, actual_val, expected_ratio

def calculate_local_weights(ratings: Dict[str, float], config: Dict[str, Any]) -> Dict[str, float]:
    """Uses pymcdm.normalizations to convert 1-10 ratings to local weights."""
    local_weights = {}
    
    for domain in config.get("domains", []):
        domain_id = domain["id"]
        domain_factors = [f["id"] for f in config.get("factors", []) if f["domain_id"] == domain_id]
        
        if not domain_factors:
            continue
            
        ratings_list = [ratings[fid] for fid in domain_factors]
        ratings_array = np.array(ratings_list, dtype=float)
        
        # Normalize ratings using PyMCDM sum normalization
        norm_array = sum_normalization(ratings_array, cost=False)
        
        for idx, fid in enumerate(domain_factors):
            local_weights[fid] = float(norm_array[idx])
            
    return local_weights

def calculate_global_weights(cat_weights: Dict[str, float], local_weights: Dict[str, float], config: Dict[str, Any]) -> Dict[str, float]:
    """Multiplies category weights by local weights to yield global criteria weights."""
    global_weights = {}
    for factor in config.get("factors", []):
        fid = factor["id"]
        did = factor["domain_id"]
        global_weights[fid] = cat_weights.get(did, 0.0) * local_weights.get(fid, 0.0)
    return global_weights

# ==========================================
# PUBLIC API FOR STREAMLIT UI
# ==========================================
def load_or_initialize_weights() -> Dict[str, Any]:
    """Orchestrates loading factors, AHP matrix, and ratings, then computes final weights bundle."""
    config = load_factors_config()
    domains = config.get("domains", [])
    
    matrix = load_ahp_matrix(domains)
    ratings = load_criteria_ratings(config)
    
    cat_w_array, cr = calculate_ahp_weights(matrix)
    cat_weights = {domains[i]["id"]: float(cat_w_array[i]) for i in range(len(domains))}
    loc_weights = calculate_local_weights(ratings, config)
    glob_weights = calculate_global_weights(cat_weights, loc_weights, config)
    
    weights_bundle = {
        "category_weights": cat_weights,
        "local_weights": loc_weights,
        "global_weights": glob_weights,
        "cr_status": cr
    }
    
    save_state(matrix, ratings, weights_bundle)
    return weights_bundle

def get_global_weights() -> Dict[str, float]:
    """Convenience wrapper to quickly retrieve global weights dictionary."""
    weights = load_or_initialize_weights()
    return weights["global_weights"]