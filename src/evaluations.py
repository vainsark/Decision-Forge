"""
Decision Support System - Evaluations & Rating Data Layer
Handles collecting, validating, calculating fuzzy trapezoids, and persisting joint evaluations.
"""

import os
import json
from typing import Dict, List, Any, Tuple

from src.factors_manager import load_factors_config
from src.project_manager import get_active_project_dir

# ==========================================
# FILE PATHS & CONSTANTS
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dynamic path resolution functions for active project workspace isolation
def _get_project_data_dir() -> str:
    """Returns the active project directory, asserting it is not None."""
    proj_dir = get_active_project_dir()
    assert proj_dir is not None, "Active project directory is required."
    return proj_dir

def get_rating_config_filepath() -> str:
    return os.path.join(_get_project_data_dir(), 'rating_config.json')

def get_evaluations_filepath() -> str:
    return os.path.join(_get_project_data_dir(), 'evaluations.json')

# ==========================================
# CONFIGURATION MANAGEMENT
# ==========================================
def _ensure_data_dir():
    proj_dir = _get_project_data_dir()
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)

def load_rating_config() -> Dict[str, Any]:
    """Loads the dynamic rating coefficients and alternatives."""
    _ensure_data_dir()
    rating_config_path = get_rating_config_filepath()
    default_config = {
        "alternatives": ["Alternative 1", "Alternative 2"],
        "coefficients": {
            "Kv": 0.5,
            "Ke": 0.5,
            "Kb": 1.0,
            "Kv_numeric": 5.0,
            "Ke_numeric": 5.0,
            "Kb_numeric": 2.0
        },
        "promethee_q": 0.5,
        "promethee_p": 3.5,
        "promethee_pref_func": "vshape_2",
        "normalization_mode": "default",
        "normalization_ceiling": 10.0,
        "waspas_lambda": 0.5
    }
    
    if os.path.exists(rating_config_path):
        with open(rating_config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Merge defaults for any missing keys
            default_config.update(data)
            return default_config
            
    with open(rating_config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=4)
    return default_config

def save_rating_config(config: Dict[str, Any]):
    rating_config_path = get_rating_config_filepath()
    with open(rating_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

# ==========================================
# MATHEMATICAL CORE: TRAPEZOID CONSTRUCTION
# ==========================================
def calculate_trapezoid(rating: float, volatility: float, uncertainty: float, bias: str, coeffs: dict, criterion_id: str = None) -> tuple:
    """
    Calculates the fuzzy trapezoid (a, b, c, d) for an evaluation.
    Supports both absolute 0-10 scale/binary evaluations and percentage-relative numeric bounds.
    """
    r = float(rating)
    v = float(volatility)
    u = float(uncertainty)

    # Determine evaluation type if criterion_id is provided
    eval_type = "scale"
    if criterion_id:
        try:
            f_config = load_factors_config()
            factor = next((f for f in f_config.get("factors", []) if f["id"] == criterion_id), None)
            if factor:
                eval_type = factor.get("evaluation_type", "scale")
        except Exception:
            pass

    if eval_type == "numeric":
        # Extract numeric percentage coefficients (default 5%, 5%, 2%)
        kv_num = coeffs.get('Kv_numeric', 5.0) / 100.0
        ke_num = coeffs.get('Ke_numeric', 5.0) / 100.0
        kb_num = coeffs.get('Kb_numeric', 2.0) / 100.0

        u_spread = r * (u * ke_num)
        v_spread = r * (v * kv_num)

        a = r - u_spread - v_spread
        b = r - u_spread
        c = r + u_spread
        d = r + u_spread + v_spread

        kb_val = r * kb_num
        if bias == 'opt':
            a = min(a + kb_val, r)
            b = min(b + kb_val, r)
        elif bias == 'pes':
            c = max(c - kb_val, r)
            d = max(d - kb_val, r)

        a = max(0.0, a)
        b = max(a, b)
        c = max(b, c)
        d = max(c, d)

        return (a, b, c, d)

    else:
        # Scale or Binary (binary evaluations map to 0-10 scale)
        kv = coeffs.get('Kv', 0.5)
        ke = coeffs.get('Ke', 0.5)
        kb = coeffs.get('Kb', coeffs.get('bias_coefficient', 1.0))

        a = r - (u * ke) - (v * kv)
        b = r - (u * ke)
        c = r + (u * ke)
        d = r + (u * ke) + (v * kv)

        if bias == 'opt':
            a = min(a + kb, r)
            b = min(b + kb, r)
        elif bias == 'pes':
            c = max(c - kb, r)
            d = max(d - kb, r)

        a = max(0.0, min(10.0, a))
        b = max(0.0, min(10.0, b))
        c = max(0.0, min(10.0, c))
        d = max(0.0, min(10.0, d))

        b = max(a, b)
        c = max(b, c)
        d = max(c, d)

        return (a, b, c, d)

# ==========================================
# EVALUATION I/O & MOCK DATA INJECTION
# ==========================================
def load_evaluations(rating_config: Dict[str, Any], factors_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Loads joint evaluations. If missing, auto-generates neutral data for testing UI."""
    filepath = get_evaluations_filepath()
    
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    evals = []
    alternatives = rating_config.get("alternatives", rating_config.get("countries", []))
    coeffs = rating_config.get("coefficients", {})
    
    for f in factors_config.get("factors", []):
        for alt in alternatives:
            init_val = 10.0 if f.get("evaluation_type") == "binary" else (0.0 if f.get("evaluation_type") == "numeric" else 5.0)
            trap = calculate_trapezoid(init_val, 0, 0, "neutral", coeffs, criterion_id=f["id"])
            evals.append({
                "alternative": alt,
                "country": alt,  # Legacy key support
                "criterion_id": f["id"],
                "rating": init_val,
                "volatility": 0,
                "uncertainty": 0,
                "bias": "neutral",
                "coefficients": coeffs,
                "trapezoid": trap
            })
            
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(evals, f, indent=4)
    return evals

def save_evaluations(evaluations: List[Dict[str, Any]]):
    filepath = get_evaluations_filepath()
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(evaluations, f, indent=4)