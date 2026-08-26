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
            "Kb": 1.0
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
def calculate_trapezoid(rating: float, volatility: float, uncertainty: float, bias: str, coeffs: dict) -> tuple:
    """
    Calculates the fuzzy trapezoid (a, b, c, d) for an evaluation.

    Base (Neutral) Construction:
        a = r - E*Ke - V*Kv
        b = r - E*Ke
        c = r + E*Ke
        d = r + E*Ke + V*Kv

    Directional Bias:
        - Optimistic ('opt'): Shifts ONLY the lower bounds [a, b] toward r by Kb.
          Represents a mitigated downside (less lower-end risk).
        - Pessimistic ('pes'): Shifts ONLY the upper bounds [c, d] toward r by Kb.
          Represents a mitigated upside (less higher-end potential).
        * The opposite half remains completely unchanged.
        * The shifted values are capped so they never cross the center rating r.

    Args:
        rating (float): The center rating (r)
        volatility (float): Volatility score (V)
        uncertainty (float): Uncertainty score (E)
        bias (str): 'neutral', 'opt', or 'pes'
        coeffs (dict): Dictionary containing 'Kv', 'Ke', 'Kb' multipliers

    Returns:
        tuple: (a, b, c, d) bounded between 0 and 10.
    """
    r = float(rating)
    v = float(volatility)
    u = float(uncertainty)

    # Extract coefficients (with safe fallbacks)
    kv = coeffs.get('Kv', 0.5)
    ke = coeffs.get('Ke', 0.5)
    kb = coeffs.get('Kb', coeffs.get('bias_coefficient', 1.0))

    # 1. Construct the neutral base trapezoid
    a = r - (u * ke) - (v * kv)
    b = r - (u * ke)
    c = r + (u * ke)
    d = r + (u * ke) + (v * kv)

    # 2. Apply Directional Bias
    if bias == 'opt':
        # Shift lower bounds up, but do not let them cross the center rating `r`
        a = min(a + kb, r)
        b = min(b + kb, r)
    elif bias == 'pes':
        # Shift upper bounds down, but do not let them cross the center rating `r`
        c = max(c - kb, r)
        d = max(d - kb, r)

    # 3. Clip to the absolute 0-10 scale
    a = max(0.0, min(10.0, a))
    b = max(0.0, min(10.0, b))
    c = max(0.0, min(10.0, c))
    d = max(0.0, min(10.0, d))

    # 4. Enforce structural validity (a <= b <= c <= d)
    b = max(a, b)
    c = max(b, c)
    d = max(c, d)

    return (a, b, c, d)

# ==========================================
# EVALUATION I/O & MOCK DATA INJECTION
# ==========================================
def load_evaluations(rating_config: Dict[str, Any], factors_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Loads joint evaluations. If missing, auto-generates neutral (5,0,0) data for testing UI."""
    filepath = get_evaluations_filepath()
    
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    # Auto-generate mock data if file doesn't exist
    evals = []
    alternatives = rating_config.get("alternatives", rating_config.get("countries", []))
    coeffs = rating_config.get("coefficients", {})
    
    for f in factors_config.get("factors", []):
        for alt in alternatives:
            trap = calculate_trapezoid(5.0, 0, 0, "neutral", coeffs)
            evals.append({
                "alternative": alt,
                "country": alt,  # Legacy key support
                "criterion_id": f["id"],
                "rating": 5.0,
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