"""
Decision Support System - MCDM Method Registry
Provides a unified interface for deterministic and fuzzy MCDM methods.
"""

import numpy as np
from typing import Dict, Any, List
from pymcdm.methods import WSM, WPM, WASPAS, TOPSIS, VIKOR, PROMETHEE_II
from pymcdm.helpers import rankdata
from pymcdm.normalizations import sum_normalization, linear_normalization
from src.evaluations import load_rating_config


def _get_normalization_function(default_func):
    """
    Reads normalization mode from rating_config.json and returns the callable function.
    """
    try:
        config = load_rating_config()
    except Exception:
        config = {}
        
    mode = config.get("normalization_mode", "default")
    ceiling = float(config.get("normalization_ceiling", 10.0))
    
    if mode == "sum":
        return sum_normalization
    elif mode == "linear":
        return linear_normalization
    elif mode == "absolute":
        # Custom absolute ceiling normalization function
        def absolute_norm(x, cost=False):
            if cost:
                return (ceiling - x) / ceiling
            return x / ceiling
        return absolute_norm
    else:
        # Fall back to the method's native default
        return default_func

class BaseMCDM:
    name = "Base"
    method_type = "deterministic"
    
    def execute(self, matrix: np.ndarray, weights: np.ndarray, types: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Method must implement execute()")

# ==========================================
# DETERMINISTIC METHODS (via PyMCDM)
# ==========================================
class WSM_Method(BaseMCDM):
    name = "WSM"
    def execute(self, matrix, weights, types, parameters):
        norm_func = _get_normalization_function(sum_normalization)
        scores = WSM(normalization_function=norm_func)(matrix, weights, types)
        ranking = rankdata(scores, reverse=True) 
        return {"status": "success", "scores": np.array(scores).tolist(), "ranking": np.array(ranking).tolist()}

class WPM_Method(BaseMCDM):
    name = "WPM"
    def execute(self, matrix, weights, types, parameters):
        norm_func = _get_normalization_function(sum_normalization)
        scores = WPM(normalization_function=norm_func)(matrix, weights, types)
        ranking = rankdata(scores, reverse=True)
        return {"status": "success", "scores": np.array(scores).tolist(), "ranking": np.array(ranking).tolist()}

class WASPAS_Method(BaseMCDM):
    name = "WASPAS"
    def execute(self, matrix, weights, types, parameters):
        lmbd = parameters.get("WASPAS_lambda", 0.5)
        norm_func = _get_normalization_function(linear_normalization)
        scores = WASPAS(normalization_function=norm_func, l=lmbd)(matrix, weights, types)
        ranking = rankdata(scores, reverse=True)
        return {"status": "success", "scores": np.array(scores).tolist(), "ranking": np.array(ranking).tolist()}

class TOPSIS_Method(BaseMCDM):
    name = "TOPSIS"
    def execute(self, matrix, weights, types, parameters):
        scores = TOPSIS()(matrix, weights, types)
        ranking = rankdata(scores, reverse=True)
        return {"status": "success", "scores": np.array(scores).tolist(), "ranking": np.array(ranking).tolist()}

class PROMETHEE_II_Method(BaseMCDM):
    name = "PROMETHEE II"
    def execute(self, matrix, weights, types, parameters):
        mat_arr = np.array(matrix)
        num_crit = mat_arr.shape[1] if mat_arr.ndim == 2 else len(mat_arr[0])
        
        # Pull preference function and thresholds from parameters (with robust fallbacks)
        pref_func_name = parameters.get("promethee_pref_func")
        q_val = float(parameters.get("promethee_q", 0.5))
        p_val = float(parameters.get("promethee_p", 3.5))
        
        # Broadcast scalar thresholds into criteria-matching arrays
        q = np.full(num_crit, q_val)
        p = np.full(num_crit, p_val)
        
        # Initialize PyMCDM's PROMETHEE II with the user-selected preference function
        promethee = PROMETHEE_II(pref_func_name, q=q, p=p)
        scores = promethee(mat_arr, weights, types)
        ranking = rankdata(scores, reverse=True)
        
        return {
            "status": "success", 
            "scores": np.array(scores).tolist(), 
            "ranking": np.array(ranking).tolist()
        }


class VIKOR_Method(BaseMCDM):
    name = "VIKOR"
    def execute(self, matrix, weights, types, parameters):
        scores = VIKOR()(matrix, weights, types)
        ranking = rankdata(scores, reverse=False) 
        return {"status": "success", "scores": np.array(scores).tolist(), "ranking": np.array(ranking).tolist()}

# ==========================================
# FUZZY METHODS (Placeholders)
# ==========================================
class FuzzyTOPSIS_Method(BaseMCDM):
    name = "Fuzzy TOPSIS (Trapezoidal)"
    method_type = "fuzzy"
    def execute(self, matrix, weights, types, parameters):
        return {
            "status": "not_implemented",
            "warnings": ["Fuzzy mathematics pending. Ready for backend implementation."]
        }

class FuzzyVIKOR_Method(BaseMCDM):
    name = "Fuzzy VIKOR (Trapezoidal)"
    method_type = "fuzzy"
    def execute(self, matrix, weights, types, parameters):
        return {
            "status": "not_implemented",
            "warnings": ["Fuzzy mathematics pending. Ready for backend implementation."]
        }

class FuzzyPROMETHEE:
    def __init__(self):
        self.name = "Fuzzy PROMETHEE (Net Flow)"
        self.method_type = "fuzzy"  

    def preference_mapping(self, diff: float, q: float = 0.5, p: float = 3.5) -> float:
        """
        Symmetric Type V PROMETHEE Preference Function.
        diff: Defuzzified score difference (on a -10 to +10 scale)
        q: Indifference threshold
        p: Absolute preference threshold
        """
        # Safety guard to prevent division by zero
        if p <= q:
            p = q + 1e-6

        sign = np.sign(diff)
        abs_diff = abs(diff)
        
        if abs_diff <= q:
            return 0.0
        elif abs_diff >= p:
            return float(sign * 1.0)
        else:
            pref = (abs_diff - q) / (p - q)
            return float(sign * pref)

    def _sub_trapezoid(self, t1: tuple, t2: tuple, q: float, p: float) -> list:
        # Cross-tail subtraction: (a1-d2, b1-c2, c1-b2, d1-a2)
        diff_trap = (t1[0] - t2[3], t1[1] - t2[2], t1[2] - t2[1], t1[3] - t2[0])
        return [self.preference_mapping(v, q=q, p=p) for v in diff_trap]

    def execute(self, matrix, weights, types=None, parameters=None):
        num_alt = len(matrix)
        num_crit = len(matrix[0])
        
        # Load parameters with sensible defaults
        params = parameters or {}
        w_trap = params.get("defuzz_weights", [1/6, 2/6, 2/6, 1/6])
        q = float(params.get("promethee_q", 0.5))
        p = float(params.get("promethee_p", 3.5))

        net_flows = np.zeros(num_alt)
        
        for i in range(num_alt):
            for k in range(num_alt):
                if i == k: 
                    continue
                
                pair_flow = 0.0
                for j in range(num_crit):
                    pref_trap = self._sub_trapezoid(matrix[i][j], matrix[k][j], q=q, p=p)
                    net_pref = sum(pv * wv for pv, wv in zip(pref_trap, w_trap))
                    pair_flow += net_pref * weights[j]
                    
                net_flows[i] += pair_flow
        
        if num_alt > 1:
            net_flows /= (num_alt - 1)
            
        ranks = (-net_flows).argsort().argsort() + 1
        
        return {
            "status": "success",
            "scores": net_flows.tolist(),
            "ranking": ranks.tolist()
        }

# ==========================================
# REGISTRY
# ==========================================
METHOD_REGISTRY = {
    "WSM": WSM_Method(),
    "WPM": WPM_Method(),
    "WASPAS": WASPAS_Method(),
    "TOPSIS": TOPSIS_Method(),
    "PROMETHEE II": PROMETHEE_II_Method(),
    # "VIKOR": VIKOR_Method(),
    # "Fuzzy TOPSIS": FuzzyTOPSIS_Method(),
    # "Fuzzy VIKOR": FuzzyVIKOR_Method(),
    "Fuzzy PROMETHEE": FuzzyPROMETHEE()
}