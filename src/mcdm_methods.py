"""
Decision Support System - MCDM Method Registry
Provides a unified interface for deterministic and fuzzy MCDM methods.
"""

import numpy as np
from typing import Dict, Any, List
from pymcdm.methods import WSM, WPM, WASPAS, TOPSIS, VIKOR
from pymcdm.helpers import rankdata
from pymcdm.normalizations import linear_normalization

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
        scores = WSM()(matrix, weights, types)
        ranking = rankdata(scores, reverse=True) 
        return {"status": "success", "scores": np.array(scores).tolist(), "ranking": np.array(ranking).tolist()}

class WPM_Method(BaseMCDM):
    name = "WPM"
    def execute(self, matrix, weights, types, parameters):
        scores = WPM()(matrix, weights, types)
        ranking = rankdata(scores, reverse=True)
        return {"status": "success", "scores": np.array(scores).tolist(), "ranking": np.array(ranking).tolist()}

class WASPAS_Method(BaseMCDM):
    name = "WASPAS"
    def execute(self, matrix, weights, types, parameters):
        lmbd = parameters.get("WASPAS_lambda", 0.5)
        # Pass the normalization function explicitly to properly inject custom lambda
        scores = WASPAS(normalization_function=linear_normalization, l=lmbd)(matrix, weights, types)
        ranking = rankdata(scores, reverse=True)
        return {"status": "success", "scores": np.array(scores).tolist(), "ranking": np.array(ranking).tolist()}

class TOPSIS_Method(BaseMCDM):
    name = "TOPSIS"
    def execute(self, matrix, weights, types, parameters):
        scores = TOPSIS()(matrix, weights, types)
        ranking = rankdata(scores, reverse=True)
        return {"status": "success", "scores": np.array(scores).tolist(), "ranking": np.array(ranking).tolist()}

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
    # "VIKOR": VIKOR_Method(),
    # "Fuzzy TOPSIS": FuzzyTOPSIS_Method(),
    # "Fuzzy VIKOR": FuzzyVIKOR_Method(),
    "Fuzzy PROMETHEE": FuzzyPROMETHEE()
}