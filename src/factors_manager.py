"""
Decision Support System - Factors Manager
Handles loading metadata, saving metadata, intelligent ID generation, Cascading Deletions,
and Ghost Category management for Single Flat Weighting mode.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional
from src.project_manager import get_active_project_dir

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dynamic path resolution functions for active project workspace isolation
def _get_project_data_dir() -> str:
    """Returns the active project directory, asserting it is not None."""
    proj_dir = get_active_project_dir()
    assert proj_dir is not None, "Active project directory is required."
    return proj_dir

def get_factors_filepath() -> str:
    return os.path.join(_get_project_data_dir(), 'factors_config.json')

def get_weights_filepath() -> str:
    return os.path.join(_get_project_data_dir(), 'weights.json')

def get_evaluations_filepath() -> str:
    return os.path.join(_get_project_data_dir(), 'evaluations.json')

def load_factors_config() -> Dict[str, Any]:
    factors_file = get_factors_filepath()
    if not os.path.exists(factors_file):
        return {"domains": [], "factors": []}
    with open(factors_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_factors_config(config: Dict[str, Any]):
    """Saves the config while ensuring domains and factors are strictly sorted by ID."""
    if "domains" in config:
        config["domains"] = sorted(config["domains"], key=lambda x: x["id"])
    if "factors" in config:
        config["factors"] = sorted(config["factors"], key=lambda x: x["id"])
        
    proj_dir = _get_project_data_dir()
    os.makedirs(proj_dir, exist_ok=True)
    factors_file = get_factors_filepath()
    with open(factors_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

# ==========================================
# GHOST CATEGORY MANAGEMENT (SINGLE FLAT MODE)
# ==========================================
def ensure_ghost_category():
    """Creates the ghost category (d01) for Single Flat Weighting mode and assigns all factors to it."""
    config = load_factors_config()
    domains = config.get("domains", [])
    factors = config.get("factors", [])
    
    has_d01 = any(d.get("id") == "d01" for d in domains)
    if not has_d01:
        ghost_dom = {
            "id": "d01",
            "name": "General Category",
            "short_name": "GEN",
            "description": "Ghost category for single flat weighting mode."
        }
        domains.insert(0, ghost_dom)
        config["domains"] = domains
        
    for f in factors:
        f["domain_id"] = "d01"
        
    save_factors_config(config)
    
    weights_file = get_weights_filepath()
    if os.path.exists(weights_file):
        try:
            with open(weights_file, 'r', encoding='utf-8') as wf:
                w_data = json.load(wf)
            w_data["category_weights"] = {"d01": 1.0}
            w_data["domain_ids"] = [d["id"] for d in config.get("domains", [])]
            with open(weights_file, 'w', encoding='utf-8') as wf:
                json.dump(w_data, wf, indent=4)
        except Exception:
            pass

def remove_ghost_category():
    """Removes the ghost category (d01) and purges flat-mode state when switching back to Dual Hybrid mode."""
    config = load_factors_config()
    factors = config.get("factors", [])
    
    # Only remove if no criteria exist
    if len(factors) == 0:
        config["domains"] = [d for d in config.get("domains", []) if d.get("id") != "d01"]
        save_factors_config(config)
        
        # Purge weights.json flat-mode residues
        weights_file = get_weights_filepath()
        if os.path.exists(weights_file):
            try:
                with open(weights_file, 'r', encoding='utf-8') as wf:
                    w_data = json.load(wf)
                w_data["category_weights"] = {}
                w_data["domain_ids"] = []
                w_data["ahp_matrix"] = []
                with open(weights_file, 'w', encoding='utf-8') as wf:
                    json.dump(w_data, wf, indent=4)
            except Exception:
                pass

# ==========================================
# INTELLIGENT ID GENERATORS (d01 -> c101)
# ==========================================
def get_next_domain_id(domains_list) -> str:
    max_idx = 0
    for d in domains_list:
        try:
            idx = int(''.join(filter(str.isdigit, d["id"])))
            max_idx = max(max_idx, idx)
        except ValueError:
            pass
    return f"d{max_idx + 1:02d}"

def get_next_criterion_id(domain_id: str, factors_list) -> str:
    try:
        prefix_num = int(''.join(filter(str.isdigit, domain_id)))
    except ValueError:
        prefix_num = 99
        
    prefix_str = f"c{prefix_num}"
    domain_factors = [f for f in factors_list if f.get("domain_id") == domain_id]
    max_seq = 0
    
    for f in domain_factors:
        fid = f["id"]
        if fid.startswith(prefix_str) and fid[len(prefix_str):].isdigit():
            seq = int(fid[len(prefix_str):])
            max_seq = max(max_seq, seq)
            
    return f"{prefix_str}{max_seq + 1:02d}"

# ==========================================
# ADD FACTORS
# ==========================================
def add_domain(name: str, description: str, short_name: str = "") -> str:
    config = load_factors_config()
    new_id = get_next_domain_id(config.get("domains", []))
    if not short_name: short_name = name[:5].upper()
        
    config.setdefault("domains", []).append({
        "id": new_id, "name": (name or "").strip(), "short_name": short_name.strip(), "description": (description or "").strip()
    })
    save_factors_config(config)
    return new_id

def add_criterion(domain_id: str, name: str, description: str, type_val: int, evaluation_type: str = "scale", unit: str = "", short_name: str = "") -> str:
    config = load_factors_config()
    new_id = get_next_criterion_id(domain_id, config.get("factors", []))
    if not short_name: short_name = name[:5].upper()
        
    config.setdefault("factors", []).append({
        "id": new_id, 
        "domain_id": domain_id, 
        "name": (name or "").strip(), 
        "short_name": short_name.strip(), 
        "description": (description or "").strip(), 
        "type": type_val,
        "evaluation_type": evaluation_type,
        "unit": unit.strip() if evaluation_type == "numeric" else ""
    })
    save_factors_config(config)
    return new_id

# ==========================================
# CASCADING DELETION LOGIC (SAFE ID PATCHING)
# ==========================================
def _rebuild_and_cascade_ids(config: Dict[str, Any], deleted_domain_id: Optional[str] = None, deleted_factor_id: Optional[str] = None):
    weights_file = get_weights_filepath()
    evaluations_file = get_evaluations_filepath()
    
    weights = {"category_weights": {}, "local_weights": {}, "global_weights": {}, "raw_ratings": {}, "ahp_matrix": [], "domain_ids": []}
    if os.path.exists(weights_file):
        with open(weights_file, 'r', encoding='utf-8') as f: weights = json.load(f)
            
    evals = []
    if os.path.exists(evaluations_file):
        with open(evaluations_file, 'r', encoding='utf-8') as f: evals = json.load(f)

    saved_matrix = weights.get("ahp_matrix", [])
    saved_domain_ids = weights.get("domain_ids", [])
    if not saved_domain_ids and weights.get("category_weights"):
        saved_domain_ids = list(weights["category_weights"].keys())

    old_cat_weights = {}
    if deleted_domain_id and saved_matrix:
        old_mat = np.array(saved_matrix)
        if deleted_domain_id in saved_domain_ids and old_mat.shape[0] == len(saved_domain_ids):
            del_idx = saved_domain_ids.index(deleted_domain_id)
            new_mat = np.delete(np.delete(old_mat, del_idx, axis=0), del_idx, axis=1)
            weights["ahp_matrix"] = new_mat.tolist()
            saved_domain_ids.pop(del_idx)
            
            n_mat = new_mat.shape[0]
            if n_mat <= 1:
                w_res = np.array([1.0]) if n_mat == 1 else np.array([])
            elif n_mat == 2:
                w_res = np.array([new_mat[0, 1] / (1.0 + new_mat[0, 1]), 1.0 / (1.0 + new_mat[0, 1])])
            else:
                evals_eig, evecs = np.linalg.eig(new_mat)
                max_idx = np.argmax(np.real(evals_eig))
                w_res = np.real(evecs[:, max_idx])
                w_res = np.abs(w_res) / np.sum(np.abs(w_res))
            
            old_cat_weights = {saved_domain_ids[idx]: float(w_res[idx]) for idx in range(n_mat)}
        else:
            weights["ahp_matrix"] = []
            
    if deleted_factor_id:
        for k in ["local_weights", "global_weights", "raw_ratings"]:
            if deleted_factor_id in weights.get(k, {}): del weights[k][deleted_factor_id]
        evals = [e for e in evals if e["criterion_id"] != deleted_factor_id]

    old_to_new_d = {}
    old_to_new_f = {}
    
    for i, d in enumerate(config.get("domains", [])):
        old_id = d["id"]
        new_id = f"d{i+1:02d}"
        old_to_new_d[old_id] = new_id
        d["id"] = new_id
        
    factors_by_domain = {}
    for f in config.get("factors", []):
        if f["domain_id"] in old_to_new_d: f["domain_id"] = old_to_new_d[f["domain_id"]]
        factors_by_domain.setdefault(f["domain_id"], []).append(f)
        
    for dom_id, dom_factors in factors_by_domain.items():
        prefix_num = int(''.join(filter(str.isdigit, dom_id)))
        prefix_str = f"c{prefix_num}"
        for j, f in enumerate(dom_factors):
            old_id = f["id"]
            new_id = f"{prefix_str}{j+1:02d}"
            old_to_new_f[old_id] = new_id
            f["id"] = new_id

    save_factors_config(config)
        
    new_cat_weights = {}
    source_weights = old_cat_weights if old_cat_weights else weights.get("category_weights", {})
    
    for old_k, v in source_weights.items():
        if old_k in old_to_new_d:
            new_cat_weights[old_to_new_d[old_k]] = v
            
    if not new_cat_weights:
        curr_domains = config.get("domains", [])
        equal_w = 1.0 / len(curr_domains) if curr_domains else 1.0
        for d in curr_domains:
            new_cat_weights[d["id"]] = equal_w

    weights["category_weights"] = new_cat_weights
    weights["domain_ids"] = [d["id"] for d in config.get("domains", [])]
    
    for dict_key in ["local_weights", "global_weights", "raw_ratings"]:
        new_dict = {}
        for old_k, v in weights.get(dict_key, {}).items():
            if old_k in old_to_new_f: new_dict[old_to_new_f[old_k]] = v
        weights[dict_key] = new_dict

    factors = config.get("factors", [])
    raw_ratings = weights.get("raw_ratings", {})
    local_weights = {}
    for d in config.get("domains", []):
        d_id = d["id"]
        d_factors = [f["id"] for f in factors if f["domain_id"] == d_id]
        domain_sum = sum(raw_ratings.get(fid, 0) for fid in d_factors)
        for fid in d_factors:
            local_weights[fid] = raw_ratings.get(fid, 0) / domain_sum if domain_sum > 0 else 0.0
    weights["local_weights"] = local_weights

    global_weights = {}
    for f in factors:
        c_w = new_cat_weights.get(f["domain_id"], 0.0)
        l_w = local_weights.get(f["id"], 0.0)
        global_weights[f["id"]] = c_w * l_w
    weights["global_weights"] = global_weights
        
    with open(weights_file, 'w', encoding='utf-8') as f: json.dump(weights, f, indent=4)
        
    if os.path.exists(evaluations_file):
        for e in evals:
            if e["criterion_id"] in old_to_new_f: e["criterion_id"] = old_to_new_f[old_to_new_f[e["criterion_id"]]] # safely mapped
        with open(evaluations_file, 'w', encoding='utf-8') as f: json.dump(evals, f, indent=4)

def delete_domain(domain_id: str):
    config = load_factors_config()
    attached = [f for f in config.get("factors", []) if f.get("domain_id") == domain_id]
    if attached: raise ValueError("Cannot delete category. It still has criteria attached to it.")
        
    config["domains"] = [d for d in config["domains"] if d["id"] != domain_id]
    _rebuild_and_cascade_ids(config, deleted_domain_id=domain_id)

def delete_criterion(criterion_id: str):
    config = load_factors_config()
    config["factors"] = [f for f in config["factors"] if f["id"] != criterion_id]
    _rebuild_and_cascade_ids(config, deleted_factor_id=criterion_id)