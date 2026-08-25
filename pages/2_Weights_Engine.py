"""
Decision Support System - Weights Engine Page
Hybrid weighting engine supporting both Dual Hybrid (Categories -> Criteria AHP with full CR diagnostics and Inverse AHP) 
and Single Flat Weighting (Direct Criteria Pool) architectures.
"""

import streamlit as st
import pandas as pd
import numpy as np
import itertools
import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.factors_manager import load_factors_config
WEIGHTS_FILE = os.path.join(BASE_DIR, 'data', 'weights.json')
RATING_CONFIG_FILE = os.path.join(BASE_DIR, 'data', 'rating_config.json')

st.set_page_config(page_title="Weights Engine", page_icon="⚖️", layout="wide")

# ==========================================
# LOAD WEIGHT SYSTEM MODE
# ==========================================
def get_weight_system_mode() -> str:
    if os.path.exists(RATING_CONFIG_FILE):
        try:
            with open(RATING_CONFIG_FILE, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                return cfg.get("weight_system_mode", "Dual Hybrid (Categories & Criteria)")
        except Exception:
            pass
    return "Dual Hybrid (Categories & Criteria)"

weight_system_mode = get_weight_system_mode()

# ==========================================
# STATE MANAGEMENT & VALIDATION HELPERS
# ==========================================
def load_safe_weights():
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "domain_ids" not in data:
                data["domain_ids"] = []
            return data
    return {
        "category_weights": {}, 
        "local_weights": {}, 
        "global_weights": {}, 
        "raw_ratings": {}, 
        "ahp_matrix": [],
        "ahp_matrix_backup": None,
        "domain_ids": []
    }

def save_weights_state(new_weights):
    with open(WEIGHTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_weights, f, indent=4)

# ==========================================
# WEIGHT PRESET MANAGER (SIDEBAR)
# ==========================================
PRESET_WEIGHTS_DIR = os.path.join(BASE_DIR, 'data', 'weight_presets')
if not os.path.exists(PRESET_WEIGHTS_DIR):
    os.makedirs(PRESET_WEIGHTS_DIR)

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Weight Presets")

weight_presets = [f.replace(".json", "") for f in os.listdir(PRESET_WEIGHTS_DIR) if f.endswith(".json")]

weight_save_mode = st.sidebar.radio("Save Action:", ["Save as New", "Overwrite Existing"], key="weight_save_mode", horizontal=True)

if weight_save_mode == "Save as New" or not weight_presets:
    raw_new_w_name = st.sidebar.text_input("New Weight Preset:", value="My_Weight_Structure", key="weight_preset_input")
    new_weight_name = (raw_new_w_name or "My_Weight_Structure").strip()
    if st.sidebar.button("💾 Save New Weight Preset", key="save_weight_preset"):
        if new_weight_name:
            preset_path = os.path.join(PRESET_WEIGHTS_DIR, f"{new_weight_name}.json")
            current_weights = load_safe_weights()
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(current_weights, f, indent=4)
            st.sidebar.success(f"Saved weight preset '{new_weight_name}'!")
            st.rerun()
        else:
            st.sidebar.error("Enter a valid preset name.")
else:
    target_weight_preset = st.sidebar.selectbox("Select Weight Preset to Overwrite:", weight_presets, key="overwrite_weight_sel")
    if st.sidebar.button("⚠️ Overwrite Weight Preset", key="overwrite_weight_btn"):
        preset_path = os.path.join(PRESET_WEIGHTS_DIR, f"{target_weight_preset}.json")
        current_weights = load_safe_weights()
        with open(preset_path, 'w', encoding='utf-8') as f:
            json.dump(current_weights, f, indent=4)
        st.sidebar.success(f"Overwrote weight preset '{target_weight_preset}' successfully!")
        st.rerun()

if weight_presets:
    st.sidebar.markdown("---")
    selected_weight = st.sidebar.selectbox("Load Weight Preset:", weight_presets, key="load_weight_sel")
    if st.sidebar.button("📂 Load Weight Preset into App", key="load_weight_btn"):
        preset_path = os.path.join(PRESET_WEIGHTS_DIR, f"{selected_weight}.json")
        with open(preset_path, 'r', encoding='utf-8') as f:
            loaded_w = json.load(f)
        save_weights_state(loaded_w)
        st.sidebar.success(f"Loaded weight preset '{selected_weight}' successfully!")
        st.rerun()

config = load_factors_config()
weights = load_safe_weights()
domains = config.get("domains", [])
factors = config.get("factors", [])
num_domains = len(domains)

# ==============================================================================
# MODE A: SINGLE FLAT WEIGHTING (DIRECT CRITERIA POOL)
# ==============================================================================
if weight_system_mode == "Single Flat Weighting (Direct Criteria Pool)":
    st.title("⚖️ Weights Engine — Single Flat Weighting")
    st.caption("Active Mode: **Single Flat Weighting**. All criteria are evaluated in a unified pool and normalized directly[cite: 3].")

    if not factors:
        st.warning("No criteria found. Please add criteria in 'Criteria Overview' first.")
        st.stop()

    saved_ratings = weights.get("raw_ratings", {})
    missing_factors = [f for f in factors if f["id"] not in saved_ratings]
    
    if missing_factors or not weights.get("global_weights"):
        for f in factors:
            if f["id"] not in saved_ratings:
                saved_ratings[f["id"]] = 5.0
        weights["raw_ratings"] = saved_ratings
        
        total_sum = sum(saved_ratings.get(f["id"], 5.0) for f in factors)
        global_w = {}
        for f in factors:
            val = saved_ratings.get(f["id"], 5.0)
            global_w[f["id"]] = val / total_sum if total_sum > 0 else (1.0 / len(factors))
            
        weights["category_weights"] = {"General": 1.0}
        weights["local_weights"] = global_w
        weights["global_weights"] = global_w
        save_weights_state(weights)

    tab_flat_dash, tab_flat_edit = st.tabs(["📊 Flat Weights Dashboard", "🎯 Edit Criteria Weights"])

    with tab_flat_dash:
        st.header("Unified Criteria Weights")
        st.info("ℹ️ **Single Flat Weighting Mode:** Bypasses hierarchical categories. All active criteria are scaled directly against each other and normalized to 100% total global impact.")
        c_data = []
        domain_map = {d["id"]: d["name"] for d in domains}
        for f in factors:
            gw = weights["global_weights"].get(f["id"], 0.0)
            c_data.append({
                "ID": f["id"],
                "Category": domain_map.get(f["domain_id"], "General"),
                "Criterion": f["name"],
                "Optimization": "Benefit (+1)" if f.get("type", 1) == 1 else "Cost (-1)",
                "Global Impact (%)": gw * 100.0
            })
        
        df_flat = pd.DataFrame(c_data).sort_values("Global Impact (%)", ascending=False)
        st.dataframe(
            df_flat, 
            column_config={
                "Global Impact (%)": st.column_config.ProgressColumn("Global Impact (%)", format="%.1f%%", min_value=0, max_value=100)
            },
            hide_index=True, 
            use_container_width=True
        )

    with tab_flat_edit:
        st.header("Direct Criteria Weight Editor")
        st.caption("Rate criteria on a 1–10 scale. The engine will automatically normalize them to sum to 100%[cite: 3].")

        with st.form("flat_weights_form"):
            new_flat_ratings = {}
            domain_map = {d["id"]: d["name"] for d in domains}
            
            for f in factors:
                cur_val = weights["raw_ratings"].get(f["id"], 5.0)
                cat_label = domain_map.get(f["domain_id"], "General")
                new_flat_ratings[f["id"]] = st.slider(
                    f"[{cat_label}] {f['name']} ({f['id']})", 
                    1.0, 10.0, 
                    float(cur_val), 
                    0.5,
                    help=f"Description: {f.get('description', 'No description.')}"
                )
            
            if st.form_submit_button("💾 Save & Normalize Global Weights", type="primary"):
                total_sum = sum(new_flat_ratings.values())
                if total_sum <= 0:
                    st.error("Total sum must be greater than zero.")
                else:
                    norm_weights = {fid: val / total_sum for fid, val in new_flat_ratings.items()}
                    weights["raw_ratings"] = new_flat_ratings
                    weights["category_weights"] = {"General": 1.0}
                    weights["local_weights"] = norm_weights
                    weights["global_weights"] = norm_weights
                    save_weights_state(weights)
                    st.success("Criteria weights updated and normalized successfully!")
                    st.rerun()

# ==============================================================================
# MODE B: DUAL HYBRID (CATEGORIES & CRITERIA AHP)
# ==============================================================================
else:
    SAATY_VALUES = [
        1/9.0, 1/8.0, 1/7.0, 1/6.0, 1/5.0, 1/4.0, 1/3.0, 1/2.0,
        1.0,
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
    ]

    def snap_to_saaty(val: float) -> float:
        clamped = max(1/9.0, min(9.0, val))
        return min(SAATY_VALUES, key=lambda x: abs(x - clamped))

    def format_saaty_label(val: float, dom_a: str, dom_b: str) -> str:
        if abs(val - 1.0) < 1e-3:
            return "1.0 (Equal importance)"
        elif val > 1.0:
            return f"{val:.1f}× ({dom_a} preferred)"
        else:
            recip = 1.0 / val
            return f"1/{recip:.0f} ({dom_b} preferred)"

    def calculate_ahp(matrix: np.ndarray):
        n = matrix.shape[0]
        if n <= 1: 
            return np.array([1.0]), 0.0, None
        if n == 2:
            w = np.array([matrix[0, 1] / (1.0 + matrix[0, 1]), 1.0 / (1.0 + matrix[0, 1])])
            return w, 0.0, None

        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_idx = np.argmax(np.real(eigenvalues))
        w = np.real(eigenvectors[:, max_idx])
        w = np.abs(w) / np.sum(np.abs(w))
        
        RI_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        RI = RI_dict.get(n, 1.49)
        lambda_max = np.real(eigenvalues[max_idx])
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0
        CR = max(0.0, CI / RI) if RI > 0 else 0
        
        max_error = 0
        inconsistent_pair = None
        for i in range(n):
            for j in range(i + 1, n):
                expected = w[i] / w[j] if w[j] > 0 else 1.0
                actual = matrix[i, j]
                error = max(actual / expected, expected / actual) if actual > 0 and expected > 0 else 0
                if error > max_error:
                    max_error = error
                    inconsistent_pair = (i, j)
        return w, float(CR), inconsistent_pair

    def detect_transitivity_cycles(matrix: np.ndarray, domain_names: list) -> list:
        n = matrix.shape[0]
        cycles = []
        for i in range(n):
            for j in range(n):
                if i == j or matrix[i, j] <= 1.25:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if matrix[j, k] > 1.25 and matrix[k, i] > 1.10:
                        cycles.append({
                            "triplet": (domain_names[i], domain_names[j], domain_names[k]),
                            "step1": f"{domain_names[i]} > {domain_names[j]} ({matrix[i, j]:.1f}×)",
                            "step2": f"{domain_names[j]} > {domain_names[k]} ({matrix[j, k]:.1f}×)",
                            "contradiction": f"{domain_names[k]} > {domain_names[i]} ({matrix[k, i]:.1f}×)"
                        })
        return cycles[:4]

    def rank_top_cr_recommendations(matrix: np.ndarray, domain_names: list, current_cr: float, top_n: int = 3) -> list:
        n = matrix.shape[0]
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                orig_val = matrix[i, j]
                best_pair_option = None
                min_pair_cr = current_cr
                for test_val in SAATY_VALUES:
                    if abs(test_val - orig_val) < 1e-3:
                        continue
                    temp_mat = np.copy(matrix)
                    temp_mat[i, j] = test_val
                    temp_mat[j, i] = 1.0 / test_val
                    _, test_cr, _ = calculate_ahp(temp_mat)
                    if test_cr < min_pair_cr:
                        min_pair_cr = test_cr
                        preserves_dir = (test_val >= 1.0 and orig_val >= 1.0) or (test_val <= 1.0 and orig_val <= 1.0)
                        best_pair_option = {
                            "pair": (i, j), "dom_a": domain_names[i], "dom_b": domain_names[j],
                            "current_val": orig_val, "recommended_val": test_val, "new_cr": test_cr,
                            "cr_reduction": current_cr - test_cr, "preserves_direction": preserves_dir
                        }
                if best_pair_option and best_pair_option["cr_reduction"] > 0.005:
                    candidates.append(best_pair_option)
        candidates.sort(key=lambda x: x["new_cr"])
        return candidates[:top_n]

    def auto_tune_ahp_matrix(matrix: np.ndarray, target_cr: float = 0.08, lock_top_n: int = 3) -> tuple[np.ndarray, list[str]]:
        n = matrix.shape[0]
        curr_mat = np.copy(matrix)
        w_init, cr_init, _ = calculate_ahp(curr_mat)
        if cr_init <= target_cr:
            return curr_mat, []

        top_order = np.argsort(-w_init)[:min(lock_top_n, n)]

        def is_valid_ranking(w_test):
            for k in range(len(top_order) - 1):
                if w_test[top_order[k]] < w_test[top_order[k+1]]:
                    return False
            return True

        changes_applied = []
        best_mat = np.copy(curr_mat)
        best_cr = cr_init

        for _ in range(8):
            improved = False
            step_best_mat = None
            step_best_cr = best_cr
            step_change_desc = ""
            for i in range(n):
                for j in range(i + 1, n):
                    orig_val = curr_mat[i, j]
                    for test_val in SAATY_VALUES:
                        if abs(test_val - orig_val) < 1e-3:
                            continue
                        if (orig_val > 1.0 and test_val < 1.0) or (orig_val < 1.0 and test_val > 1.0):
                            continue
                        test_mat = np.copy(curr_mat)
                        test_mat[i, j] = test_val
                        test_mat[j, i] = 1.0 / test_val
                        w_test, cr_test, _ = calculate_ahp(test_mat)
                        if cr_test < step_best_cr:
                            if is_valid_ranking(w_test):
                                step_best_cr = cr_test
                                step_best_mat = test_mat
                                step_change_desc = f"Adjusted Pair ({i+1}, {j+1}) from {orig_val:.2f} to {test_val:.2f}"
                                improved = True

            if improved and step_best_mat is not None:
                curr_mat = step_best_mat
                best_cr = step_best_cr
                best_mat = np.copy(curr_mat)
                changes_applied.append(step_change_desc)
                if best_cr <= target_cr:
                    break
            else:
                break
        return best_mat, changes_applied

    def update_local_and_global_weights():
        if "local_weights" not in weights: weights["local_weights"] = {}
        if "global_weights" not in weights: weights["global_weights"] = {}
        
        for d in domains:
            d_factors = [f["id"] for f in factors if f["domain_id"] == d["id"]]
            domain_sum = sum(weights["raw_ratings"].get(fid, 0) for fid in d_factors)
            for fid in d_factors:
                weights["local_weights"][fid] = weights["raw_ratings"].get(fid, 0) / domain_sum if domain_sum > 0 else 0.0
                
        for f in factors:
            c_w = weights['category_weights'].get(f["domain_id"], 0.0)
            l_w = weights['local_weights'].get(f["id"], 0.0)
            weights['global_weights'][f["id"]] = c_w * l_w

    def save_matrix_with_backup(new_mat: np.ndarray, action_label: str = "Tuning"):
        weights["ahp_matrix_backup"] = {
            "matrix": list(weights.get("ahp_matrix", [])),
            "action_label": action_label
        }
        w_new, _, _ = calculate_ahp(new_mat)
        weights['category_weights'] = {d['id']: float(w_new[k]) for k, d in enumerate(domains)}
        weights['ahp_matrix'] = new_mat.tolist()
        weights["domain_ids"] = [d['id'] for d in domains]
        update_local_and_global_weights()
        save_weights_state(weights)

    saved_matrix = weights.get("ahp_matrix", [])
    saved_domain_ids = weights.get("domain_ids", [])
    if not saved_domain_ids and weights.get("category_weights"):
        saved_domain_ids = list(weights["category_weights"].keys())

    current_domain_ids = [d['id'] for d in domains]
    added_ids = [did for did in current_domain_ids if did not in saved_domain_ids]
    removed_ids = [sid for sid in saved_domain_ids if sid not in current_domain_ids]

    if saved_matrix and removed_ids and not added_ids:
        old_mat = np.array(saved_matrix)
        new_mat = np.ones((num_domains, num_domains))
        for i, id_i in enumerate(current_domain_ids):
            for j, id_j in enumerate(current_domain_ids):
                if id_i in saved_domain_ids and id_j in saved_domain_ids:
                    old_i = saved_domain_ids.index(id_i)
                    old_j = saved_domain_ids.index(id_j)
                    if old_i < old_mat.shape[0] and old_j < old_mat.shape[1]:
                        new_mat[i, j] = old_mat[old_i, old_j]
        saved_matrix = new_mat.tolist()
        weights["ahp_matrix"] = saved_matrix
        weights["domain_ids"] = current_domain_ids
        w_res, _, _ = calculate_ahp(new_mat)
        weights['category_weights'] = {d['id']: float(w_res[idx]) for idx, d in enumerate(domains)}
        update_local_and_global_weights()
        save_weights_state(weights)
        saved_domain_ids = current_domain_ids
        added_ids = []
        removed_ids = []

    added_indices = [idx for idx, d in enumerate(domains) if d['id'] in added_ids]
    missing_pairs = []
    if added_ids:
        for i in range(num_domains):
            for j in range(i + 1, num_domains):
                if i in added_indices or j in added_indices:
                    missing_pairs.append((i, j))
    elif not saved_matrix or len(saved_matrix) != num_domains:
        missing_pairs = list(itertools.combinations(range(num_domains), 2))

    matrix_is_valid = bool(saved_matrix) and len(saved_matrix) == num_domains and not added_ids
    saved_ratings = weights.get("raw_ratings", {})
    missing_factors = [f for f in factors if f["id"] not in saved_ratings]

    try:
        with open(RATING_CONFIG_FILE, 'r', encoding='utf-8') as f:
            global_rating_config = json.load(f)
    except FileNotFoundError:
        global_rating_config = {"weight_init_mode": "🧮 AHP Pairwise Comparisons"}

    st.title("⚖️ Weights Engine — Dual Hybrid Architecture")
    st.caption("Active Mode: **Dual Hybrid**. Calibrate category priorities via AHP and local criteria via ratings[cite: 3].")

    if not matrix_is_valid or missing_pairs:
        st.header("Step 1: Category Weights Initialization")
        st.warning("🚨 AHP Matrix needs initialization or update.")
        
        default_global_mode = global_rating_config.get("weight_init_mode", "🧮 AHP Pairwise Comparisons")
        init_options = ["🧮 AHP Pairwise Comparisons", "🎛️ Direct Weight Sliders"]
        default_index = init_options.index(default_global_mode) if default_global_mode in init_options else 0
        
        init_style = st.radio(
            "Choose Initialization Method:", 
            init_options, 
            index=default_index,
            horizontal=True,
            key="init_style_radio"
        )
        
        pairs_to_ask = missing_pairs if missing_pairs else list(itertools.combinations(range(num_domains), 2))
        
        if not pairs_to_ask and num_domains == 1:
            weights['category_weights'] = {domains[0]['id']: 1.0}
            weights['ahp_matrix'] = [[1.0]]
            weights['domain_ids'] = current_domain_ids
            update_local_and_global_weights()
            save_weights_state(weights)
            st.rerun()
            
        if init_style == "🧮 AHP Pairwise Comparisons":
            with st.form("ahp_init_form"):
                ahp_answers = {}
                matrix_ref = np.array(weights.get("ahp_matrix")) if weights.get("ahp_matrix") else None
                
                for i, j in pairs_to_ask:
                    dom_A, dom_B = domains[i]['name'], domains[j]['name']
                    scale_opts = {
                        f"9: {dom_A} is extremely more important": 9.0, 
                        f"7: {dom_A} is very strongly preferred": 7.0,
                        f"5: {dom_A} is strongly preferred": 5.0, 
                        f"3: {dom_A} is weakly preferred": 3.0,
                        f"1: {dom_A} and {dom_B} are equally important": 1.0, 
                        f"1/3: {dom_B} is weakly preferred": 1/3.0,
                        f"1/5: {dom_B} is strongly preferred": 1/5.0, 
                        f"1/7: {dom_B} is very strongly preferred": 1/7.0,
                        f"1/9: {dom_B} is extremely more important": 1/9.0
                    }
                    
                    current_val = 1.0
                    if matrix_ref is not None and matrix_ref.shape[0] > i and matrix_ref.shape[1] > j:
                        current_val = float(matrix_ref[i, j])
                    
                    closest_key = min(scale_opts.keys(), key=lambda k: abs(scale_opts[k] - current_val))
                    default_idx = list(scale_opts.keys()).index(closest_key)
                    
                    st.markdown(f"**Compare: {dom_A} vs {dom_B}**")
                    selection = st.selectbox("Select relative importance:", options=list(scale_opts.keys()), index=default_idx, key=f"ahp_{i}_{j}")
                    ahp_answers[(i, j)] = scale_opts[selection]
                    st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
                    
                if st.form_submit_button("🧮 Build Matrix & Calculate Weights", type="primary"):
                    matrix = np.ones((num_domains, num_domains))
                    if saved_matrix:
                        old_mat = np.array(saved_matrix)
                        old_n = old_mat.shape[0]
                        overlap = min(old_n, num_domains)
                        matrix[:overlap, :overlap] = old_mat[:overlap, :overlap]
                        
                    for (i, j), val in ahp_answers.items():
                        matrix[i, j] = val
                        matrix[j, i] = 1.0 / val
                        
                    w, _, _ = calculate_ahp(matrix)
                    weights['category_weights'] = {d['id']: float(w[idx]) for idx, d in enumerate(domains)}
                    weights['ahp_matrix'] = matrix.tolist()
                    weights['domain_ids'] = current_domain_ids
                    weights['ahp_matrix_backup'] = None
                    update_local_and_global_weights()
                    save_weights_state(weights)
                    st.success("AHP matrix initialized successfully!")
                    st.rerun()
        else:
            with st.form("slider_init_form"):
                slider_vals = {}
                existing_cat_w = weights.get("category_weights", {})
                for d in domains:
                    default_pct = existing_cat_w.get(d['id'], 0.0) * 100.0
                    slider_vals[d['id']] = st.slider(f"{d['name']} (%)", 0.0, 100.0, float(default_pct), 0.01)
                
                if st.form_submit_button("🎛️ Save & Generate Matrix", type="primary"):
                    total_val = sum(slider_vals.values())
                    if total_val <= 0:
                        st.error("Total weight must be greater than 0.")
                    else:
                        norm_weights = np.array([slider_vals[d['id']] / total_val for d in domains])
                        matrix = np.ones((num_domains, num_domains))
                        for i in range(num_domains):
                            for j in range(num_domains):
                                if norm_weights[j] > 0:
                                    matrix[i, j] = norm_weights[i] / norm_weights[j]
                                else:
                                    matrix[i, j] = 1.0
                                    
                        weights['category_weights'] = {domains[idx]['id']: float(norm_weights[idx]) for idx in range(num_domains)}
                        weights['ahp_matrix'] = matrix.tolist()
                        weights['domain_ids'] = current_domain_ids
                        weights['ahp_matrix_backup'] = None
                        update_local_and_global_weights()
                        save_weights_state(weights)
                        st.success("Weights generated successfully!")
                        st.rerun()

    elif missing_factors:
        st.header("Step 2: Rate Criteria by Category (1–10)")
        with st.form("missing_crit_form"):
            cat_ratings = {}
            for f in missing_factors:
                cat_ratings[f["id"]] = st.slider(f"[{f['id']}] {f['name']}", 1.0, 10.0, 5.0, 0.5)
            if st.form_submit_button("💾 Save Criteria Ratings", type="primary"):
                for fid, val in cat_ratings.items():
                    weights["raw_ratings"][fid] = val
                update_local_and_global_weights()
                save_weights_state(weights)
                st.success("Criteria ratings saved!")
                st.rerun()
    else:
        tab_dash, tab_ahp, tab_local = st.tabs(["📊 View Dashboard", "⚖️ AHP Diagnostics & Matrix", "🎯 Edit Local Criteria"])

        with tab_dash:
            st.header("1. Category Weights (High-Level Priority)")
            cat_df = pd.DataFrame([{"Category": d['name'], "Percentage": weights['category_weights'].get(d['id'], 0.0) * 100} for d in domains]).sort_values("Percentage", ascending=False)
            st.dataframe(cat_df, column_config={"Percentage": st.column_config.ProgressColumn("Impact (%)", format="%.1f%%", min_value=0, max_value=100)}, hide_index=True, use_container_width=True)

            st.markdown("---")
            st.header("2. Criteria Weights (Global Impact)")
            c_data = []
            domain_map = {d["id"]: d["name"] for d in domains}
            for f in factors:
                c_data.append({
                    "ID": f["id"], "Category": domain_map.get(f["domain_id"], "Unknown"),
                    "Criterion": f["name"], "Type": "Benefit" if f.get("type", 1) == 1 else "Cost",
                    "Local %": weights['local_weights'].get(f['id'], 0.0) * 100,
                    "Global %": weights['global_weights'].get(f['id'], 0.0) * 100
                })
            c_df = pd.DataFrame(c_data).sort_values("Global %", ascending=False)
            st.dataframe(c_df, column_config={
                "Local %": st.column_config.NumberColumn("Local %", format="%.1f%%"),
                "Global %": st.column_config.ProgressColumn("Global Impact (%)", format="%.1f%%", min_value=0, max_value=100),
            }, hide_index=True, use_container_width=True, height=700)

        with tab_ahp:
            st.header("AHP Matrix Diagnostics & Consistency Optimizer")
            
            ahp_edit_mode = st.radio(
                "Select Editing Mode:", 
                ["📊 Pairwise Grid & Diagnostics", "🎛️ Direct Weight Slider Tuner (Inverse AHP)"], 
                horizontal=True,
                key="ahp_sub_mode"
            )
            
            domain_names = [d["name"] for d in domains]
            mat_np = np.array(saved_matrix)
            w_curr, cr, inc_pair = calculate_ahp(mat_np)

            if ahp_edit_mode == "🎛️ Direct Weight Slider Tuner (Inverse AHP)":
                st.subheader("🎛️ Direct Weight Slider Tuner")
                st.markdown("Fine-tune category weights directly with **0.01% resolution**. The underlying decision matrix will instantly recalculate to match your target distribution.")
                
                current_weights_list = [weights['category_weights'].get(d['id'], 1.0 / num_domains) * 100 for d in domains]
                
                with st.form("direct_slider_tuner_form"):
                    new_slider_weights = {}
                    for idx, d in enumerate(domains):
                        new_slider_weights[d['id']] = st.slider(
                            f"{d['name']} (%)", 
                            0.0, 100.0, 
                            float(current_weights_list[idx]), 
                            0.01, 
                            key=f"tune_slider_{d['id']}"
                        )
                    
                    optimization_mode = st.radio(
                        "Matrix Scaling Preference:",
                        [
                            "📏 Standard AHP Scale (1–9 Limit)",
                            "📐 Precise Mathematical Ratios (Unbounded)"
                        ],
                        key="tuner_optimization_mode",
                        horizontal=True,
                        help="Standard AHP clamps and rounds values to fit Saaty's 1–9 scale. Precise mode applies raw mathematical ratios which can exceed 9."
                    )
                    
                    if st.form_submit_button("💾 Save Slider Weights & Update Matrix", type="primary"):
                        total_w = sum(new_slider_weights.values())
                        if total_w <= 0:
                            st.error("Total weight must be greater than 0.")
                        else:
                            norm_w = np.array([new_slider_weights[d['id']] / total_w for d in domains])
                            new_matrix = np.ones((num_domains, num_domains))
                            
                            for i in range(num_domains):
                                for j in range(num_domains):
                                    if norm_w[j] > 0:
                                        raw_ratio = norm_w[i] / norm_w[j]
                                        if "Standard AHP Scale" in optimization_mode:
                                            new_matrix[i, j] = snap_to_saaty(raw_ratio)
                                        else:
                                            new_matrix[i, j] = raw_ratio
                                    else:
                                        new_matrix[i, j] = 1.0
                                        
                            save_matrix_with_backup(new_matrix, action_label="Direct Slider Tuner Update")
                            st.success("Weights and decision matrix successfully updated via slider tuner!")
                            st.rerun()
            else:
                backup = weights.get("ahp_matrix_backup")
                if backup and backup.get("matrix"):
                    b_mat = np.array(backup["matrix"])
                    if b_mat.shape == mat_np.shape:
                        _, prev_cr, _ = calculate_ahp(b_mat)
                        u_col1, u_col2 = st.columns([3.5, 1.5])
                        with u_col1:
                            st.info(f"↩️ **Previous State Available:** Saved before `{backup.get('action_label', 'last change')}` (Previous CR was **{prev_cr:.3f}**).")
                        with u_col2:
                            st.write("")
                            if st.button("↩️ Revert / Undo Change", type="secondary", use_container_width=True):
                                weights['ahp_matrix'] = backup["matrix"]
                                weights['ahp_matrix_backup'] = None
                                w_rev, _, _ = calculate_ahp(np.array(backup["matrix"]))
                                weights['category_weights'] = {d['id']: float(w_rev[k]) for k, d in enumerate(domains)}
                                update_local_and_global_weights()
                                save_weights_state(weights)
                                st.success("Reverted to previous matrix state!")
                                st.rerun()

                c1, c2, c3 = st.columns([1, 1.2, 1.8])
                with c1:
                    st.caption("Consistency Ratio (CR)")
                    cr_color = "#22c55e" if cr <= 0.10 else ("#eab308" if cr <= 0.20 else "#ef4444")
                    st.markdown(f"<h2 style='color: {cr_color}; margin-top: -10px;'>{cr:.3f}</h2>", unsafe_allow_html=True)
                    status_text = "✅ Highly Consistent" if cr <= 0.10 else ("⚠️ Borderline" if cr <= 0.20 else "🔴 Highly Inconsistent")
                    st.caption(f"Status: **{status_text}** (Target: < 0.10)")
                
                with c2:
                    st.caption("Dominant Contradiction")
                    if inc_pair and cr > 0.01:
                        inc_text = f"<span style='color:#38bdf8; font-weight:bold;'>{domain_names[inc_pair[0]]}</span> vs <span style='color:#f472b6; font-weight:bold;'>{domain_names[inc_pair[1]]}</span>"
                    else:
                        inc_text = "<span style='color:#22c55e; font-weight:bold;'>None (Optimal)</span>"
                    st.markdown(f"<div style='font-size: 1.05rem; line-height: 1.4; margin-top: 0px;'>{inc_text}</div>", unsafe_allow_html=True)

                with c3:
                    st.caption("Matrix Dimension & Pair Count")
                    st.markdown(f"<div style='font-size: 1.05rem; line-height: 1.4; margin-top: 0px;'><b>{num_domains} Categories</b> ({int(num_domains*(num_domains-1)/2)} Pairwise Comparisons)</div>", unsafe_allow_html=True)

                st.markdown("---")

                if cr > 0.10:
                    st.subheader("💡 Intelligent Inconsistency Optimizer")
                    st.caption("Fixing inconsistencies at scale requires targeting highest-leverage comparisons or breaking circular loops.")

                    cycles = detect_transitivity_cycles(mat_np, domain_names)
                    if cycles:
                        with st.expander(f"⚠️ Circular Judgment Contradictions Detected ({len(cycles)} Loops Found)", expanded=True):
                            for cyc in cycles:
                                st.markdown(f"""
                                * **Circular Loop:** {cyc['step1']} ➔ {cyc['step2']} ➔ **Contradiction:** {cyc['contradiction']}
                                """)
                            st.info("💡 **How to resolve:** Adjust the contradictory comparison so the third item does not loop back over the first.")

                    recs = rank_top_cr_recommendations(mat_np, domain_names, cr, top_n=3)
                    if recs:
                        st.markdown("#### 🎯 Top 3 Targeted Adjustments to Resolve CR")
                        for idx, r in enumerate(recs):
                            col_r1, col_r2, col_r3, col_r4 = st.columns([2.5, 2.5, 2, 1.5])
                            with col_r1:
                                st.markdown(f"**#{idx+1}: {r['dom_a']} vs {r['dom_b']}**")
                                st.caption(f"Current: `{format_saaty_label(r['current_val'], r['dom_a'], r['dom_b'])}`")
                            with col_r2:
                                st.markdown(f"Change to: **`{format_saaty_label(r['recommended_val'], r['dom_a'], r['dom_b'])}`**")
                                dir_badge = "✅ Preserves Direction" if r['preserves_direction'] else "⚠️ Flips Direction"
                                st.caption(dir_badge)
                            with col_r3:
                                st.markdown(f"Projected CR: **{r['new_cr']:.3f}**")
                                st.caption(f"Reduction: **-{r['cr_reduction']:.3f}**")
                            with col_r4:
                                st.write("")
                                if st.button(f"⚡ Apply Fix", key=f"apply_rec_{idx}", type="primary"):
                                    i, j = r["pair"]
                                    new_v = r["recommended_val"]
                                    new_mat = np.copy(mat_np)
                                    new_mat[i, j] = new_v
                                    new_mat[j, i] = 1.0 / new_v
                                    save_matrix_with_backup(new_mat, action_label=f"Applied Fix on {r['dom_a']} vs {r['dom_b']}")
                                    st.success(f"Applied fix for {r['dom_a']} vs {r['dom_b']}!")
                                    st.rerun()
                            st.markdown("<hr style='margin: 6px 0;'>", unsafe_allow_html=True)

                    top_ranked_names = [domain_names[idx] for idx in np.argsort(-w_curr)[:min(3, num_domains)]]
                    rank_str = " > ".join(top_ranked_names)

                    with st.expander("🪄 Automated Minimum-Revision Tuning (One-Click Auto-Harmonizer)"):
                        st.write(f"Automatically resolves pairwise inconsistencies to reach **CR ≤ 0.08** while strictly locking your top priority hierarchy: **{rank_str}**.")
                        if st.button("🪄 Auto-Tune Matrix for Consistency (CR ≤ 0.08)", type="secondary"):
                            tuned_mat, changes = auto_tune_ahp_matrix(mat_np, target_cr=0.08, lock_top_n=3)
                            _, cr_tuned, _ = calculate_ahp(tuned_mat)
                            if np.array_equal(tuned_mat, mat_np):
                                st.warning("Matrix is already at the lowest achievable CR without violating your top 3 locked category rankings.")
                            else:
                                save_matrix_with_backup(tuned_mat, action_label=f"Auto-Tune (CR {cr:.3f} ➔ {cr_tuned:.3f})")
                                st.success(f"Matrix auto-tuned! CR improved from {cr:.3f} ➔ {cr_tuned:.3f} while preserving '{rank_str}'.")
                                st.rerun()

                    st.markdown("---")

                st.subheader("Current Decision Matrix")
                MAX_LEN = 16
                full_domain_names = [d["name"] for d in domains]
                short_domain_names = []
                for name in full_domain_names:
                    if len(name) > MAX_LEN:
                        short_domain_names.append(name[:MAX_LEN - 3] + "...")
                    else:
                        diff = MAX_LEN - len(name)
                        short_domain_names.append(name + ("\u00a0" * diff))
                
                df_matrix = pd.DataFrame(saved_matrix, columns=short_domain_names, index=short_domain_names)
                
                def color_heatmap(val):
                    opacity = min(0.8, max(0.05, val / 9.0)) if val >= 1.0 else min(0.8, max(0.05, (1.0 / val) / 9.0))
                    bg_color = "56, 189, 248" if val >= 1.0 else "244, 114, 182"
                    return f'background-color: rgba({bg_color}, {opacity}); color: white;'

                styled_df = df_matrix.style.map(color_heatmap).format("{:.2f}")
                box_height = min(450, max(220, (len(domain_names) + 1) * 45))
                st.dataframe(styled_df, use_container_width=True, height=box_height)
                
                with st.expander("🔍 View Full Category Name Mapping"):
                    for full, short in zip(full_domain_names, short_domain_names):
                        cleaned_short = short.replace("\u00a0", " ")
                        if full != cleaned_short:
                            st.markdown(f"* **`{cleaned_short}`** ➔ {full}")

                st.markdown("---")
                st.subheader("✏️ Manual Pairwise Editor")
                pair_opts = {}
                for i in range(num_domains):
                    for j in range(i + 1, num_domains):
                        pair_opts[f"{domain_names[i]} vs {domain_names[j]}"] = (i, j)

                if pair_opts:
                    col_sel, col_upd = st.columns([1, 1])
                    with col_sel:
                        sel_pair_label = st.selectbox("Select pair to tweak:", list(pair_opts.keys()))
                        i, j = pair_opts[sel_pair_label]
                        dom_A, dom_B = domain_names[i], domain_names[j]
                        current_val = saved_matrix[i][j]

                        scale_opts = {
                            f"9: {dom_A} is extremely more important": 9.0, 
                            f"7: {dom_A} is very strongly preferred": 7.0,
                            f"5: {dom_A} is strongly preferred": 5.0, 
                            f"3: {dom_A} is weakly preferred": 3.0,
                            f"1: {dom_A} and {dom_B} are equally important": 1.0, 
                            f"1/3: {dom_B} is weakly preferred": 1/3.0,
                            f"1/5: {dom_B} is strongly preferred": 1/5.0, 
                            f"1/7: {dom_B} is very strongly preferred": 1/7.0,
                            f"1/9: {dom_B} is extremely more important": 1/9.0
                        }
                        closest_key = min(scale_opts.keys(), key=lambda k: abs(scale_opts[k] - current_val))
                        default_idx = list(scale_opts.keys()).index(closest_key)

                    with col_upd:
                        with st.form("edit_pair_form"):
                            new_selection = st.selectbox("Update importance:", options=list(scale_opts.keys()), index=default_idx)
                            if st.form_submit_button("💾 Update Pair & Recalculate", type="primary"):
                                new_val = scale_opts[new_selection]
                                new_mat = np.copy(saved_matrix)
                                new_mat[i, j] = new_val
                                new_mat[j, i] = 1.0 / new_val
                                save_matrix_with_backup(new_mat, action_label=f"Manual Edit on {dom_A} vs {dom_B}")
                                st.success("Matrix updated successfully!")
                                st.rerun()

                st.markdown("---")
                with st.expander("⚠️ Danger Zone (Wipe AHP)"):
                    st.error("This will permanently delete your entire AHP matrix and prompt you to re-evaluate it from scratch.")
                    if st.button("🗑️ Wipe & Re-evaluate Entire Matrix", type="secondary"):
                        weights["ahp_matrix"] = []
                        weights["ahp_matrix_backup"] = None
                        weights["domain_ids"] = []
                        save_weights_state(weights)
                        st.rerun()

        with tab_local:
            st.header("Targeted Criteria Editing")
            dom_map_opts = {d["name"]: d["id"] for d in domains}
            if dom_map_opts:
                sel_dom_name = st.selectbox("Select Category to edit:", list(dom_map_opts.keys()))
                with st.form("edit_local_form"):
                    new_ratings = {}
                    for f in [f for f in factors if f["domain_id"] == dom_map_opts[sel_dom_name]]:
                        new_ratings[f["id"]] = st.slider(f"[{f['id']}] {f['name']}", 1.0, 10.0, float(weights["raw_ratings"].get(f["id"], 5.0)), 0.5)
                        if f.get('description'):
                            st.caption(f"_{f['description']}_")
                        st.markdown("---")
                        
                    if st.form_submit_button("💾 Save Local Ratings", type="primary"):
                        for fid, val in new_ratings.items(): 
                            weights["raw_ratings"][fid] = val
                        update_local_and_global_weights()
                        save_weights_state(weights)
                        st.success("Local weights updated successfully!")
                        st.rerun()