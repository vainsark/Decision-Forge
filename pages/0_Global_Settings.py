"""
Decision Support System - Global Settings Page
Manages fuzzy logic coefficients, defuzzification weights, PROMETHEE thresholds, 
WASPAS Lambda parameter, weighting architecture (Dual Hybrid vs Single Flat) with strict switch validation,
weight initialization preferences, and active decision alternatives.
"""

import streamlit as st
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluations import calculate_trapezoid
from src.factors_manager import load_factors_config, ensure_ghost_category, remove_ghost_category
try:
    from src.mcdm_engine import load_engine_config, save_engine_config
except ImportError:
    load_engine_config = None
    save_engine_config = None

RATING_CONFIG_FILE = os.path.join(BASE_DIR, 'data', 'rating_config.json')
EVALUATIONS_FILE = os.path.join(BASE_DIR, 'data', 'evaluations.json')

st.set_page_config(page_title="Global Settings", page_icon="⚙️", layout="wide")
st.title("⚙️ Global System Settings")
st.caption("Configure global fuzzy parameters, weighting architecture, and active decision alternatives.")

# Load config with backward compatibility for legacy keys
try:
    with open(RATING_CONFIG_FILE, 'r', encoding='utf-8') as f:
        rating_config = json.load(f)
        if "alternatives" not in rating_config and "countries" in rating_config:
            rating_config["alternatives"] = rating_config.pop("countries")
except FileNotFoundError:
    rating_config = {
        "alternatives": [],
        "coefficients": {"Kv": 0.5, "Ke": 0.5, "Kb": 1.0},
        "defuzz_weights": [0.1667, 0.3333, 0.3333, 0.1667],
        "promethee_q": 0.5,
        "promethee_p": 3.5,
        "waspas_lambda": 0.5,
        "weight_system_mode": "Dual Hybrid (Categories & Criteria)",
        "weight_init_mode": "🧮 AHP Pairwise Comparisons"
    }

def save_rating_config(config):
    with open(RATING_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def recalculate_all_evaluations(coeffs):
    if not os.path.exists(EVALUATIONS_FILE): 
        return
    with open(EVALUATIONS_FILE, 'r', encoding='utf-8') as f:
        evals = json.load(f)
        
    for ev in evals:
        trap = calculate_trapezoid(ev['rating'], ev['volatility'], ev['uncertainty'], ev['bias'], coeffs)
        ev['trapezoid'] = trap
        ev['coefficients'] = coeffs
        
    with open(EVALUATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(evals, f, indent=4)

# UI TABS
tab_math, tab_weights, tab_alternatives = st.tabs([
    "🧮 Mathematical & Fuzzy Parameters", 
    "⚖️ Weight System & Initialization", 
    "🎯 Manage Alternatives"
])

with tab_math:
    # ----------------------------------------------------
    # SECTION 1: PROMETHEE PARAMETERS
    # ----------------------------------------------------
    with st.form("promethee_settings_form"):
        st.subheader("1. Fuzzy PROMETHEE Parameters")
        st.caption("Set the Type V preference function thresholds on the 0–10 rating difference scale.")
        
        pq1, pq2 = st.columns(2)
        with pq1:
            new_q = st.number_input(
                "Indifference Threshold (q)", 
                value=float(rating_config.get("promethee_q", 0.5)), 
                min_value=0.0, 
                max_value=5.0, 
                step=0.1,
                help="Score differences at or below this value yield zero preference (noise filter)."
            )
        with pq2:
            new_p = st.number_input(
                "Strict Preference Threshold (p)", 
                value=float(rating_config.get("promethee_p", 3.5)), 
                min_value=0.1, 
                max_value=10.0, 
                step=0.1,
                help="Score differences at or above this value yield 100% preference (1.0)."
            )
            
        if st.form_submit_button("💾 Save PROMETHEE Parameters", type="primary"):
            if new_p <= new_q:
                st.error("⚠️ Preference threshold (p) must be strictly greater than indifference threshold (q).")
            else:
                rating_config["promethee_q"] = new_q
                rating_config["promethee_p"] = new_p
                save_rating_config(rating_config)
                st.success(f"PROMETHEE parameters saved: q = {new_q}, p = {new_p}")

    st.markdown("---")

    # ----------------------------------------------------
    # SECTION 2: DEFUZZIFICATION WEIGHTS
    # ----------------------------------------------------
    with st.form("defuzz_weights_form"):
        st.subheader("2. Defuzzification Weights (Centroid / GMIR)")
        st.caption("Set the voting power for each point of the trapezoid (a, b, c, d). Standard GMIR is [1/6, 2/6, 2/6, 1/6].")
        
        d_weights = rating_config.get("defuzz_weights", [0.1667, 0.3333, 0.3333, 0.1667])
        if len(d_weights) != 4: 
            d_weights = [0.1667, 0.3333, 0.3333, 0.1667]
        
        dw1, dw2, dw3, dw4 = st.columns(4)
        with dw1:
            w_a = st.number_input("Weight 'a' (Worst case)", value=float(d_weights[0]), step=0.05, format="%.4f")
        with dw2:
            w_b = st.number_input("Weight 'b' (Lower core)", value=float(d_weights[1]), step=0.05, format="%.4f")
        with dw3:
            w_c = st.number_input("Weight 'c' (Upper core)", value=float(d_weights[2]), step=0.05, format="%.4f")
        with dw4:
            w_d = st.number_input("Weight 'd' (Best case)", value=float(d_weights[3]), step=0.05, format="%.4f")

        col_save_dw, col_reset_dw = st.columns([1, 1])
        with col_save_dw:
            save_dw = st.form_submit_button("💾 Save & Normalize Weights")
        
        if save_dw:
            total_w = w_a + w_b + w_c + w_d
            if total_w == 0: 
                total_w = 1.0
            normalized_dw = [round(w_a / total_w, 4), round(w_b / total_w, 4), round(w_c / total_w, 4), round(w_d / total_w, 4)]
            rating_config["defuzz_weights"] = normalized_dw
            save_rating_config(rating_config)
            st.success(f"Defuzzification weights normalized to: {normalized_dw}")

    st.markdown("---")

    # ----------------------------------------------------
    # SECTION 3: TRAPEZOID COEFFICIENTS (RECALCULATES DATA)
    # ----------------------------------------------------
    with st.form("coeffs_form"):
        st.subheader("3. Fuzzy Trapezoid Multipliers")
        st.caption("Tune how volatility, uncertainty, and bias physically shape bounds. **Saves & recalculates all stored evaluations.**")
        
        coeffs = rating_config.get("coefficients", {})
        c1, c2, c3 = st.columns(3)
        with c1:
            new_kv = st.number_input("Volatility (Kv)", value=float(coeffs.get("Kv", 0.5)), step=0.1)
        with c2:
            new_ke = st.number_input("Uncertainty (Ke)", value=float(coeffs.get("Ke", 0.5)), step=0.1)
        with c3:
            new_kb = st.number_input("Bias Shift (Kb)", value=float(coeffs.get("Kb", 1.0)), step=0.1)

        if st.form_submit_button("💾 Save Multipliers & Recalculate Evaluations", type="primary"):
            new_coeffs = {"Kv": new_kv, "Ke": new_ke, "Kb": new_kb}
            rating_config["coefficients"] = new_coeffs
            save_rating_config(rating_config)
            recalculate_all_evaluations(new_coeffs)
            st.success("Coefficients saved and all stored evaluations updated!")

    st.markdown("---")

    # ----------------------------------------------------
    # SECTION 4: WASPAS ENGINE PARAMETER (LAMBDA)
    # ----------------------------------------------------
    with st.form("waspas_settings_form"):
        st.subheader("4. WASPAS Engine Parameter (λ)")
        st.caption("Control the balance between Additive (WSM) and Multiplicative (WPM) scoring models in WASPAS.")
        
        default_lambda = float(rating_config.get("waspas_lambda", 0.5))
        if load_engine_config:
            try:
                eng_cfg = load_engine_config()
                default_lambda = float(eng_cfg.get("parameters", {}).get("WASPAS_lambda", default_lambda))
            except Exception:
                pass

        new_lambda = st.slider(
            "WASPAS Lambda (λ)", 
            min_value=0.0, 
            max_value=1.0, 
            value=default_lambda, 
            step=0.05,
            help="λ = 1.0 is pure WSM (additive utility), λ = 0.0 is pure WPM (multiplicative utility), λ = 0.5 is standard equal balance."
        )
        
        if st.form_submit_button("💾 Save WASPAS Parameter", type="primary"):
            rating_config["waspas_lambda"] = new_lambda
            save_rating_config(rating_config)
            if save_engine_config and load_engine_config:
                try:
                    eng_cfg = load_engine_config()
                    eng_cfg.setdefault("parameters", {})["WASPAS_lambda"] = new_lambda
                    save_engine_config(eng_cfg)
                except Exception:
                    pass
            st.success(f"WASPAS Lambda successfully saved: λ = {new_lambda}")


with tab_weights:
    st.subheader("Weighting Architecture & Default Initialization")
    st.caption("Configure how criteria weights are structured and initialized across the system.")
    
    with st.form("weight_architecture_form"):
        st.markdown("### 1. Weight System Architecture")
        st.caption("Choose between a hierarchical Dual Hybrid system (Categories → Criteria) or a simplified Single Flat Weighting system.")
        
        current_sys_mode = rating_config.get("weight_system_mode", "Dual Hybrid (Categories & Criteria)")
        sys_options = ["Dual Hybrid (Categories & Criteria)", "Single Flat Weighting (Direct Criteria Pool)"]
        default_sys_idx = sys_options.index(current_sys_mode) if current_sys_mode in sys_options else 0
        
        selected_sys_mode = st.radio(
            "Select Weighting System",
            options=sys_options,
            index=default_sys_idx,
            help="Dual Hybrid uses high-level categories with AHP pairwise matrices. Single Flat treats all criteria in a unified flat pool."
        )
        
        st.markdown("---")
        st.markdown("### 2. Default Category Weight Initialization")
        st.caption("Choose whether new or reset weight structures default to structured AHP Pairwise Comparisons or Direct Sliders (applicable in Dual Hybrid mode).")
        
        current_init_mode = rating_config.get("weight_init_mode", "🧮 AHP Pairwise Comparisons")
        init_options = ["🧮 AHP Pairwise Comparisons", "🎛️ Direct Weight Sliders"]
        default_init_idx = init_options.index(current_init_mode) if current_init_mode in init_options else 0
        
        selected_init_mode = st.selectbox(
            "Default Weight Method", 
            options=init_options, 
            index=default_init_idx,
            help="Determines how category priorities are initially constructed in the Weights Engine."
        )
        
        if st.form_submit_button("💾 Save Weight System Preferences", type="primary"):
            factors_cfg = load_factors_config()
            existing_factors = factors_cfg.get("factors", [])
            
            # Check if user is trying to switch FROM Single TO Dual mode
            is_switching_to_dual = (
                current_sys_mode != selected_sys_mode and 
                selected_sys_mode == "Dual Hybrid (Categories & Criteria)"
            )
            
            if is_switching_to_dual and len(existing_factors) > 0:
                st.error("⚠️ Cannot switch to Dual Hybrid mode because criteria already exist. Please go to 'Criteria Overview' and delete all criteria first before switching to Dual mode.")
            else:
                if selected_sys_mode == "Single Flat Weighting (Direct Criteria Pool)":
                    ensure_ghost_category()
                    st.success("Switched to Single Flat Weighting mode and initialized shadow category (d01)!")
                else:
                    remove_ghost_category()
                    st.success("Switched to Dual Hybrid mode successfully!")
                    
                rating_config["weight_system_mode"] = selected_sys_mode
                rating_config["weight_init_mode"] = selected_init_mode
                save_rating_config(rating_config)
                st.rerun()


with tab_alternatives:
    st.header("Active Decision Alternatives")
    alternatives = rating_config.get("alternatives", [])
    st.write("Current alternatives in the evaluation pool:")
    st.write(", ".join([f"**{alt}**" for alt in alternatives]) if alternatives else "No alternatives added yet.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.form("add_alternative_form", clear_on_submit=True):
            st.subheader("Add Alternative")
            new_alt = st.text_input("Alternative Name")
            if st.form_submit_button("➕ Add"):
                clean_alt = (new_alt or "").strip()
                if clean_alt and clean_alt not in alternatives:
                    rating_config.setdefault("alternatives", []).append(clean_alt)
                    save_rating_config(rating_config)
                    st.success(f"'{clean_alt}' added successfully!")
                    st.rerun()
                    
    with col2:
        with st.form("remove_alternative_form"):
            st.subheader("Remove Alternative")
            rem_alt = st.selectbox("Select Alternative", [""] + alternatives)
            if st.form_submit_button("🗑️ Remove"):
                if rem_alt:
                    rating_config["alternatives"].remove(rem_alt)
                    save_rating_config(rating_config)
                    st.success(f"'{rem_alt}' removed successfully!")
                    st.rerun()