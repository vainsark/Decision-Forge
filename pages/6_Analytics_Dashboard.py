"""
Decision Support System (DSS) - Analytics Dashboard
Visualizes multi-model MCDM results, dynamically derived consensus metrics,
model-selectable domain/criteria score decompositions, and raw snapshot matrices using friendly run names.
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import sys
from typing import Optional, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.project_manager import get_active_project_dir
from src.project_manager import get_active_project_id


# Dynamic path resolution functions for active project workspace isolation
def _get_project_data_dir() -> str:
    """Returns the active project directory, asserting it is not None."""
    proj_dir = get_active_project_dir()
    assert proj_dir is not None, "Active project directory is required."
    return proj_dir

def get_runs_dir() -> str:
    d = os.path.join(_get_project_data_dir(), "runs")
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    return d

def get_rating_config_filepath() -> str:
    return os.path.join(_get_project_data_dir(), "rating_config.json")

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

# ==========================================
# ACTIVE PROJECT GUARD
# ==========================================
active_proj_id = get_active_project_id()
if not active_proj_id:
    st.warning("⚠️ **No Active Project Selected.** Please go to the **Decision Hub** to create or open a project workspace.")
    if st.button("🗂️ Go to Decision Hub", type="primary"):
        st.switch_page("pages/0_Decision_Hub.py")
    st.stop()

RUNS_DIR = get_runs_dir()
RATING_CONFIG_FILE = get_rating_config_filepath()



run_files = sorted([f for f in os.listdir(RUNS_DIR) if f.endswith(".json")], reverse=True) if os.path.exists(RUNS_DIR) else []

# ==========================================
# GUARD: NO RUNS FOUND
# ==========================================
if not run_files:
    st.title("📊 Analytics Dashboard")
    st.warning("⚠️ No baseline MCDM runs found in this project's runs directory.")
    st.info("Please navigate to **MCDM Engine**, run a calculation, and save a baseline run first.")
    st.page_link("pages/5_MCDM_Engine.py", label="➔ Go to MCDM Engine", use_container_width=False)
    st.stop()

# ==========================================
# PRELOAD RUN METADATA (FOR FRIENDLY NAMES)
# ==========================================
run_display_map = {}
for fname in run_files:
    fpath = os.path.join(RUNS_DIR, fname)
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            run_name = data.get("name") or data.get("run_id") or fname
            timestamp_short = data.get("timestamp", "").split("T")[0]
            if timestamp_short:
                display_label = f"📁 {run_name} ({timestamp_short})"
            else:
                display_label = f"📁 {run_name}"
            run_display_map[fname] = display_label
    except Exception:
        run_display_map[fname] = f"📁 {fname}"

# ==========================================
# RUN SELECTOR & DATA LOADER
# ==========================================
st.sidebar.header("Run Selection")
selected_run_file = st.sidebar.selectbox(
    "Choose Saved Baseline Run:",
    run_files,
    format_func=lambda fname: run_display_map.get(fname, fname),
    index=0
)

run_path = os.path.join(RUNS_DIR, selected_run_file)
with open(run_path, "r", encoding="utf-8") as f:
    run_data = json.load(f)

run_name = run_data.get("name") or run_data.get("run_id") or selected_run_file
run_id = run_data.get("run_id", selected_run_file)
timestamp = run_data.get("timestamp", "Unknown")
alternatives = run_data.get("alternatives", run_data.get("countries", []))
results = run_data.get("results", {})
snapshot = run_data.get("snapshot", {})

# ==========================================
# COLOR MAPPING (DEFINED AFTER ALTERNATIVES LOAD)
# ==========================================
ALTERNATIVE_COLORS = ["#38bdf8", "#ff4b4b", "#a855f7", "#22c55e", "#eab308", "#ec4899", "#06b6d4"]

def get_alternative_color(alt_name: str, alternatives_list: list) -> str:
    try:
        idx = alternatives_list.index(alt_name)
        return ALTERNATIVE_COLORS[idx % len(ALTERNATIVE_COLORS)]
    except ValueError:
        return "#38bdf8"

alt_color_map = {alt: get_alternative_color(alt, alternatives) for alt in alternatives}

def get_alternative_val(container: Any, alt_name: str, idx: int, default: Optional[float] = 0.0) -> Optional[float]:
    """Safely extracts score/rank whether stored as a list or dictionary."""
    if isinstance(container, dict):
        return container.get(alt_name, default)
    elif isinstance(container, (list, tuple, np.ndarray)):
        if 0 <= idx < len(container):
            return container[idx]
    return default

st.title(f"📊 Analytics Dashboard: `{run_name}`")
st.caption(f"Run ID: **{run_id}** | Timestamp: **{timestamp}** | Arena: **{' vs '.join(alternatives)}**")

st.markdown("---")

# Determine weighting architecture mode
def get_run_weight_system_mode(data) -> str:
    if "weight_system_mode" in data:
        return data["weight_system_mode"]
    cfg_file = get_rating_config_filepath()
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                return cfg.get("weight_system_mode", "Dual Hybrid (Categories & Criteria)")
        except Exception:
            pass
    return "Dual Hybrid (Categories & Criteria)"

weight_sys_mode = get_run_weight_system_mode(run_data)
is_flat_mode = ("Single Flat" in weight_sys_mode)

# ==========================================
# 1. DYNAMIC CONSENSUS & RANK AGREEMENT
# ==========================================
win_counts = {alt: 0 for alt in alternatives}
valid_models_count = 0
model_rank_data = []

for model_name, model_res in results.items():
    if not isinstance(model_res, dict) or model_res.get("status") != "success":
        continue
    
    ranks = model_res.get("ranking", model_res.get("ranks", None))
    
    if not ranks:
        for score_key in ["scores", "net_flows", "preference_flows", "values"]:
            if score_key in model_res:
                s_list = model_res[score_key]
                if isinstance(s_list, (list, tuple, np.ndarray)) and len(s_list) == len(alternatives):
                    arr = np.array(s_list, dtype=float)
                    sorted_indices = np.argsort(-arr)
                    calculated_ranks = [0] * len(alternatives)
                    for r_idx, orig_idx in enumerate(sorted_indices):
                        calculated_ranks[orig_idx] = r_idx + 1
                    ranks = calculated_ranks
                    break

    row = {"Model": model_name.upper()}
    current_winner = None
    min_rank = float("inf")

    for idx, alt in enumerate(alternatives):
        r_val = get_alternative_val(ranks, alt, idx, default=-1.0)
        if r_val != -1.0 and r_val is not None:
            r_int = int(round(float(r_val)))
            row[alt] = f"Rank #{r_int}"
            if r_int < min_rank:
                min_rank = r_int
                current_winner = alt
        else:
            row[alt] = "Rank #-"
            
    if current_winner:
        win_counts[current_winner] += 1
        valid_models_count += 1
        
    model_rank_data.append(row)

if valid_models_count > 0:
    winner = max(win_counts.keys(), key=lambda k: win_counts[k])
    confidence = (win_counts[winner] / valid_models_count) * 100
else:
    winner = "Undetermined"
    confidence = 0.0

col_w, col_ranks = st.columns([1, 2])

with col_w:
    winner_color = alt_color_map.get(winner, "#4A90E2")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
                border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; padding: 22px; text-align: center;">
        <div style="font-size: 0.85rem; text-transform: uppercase; color: #A0AEC0; letter-spacing: 1px; font-weight: 700;">
            Consensus Winner
        </div>
        <div style="font-size: 2.2rem; font-weight: 800; color: {winner_color}; margin: 8px 0;">
            🏆 {winner}
        </div>
        <div style="font-size: 0.95rem; color: #A0AEC0; font-weight: 600;">
            Model Agreement: <strong style="color: #FFFFFF;">{confidence:.0f}%</strong> ({win_counts.get(winner, 0)}/{valid_models_count} Models)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_ranks:
    st.subheader("Multi-Model Rank Agreement")
    st.dataframe(pd.DataFrame(model_rank_data), hide_index=True, use_container_width=True)

st.info(
    "💡 **How to interpret Consensus & Rank Agreement:** "
    "The Consensus Winner is determined by aggregating 1st-place votes across all successfully executed MCDM models (e.g., TOPSIS, WASPAS, WSM, WPM, Fuzzy PROMETHEE). "
    "The **Model Agreement %** measures robustness: a 100% agreement indicates that all mathematical paradigms unanimously favor the top alternative, minimizing model-selection bias."
)

st.markdown("---")

# ==========================================
# 2. MODEL SCORE COMPARISON CHART
# ==========================================
st.subheader("📈 Synthesized Preference Scores")

score_records = []
for model_name, model_res in results.items():
    if not isinstance(model_res, dict) or model_res.get("status") != "success":
        continue
    
    scores = []
    for skey in ["scores", "net_flows", "preference_flows", "values"]:
        if skey in model_res:
            scores = model_res[skey]
            break

    for idx, alt in enumerate(alternatives):
        raw_score = get_alternative_val(scores, alt, idx, default=0.0)
        score_records.append({
            "Model": model_name.upper(),
            "Alternative": alt,
            "Normalized Score": float(raw_score) if raw_score is not None else 0.0
        })

if score_records:
    df_scores = pd.DataFrame(score_records)
    fig_scores = px.bar(
        df_scores,
        x="Model",
        y="Normalized Score",
        color="Alternative",
        barmode="group",
        title="Synthesized Preference Score by Engine",
        template="plotly_dark",
        color_discrete_map=alt_color_map
    )
    
    fig_scores.update_traces(
        texttemplate="%{y:.3f}", 
        textposition="outside", 
        cliponaxis=False
    )
    
    fig_scores.update_layout(
        margin=dict(l=20, r=20, t=60, b=30), 
        legend=dict(orientation="h", y=1.15, x=0.3),
        yaxis=dict(rangemode="tozero")
    )
    st.plotly_chart(fig_scores, use_container_width=True)

st.info(
    "💡 **Understanding Synthesized Scores & Zero Values:** "
    "Different decision models utilize unique normalization and aggregation philosophies. "
    "For example, in **TOPSIS**, a score of **`0.000`** indicates that an alternative represents the baseline worst-case option "
    "(it sits directly on the negative ideal reference point). "
    "Text labels are explicitly forced outside the bars so zero-score alternatives remain clearly visible on the baseline."
)

st.markdown("---")

# ==========================================
# 3. SELECTABLE DOMAIN / CRITERIA BREAKDOWN
# ==========================================
if is_flat_mode:
    st.subheader("🧩 Criteria-by-Criteria Performance Breakdown")
    st.caption("Active Mode: **Single Flat Weighting** (Showing direct criteria contributions).")
else:
    st.subheader("🧩 Category-by-Category Domain Breakdown")
    st.caption("Active Mode: **Dual Hybrid** (Showing high-level category domain contributions).")

success_models = [m.upper() for m, res in results.items() if isinstance(res, dict) and res.get("status") == "success"]
breakdown_options = ["All-Model Average"] + success_models
selected_breakdown_model = st.selectbox("Select Model Perspective for Breakdown:", breakdown_options)

factors_cfg = snapshot.get("factors_config", {})
weights_data = snapshot.get("weights", {})
evaluations = snapshot.get("evaluations", [])

domains = factors_cfg.get("domains", [])
factors = factors_cfg.get("factors", [])
global_weights = weights_data.get("global_weights", {})

if factors and evaluations:
    if is_flat_mode:
        score_breakdown = {alt: {f["name"]: 0.0 for f in factors} for alt in alternatives}
        totals = {f["name"]: 0.0 for f in factors}

        for f in factors:
            fid = f["id"]
            f_name = f.get("name", fid)
            gw = float(global_weights.get(fid, 0.0))
            totals[f_name] = gw
            
            for alt in alternatives:
                ev = next((e for e in evaluations if e.get("alternative", e.get("country")) == alt and e.get("criterion_id") == fid), None)
                if ev:
                    r = float(ev.get("rating", 5.0))
                    score_breakdown[alt][f_name] = r * gw

        if selected_breakdown_model != "All-Model Average":
            model_key = selected_breakdown_model.lower()
            actual_res = next((res for m, res in results.items() if m.lower() == model_key), {})
            scores_list = []
            for skey in ["scores", "net_flows", "preference_flows", "values"]:
                if skey in actual_res:
                    scores_list = actual_res[skey]
                    break
            
            for idx, alt in enumerate(alternatives):
                final_model_score = float(get_alternative_val(scores_list, alt, idx, default=0.0) or 0.0)
                curr_sum = sum(score_breakdown[alt].values())
                if curr_sum > 0:
                    sf = final_model_score / curr_sum
                    for f_name in score_breakdown[alt]:
                        score_breakdown[alt][f_name] *= sf
        else:
            valid_models = [res for m, res in results.items() if isinstance(res, dict) and res.get("status") == "success"]
            if valid_models:
                for idx, alt in enumerate(alternatives):
                    alt_model_scores = []
                    for res in valid_models:
                        s_list = []
                        for skey in ["scores", "net_flows", "preference_flows", "values"]:
                            if skey in res:
                                s_list = res[skey]
                                break
                        alt_model_scores.append(float(get_alternative_val(s_list, alt, idx, default=0.0) or 0.0))
                    avg_score = sum(alt_model_scores) / len(alt_model_scores) if alt_model_scores else 0.0
                    curr_sum = sum(score_breakdown[alt].values())
                    sf = (avg_score / curr_sum) if curr_sum > 0 else 1.0
                    for f_name in score_breakdown[alt]:
                        score_breakdown[alt][f_name] *= sf

        decomp_rows = []
        for f_name in totals.keys():
            for alt in alternatives:
                decomp_rows.append({
                    "Criterion": f_name,
                    "Alternative": alt,
                    "Weighted Contribution": score_breakdown[alt].get(f_name, 0.0)
                })
        df_decomp = pd.DataFrame(decomp_rows)
        fig_decomp = px.bar(
            df_decomp,
            x="Criterion",
            y="Weighted Contribution",
            color="Alternative",
            barmode="group",
            template="plotly_dark",
            title=f"Weighted Criteria Performance ({selected_breakdown_model} Perspective)",
            color_discrete_map=alt_color_map
        )
        fig_decomp.update_layout(margin=dict(l=20, r=20, t=40, b=40), xaxis_tickangle=-25)
        st.plotly_chart(fig_decomp, use_container_width=True)

    else:
        valid_domains = [d for d in domains if d["id"] != "d01"] if domains else []
        if not valid_domains:
            valid_domains = domains

        domain_scores = {alt: {d["name"]: 0.0 for d in valid_domains} for alt in alternatives}
        domain_totals = {d["name"]: 0.0 for d in valid_domains}

        for f in factors:
            fid = f["id"]
            dom_id = f.get("domain_id")
            d_obj = next((d for d in valid_domains if d["id"] == dom_id), None)
            if not d_obj:
                continue
            d_name = d_obj["name"]
            gw = float(global_weights.get(fid, 0.0))
            domain_totals[d_name] = domain_totals.get(d_name, 0.0) + gw
            
            for alt in alternatives:
                ev = next((e for e in evaluations if e.get("alternative", e.get("country")) == alt and e.get("criterion_id") == fid), None)
                if ev:
                    r = float(ev.get("rating", 5.0))
                    if d_name in domain_scores[alt]:
                        domain_scores[alt][d_name] += r * gw

        if selected_breakdown_model != "All-Model Average":
            model_key = selected_breakdown_model.lower()
            actual_res = next((res for m, res in results.items() if m.lower() == model_key), {})
            scores_list = []
            for skey in ["scores", "net_flows", "preference_flows", "values"]:
                if skey in actual_res:
                    scores_list = actual_res[skey]
                    break
            
            for idx, alt in enumerate(alternatives):
                final_model_score = float(get_alternative_val(scores_list, alt, idx, default=0.0) or 0.0)
                current_alt_sum = sum(domain_scores[alt].values())
                
                if current_alt_sum > 0:
                    scale_factor = final_model_score / current_alt_sum
                    for d_name in domain_scores[alt]:
                        domain_scores[alt][d_name] *= scale_factor
        else:
            valid_models = [res for m, res in results.items() if isinstance(res, dict) and res.get("status") == "success"]
            if valid_models:
                avg_scale_factors = {}
                for idx, alt in enumerate(alternatives):
                    alt_model_scores = []
                    for res in valid_models:
                        s_list = []
                        for skey in ["scores", "net_flows", "preference_flows", "values"]:
                            if skey in res:
                                s_list = res[skey]
                                break
                        alt_model_scores.append(float(get_alternative_val(s_list, alt, idx, default=0.0) or 0.0))
                    avg_score = sum(alt_model_scores) / len(alt_model_scores) if alt_model_scores else 0.0
                    curr_sum = sum(domain_scores[alt].values())
                    avg_scale_factors[alt] = (avg_score / curr_sum) if curr_sum > 0 else 1.0
                
                for alt in alternatives:
                    sf = avg_scale_factors[alt]
                    for d_name in domain_scores[alt]:
                        domain_scores[alt][d_name] *= sf

        decomp_rows = []
        for d_name in domain_totals.keys():
            for alt in alternatives:
                decomp_rows.append({
                    "Domain": d_name,
                    "Alternative": alt,
                    "Weighted Contribution": domain_scores[alt].get(d_name, 0.0)
                })
        
        df_decomp = pd.DataFrame(decomp_rows)
        fig_decomp = px.bar(
            df_decomp,
            x="Domain",
            y="Weighted Contribution",
            color="Alternative",
            barmode="group",
            template="plotly_dark",
            title=f"Weighted Domain Performance ({selected_breakdown_model} Perspective)",
            color_discrete_map=alt_color_map
        )
        fig_decomp.update_layout(margin=dict(l=20, r=20, t=40, b=40), xaxis_tickangle=-25)
        st.plotly_chart(fig_decomp, use_container_width=True)

st.info(
    "💡 **Interpreting Performance Breakdowns:** "
    "This chart dissects how each category (in Dual mode) or individual criterion (in Single Flat mode) contributes to an alternative's final score. "
    "By selecting a specific model perspective or the All-Model Average, you can pinpoint exactly which strengths drive an alternative's victory and where its vulnerabilities lie."
)

# ==========================================
# 4. RAW SNAPSHOT MATRIX VIEWER
# ==========================================
st.markdown("---")
with st.expander("🔍 View Raw Snapshot Decision Matrix & Fuzzy Bounds", expanded=False):
    st.markdown("Inspect the exact evaluation parameters, base ratings, volatility scores, epistemic uncertainty, psychological bias shifts, and computed trapezoidal fuzzy bounds $[a, b, c, d]$ captured at the moment this run was executed.")
    if evaluations:
        eval_df = pd.DataFrame(evaluations)
        if "trapezoid" in eval_df.columns:
            eval_df["Trap [a, b, c, d]"] = eval_df["trapezoid"].apply(
                lambda t: f"[{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}, {t[3]:.1f}]" if isinstance(t, (list, tuple)) and len(t) == 4 else str(t)
            )
        st.dataframe(eval_df, use_container_width=True)
    else:
        st.write("No evaluation snapshot stored in this run file.")