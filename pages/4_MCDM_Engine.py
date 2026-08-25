"""
Decision Support System - Multi-Model MCDM Engine Page
Orchestrates deterministic MCDM models (WSM, WPM, WASPAS, TOPSIS) 
and Fuzzy PROMETHEE, excluding VIKOR, with streamlined run analysis and history.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import contextlib
import io

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.mcdm_methods import METHOD_REGISTRY
from src.mcdm_engine import execute_run, RUNS_DIR
from src.mcdm_fuzzy_engine import execute_fuzzy_run

st.set_page_config(page_title="MCDM Engine", page_icon="🏆", layout="wide")
st.title("🏆 Multi-Model MCDM Engine & Orchestrator")
st.caption("Execute deterministic and fuzzy decision models, evaluate consensus rankings, and compare historical decision snapshots.")

if not os.path.exists(RUNS_DIR):
    os.makedirs(RUNS_DIR)

# ==========================================
# HELPER: RENDER A SINGLE RUN
# ==========================================
def render_run_dashboard(run_data):
    st.subheader(f"Results: {run_data['name']}")
    st.caption(f"Run ID: `{run_data['run_id']}` | 🕒 {run_data['timestamp'][:16].replace('T', ' ')}")
    
    alternatives = run_data.get("alternatives", run_data.get("countries", []))
    results = run_data["results"]
    
    table_rows = []
    win_counts = {alt: 0 for alt in alternatives}
    errors = []
    
    for m_name, res in results.items():
        if "vikor" in m_name.lower():
            continue
            
        if res.get("status") == "success":
            scores = res["scores"]
            ranks = res["ranking"]
            winner_idx = np.argmin(ranks)
            winner_name = alternatives[winner_idx]
            
            row = {"Method": m_name}
            for i, alt in enumerate(alternatives):
                rank = int(ranks[i])
                if rank == 1: 
                    win_counts[alt] += 1
                marker = " 🏆" if rank == 1 else ""
                row[alt] = f"{scores[i]:.4f} (R{rank}){marker}"
                
            row["Winner"] = winner_name
            table_rows.append(row)
            
        elif res.get("status") == "not_implemented":
            row = {"Method": m_name, "Winner": "Pending"}
            for alt in alternatives: 
                row[alt] = "-"
            table_rows.append(row)
        else:
            row = {"Method": m_name, "Winner": "Error"}
            for alt in alternatives: 
                row[alt] = "Error"
            table_rows.append(row)
            errors.append(f"**{m_name}:** {res.get('warnings', ['Unknown error'])[0]}")
            
    st.markdown("#### 1. Method Scores & Rankings")
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
    
    if errors:
        st.error("Execution Errors Encountered:")
        for e in errors: 
            st.write(f"- {e}")
        
    st.markdown("#### 2. Agreement Summary (1st Place Votes)")
    total_successful = sum(1 for m_name, r in results.items() if "vikor" not in m_name.lower() and r.get("status") == "success")
    
    if total_successful > 0:
        summary_rows = []
        for alt in alternatives:
            wins = win_counts[alt]
            pct = (wins / total_successful) * 100 if total_successful > 0 else 0
            summary_rows.append({"Alternative": alt, "1st Place Ranks": wins, "Agreement": f"{pct:.1f}%"})
            
        sum_df = pd.DataFrame(summary_rows).sort_values("1st Place Ranks", ascending=False)
        st.dataframe(sum_df, hide_index=True, use_container_width=False)
    else:
        st.warning("No runs completed successfully.")
        
    with st.expander("View Weight Distribution Used for this Run"):
        cat_weights = run_data.get("category_weights", run_data.get("domain_weights", {}))
        if cat_weights:
            cat_df = pd.DataFrame([{"Category": k, "Weight (%)": v} for k, v in cat_weights.items()])
            cat_df = cat_df.sort_values("Weight (%)", ascending=False)
            st.dataframe(
                cat_df, 
                column_config={"Weight (%)": st.column_config.ProgressColumn("Weight (%)", max_value=100, format="%.1f%%")},
                hide_index=True
            )

# ==========================================
# UI TABS (Run Analysis & History Only)
# ==========================================
tab_run, tab_history = st.tabs(["🚀 Run Analysis", "📚 History & Comparison"])

# ------------------------------------------
# TAB 1: RUN ANALYSIS
# ------------------------------------------
with tab_run:
    st.header("Execute New MCDM Synthesis")
    
    all_methods = [m for m in METHOD_REGISTRY.keys() if "vikor" not in m.lower()]
    det_methods = [m for m, obj in METHOD_REGISTRY.items() if obj.method_type == "deterministic" and "vikor" not in m.lower()]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_methods = st.multiselect(
            "Select Methods to Execute:", 
            options=all_methods, 
            default=det_methods,
            help="Choose which decision aggregation models to run. Deterministic and fuzzy methods automatically route to their respective calculation engines."
        )
    with col2:
        raw_run_name = st.text_input(
            "Run Name (optional):", 
            value="Multi_Model_Run",
            help="A custom label to identify this evaluation snapshot in your history archive."
        )
        run_name = (raw_run_name or "Multi_Model_Run").strip()
        
    if st.button("🚀 Calculate Decision Matrix", type="primary"):
        if not selected_methods:
            st.warning("Please select at least one method.")
        else:
            with st.spinner("Executing engines and synthesizing models..."):
                det_to_run = [m for m in selected_methods if METHOD_REGISTRY[m].method_type == "deterministic"]
                fuz_to_run = [m for m in selected_methods if METHOD_REGISTRY[m].method_type == "fuzzy"]
                
                f = io.StringIO()
                snap_det, snap_fuz = None, None
                
                with contextlib.redirect_stdout(f):
                    if det_to_run:
                        snap_det = execute_run(det_to_run, run_name)
                    if fuz_to_run:
                        snap_fuz = execute_fuzzy_run(fuz_to_run, run_name)
                
                if snap_det and snap_fuz:
                    merged_snap = snap_det.copy()
                    merged_snap["methods_executed"].extend(snap_fuz["methods_executed"])
                    merged_snap["results"].update(snap_fuz["results"])
                    
                    if "parameters" in snap_fuz:
                        merged_snap["parameters"].update(snap_fuz["parameters"])
                        
                    with open(os.path.join(RUNS_DIR, f"{merged_snap['run_id']}.json"), 'w', encoding='utf-8') as f_out:
                        json.dump(merged_snap, f_out, indent=4)
                        
                    try:
                        os.remove(os.path.join(RUNS_DIR, f"{snap_fuz['run_id']}.json"))
                    except OSError:
                        pass
                        
                    st.success("✅ Unified Hybrid Run saved successfully!")
                    render_run_dashboard(merged_snap)
                    
                elif snap_det:
                    st.success("✅ Deterministic Run saved successfully!")
                    render_run_dashboard(snap_det)
                    
                elif snap_fuz:
                    st.success("✅ Fuzzy Run saved successfully!")
                    render_run_dashboard(snap_fuz)
                    
                else:
                    st.error("Execution failed. Check validation errors below:")
                    st.code(f.getvalue())

# ------------------------------------------
# TAB 2: HISTORY & COMPARISON
# ------------------------------------------
with tab_history:
    st.header("Compare Saved Decision Runs")
    
    saved_runs = []
    for file in os.listdir(RUNS_DIR):
        if file.endswith('.json'):
            with open(os.path.join(RUNS_DIR, file), 'r', encoding='utf-8') as f:
                saved_runs.append(json.load(f))
                
    saved_runs.sort(key=lambda x: x["timestamp"], reverse=True)
    
    if not saved_runs:
        st.info("No saved runs found. Go to 'Run Analysis' to create one.")
    else:
        run_options = {f"{r['name']} ({r['timestamp'][:16].replace('T', ' ')})": r for r in saved_runs}
        
        selected_run_labels = st.multiselect(
            "Select multiple runs to compare side-by-side:",
            options=list(run_options.keys()),
            default=[list(run_options.keys())[0]],
            help="Compare ranking stability and alternative scores across different weighting or evaluation sets."
        )
        
        runs_to_compare = [run_options[label] for label in selected_run_labels]
        
        if len(runs_to_compare) == 1:
            st.markdown("---")
            render_run_dashboard(runs_to_compare[0])
            
            if st.button("🗑️ Delete this Run", type="secondary"):
                target_id = runs_to_compare[0]["run_id"]
                os.remove(os.path.join(RUNS_DIR, f"{target_id}.json"))
                st.success("Run deleted successfully! Refreshing...")
                st.rerun()
                
        elif len(runs_to_compare) > 1:
            st.markdown("---")
            st.subheader("Side-by-Side Comparison Matrix")
            
            first_run = runs_to_compare[0]
            alternatives = first_run.get("alternatives", first_run.get("countries", []))
            comp_rows = []
            
            for run in runs_to_compare:
                run_alts = run.get("alternatives", run.get("countries", []))
                for m_name, res in run["results"].items():
                    if "vikor" in m_name.lower():
                        continue
                    if res.get("status") == "success":
                        scores = res["scores"]
                        ranks = res["ranking"]
                        winner_idx = np.argmin(ranks)
                        winner_name = run_alts[winner_idx]
                        
                        row = {"Method": m_name, "Run Name": run["name"]}
                        for alt in alternatives:
                            try:
                                c_idx = run_alts.index(alt)
                                row[alt] = f"{scores[c_idx]:.4f} (R{int(ranks[c_idx])})"
                            except ValueError:
                                row[alt] = "N/A"
                        row["Winner"] = winner_name
                        comp_rows.append(row)
                        
            if comp_rows:
                comp_df = pd.DataFrame(comp_rows).sort_values(by=["Method", "Run Name"])
                st.dataframe(comp_df, hide_index=True, use_container_width=True)
            else:
                st.warning("No successful data to compare between these selected runs.")