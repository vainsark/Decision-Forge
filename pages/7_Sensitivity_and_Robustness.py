"""
Decision Support System - Sensitivity, Robustness, Epistemic & Monte Carlo UI
Includes weight-architecture awareness (Dual vs Single Flat), parameter descriptions, 
graph guides, Tornado leverage, multi-model epistemic propagation, and Monte Carlo stochastic simulation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.analysis.dispatcher import AnalysisDispatcher
from src.analysis.sensitivity import SensitivityEngine, DEFUZZ_WEIGHT_SCHEMES
from src.analysis.robustness import RobustnessEngine
from src.analysis.epistemic import EpistemicEngine, generate_discrete_grid
from src.analysis.monte_carlo import MonteCarloEngine
from src.mcdm_methods import METHOD_REGISTRY
from src.project_manager import get_active_project_dir
from src.project_manager import get_active_project_id


# Dynamic path resolution functions for active project workspace isolation
def _get_project_data_dir() -> str:
    """Returns the active project directory, asserting it is not None."""
    proj_dir = get_active_project_dir()
    assert proj_dir is not None, "Active project directory is required."
    return proj_dir

def get_rating_config_filepath() -> str:
    return os.path.join(_get_project_data_dir(), 'rating_config.json')

st.set_page_config(page_title="Sensitivity & Robustness", page_icon="🔬", layout="wide")
st.title("🔬 Sensitivity & Robustness Engine")
st.caption("Probe model stability, test factor leverage, propagate epistemic uncertainty, and simulate stochastic win probabilities.")

# ==========================================
# ACTIVE PROJECT GUARD
# ==========================================
active_proj_id = get_active_project_id()
if not active_proj_id:
    st.warning("⚠️ **No Active Project Selected.** Please go to the **Decision Hub** to create or open a project workspace.")
    if st.button("🗂️ Go to Decision Hub", type="primary"):
        st.switch_page("pages/0_Decision_Hub.py")
    st.stop()

RATING_CONFIG_FILE = get_rating_config_filepath()

def get_weight_system_mode() -> str:
    config_file = get_rating_config_filepath()
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                return cfg.get("weight_system_mode", "Dual Hybrid (Categories & Criteria)")
        except Exception:
            pass
    return "Dual Hybrid (Categories & Criteria)"

weight_sys_mode = get_weight_system_mode()
is_flat_mode = ("Single Flat" in weight_sys_mode)

# =========================================================================
# DYNAMIC COLOR UTILITIES
# =========================================================================
ALTERNATIVE_COLORS = ["#38bdf8", "#ff4b4b", "#a855f7", "#22c55e", "#eab308", "#ec4899", "#06b6d4"]

def get_alternative_color(alt_name: str, alternatives_list: list) -> str:
    try:
        idx = alternatives_list.index(alt_name)
        return ALTERNATIVE_COLORS[idx % len(ALTERNATIVE_COLORS)]
    except ValueError:
        return "#38bdf8"

# =========================================================================
# PLOTLY CHART BUILDERS (GENERALIZED)
# =========================================================================
def plot_sensitivity_curves(sens_data: dict, method_name: str, alternatives: list) -> go.Figure:
    meta = sens_data["metadata"]
    iterations = sens_data.get("iterations", [])
    dim = meta.get("dimension", "")

    x_vals = []
    y_series = {c: [] for c in alternatives}

    for it in iterations:
        p_val = it["param_value"]
        m_res = it.get("method_results", {}).get(method_name, {})
        if m_res.get("status") == "success":
            x_vals.append(p_val)
            for c in alternatives:
                y_series[c].append(m_res["scores"].get(c, np.nan))

    fig = go.Figure()
    for alt in alternatives:
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_series[alt], mode='lines+markers', name=alt,
            line=dict(width=3, color=get_alternative_color(alt, alternatives)), marker=dict(size=7)
        ))

    base_val = meta.get("baseline_value")
    if base_val is not None and isinstance(base_val, (int, float)):
        fig.add_vline(
            x=base_val, line_dash="dash", line_color="gray",
            annotation_text=f"Baseline ({base_val*100:.1f}%)" if "weight" in dim else f"Baseline ({base_val})",
            annotation_position="top left"
        )

    fig.update_layout(
        title=f"<b>{method_name}</b>: Score Sensitivity Trajectory",
        xaxis_title="Factor Weight (Fraction)" if "weight" in dim else "Parameter Value",
        yaxis_title="Net Preference Flow (Φ)" if "promethee" in method_name.lower() else "Evaluation Score",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_scheme_bar_chart(sens_data: dict, method_name: str, alternatives: list) -> go.Figure:
    iterations = sens_data.get("iterations", [])
    schemes = []
    country_scores = {c: [] for c in alternatives}

    for it in iterations:
        schemes.append(it["param_value"])
        m_res = it.get("method_results", {}).get(method_name, {})
        for c in alternatives:
            country_scores[c].append(m_res["scores"].get(c, 0.0) if m_res.get("status") == "success" else 0.0)

    fig = go.Figure()
    for c in alternatives:
        fig.add_trace(go.Bar(name=c, x=schemes, y=country_scores[c], marker=dict(color=get_alternative_color(c, alternatives))))

    fig.update_layout(
        barmode='group', title=f"<b>{method_name}</b>: Performance by Defuzzification Scheme",
        yaxis_title="Net Preference Flow (Φ)" if "promethee" in method_name.lower() else "Score",
        template="plotly_white", xaxis_tickangle=-25, margin=dict(l=40, r=40, t=60, b=80)
    )
    return fig


def plot_qp_heatmap(sens_data: dict) -> go.Figure:
    iterations = sens_data.get("iterations", [])
    records = []

    for it in iterations:
        q_val = it.get("q")
        p_val = it.get("p")
        m_res = it.get("method_results", {}).get("Fuzzy PROMETHEE", {})
        if m_res.get("status") == "success" and q_val is not None:
            scores = sorted(m_res["scores"].values(), reverse=True)
            margin = (scores[0] - scores[1]) if len(scores) >= 2 else 0.0
            winner = m_res.get("winner", "N/A")
            records.append({"q": q_val, "p": p_val, "margin": round(margin, 4), "winner": winner})

    if not records:
        return go.Figure()

    df = pd.DataFrame(records)
    pivot_margin = df.pivot(index="p", columns="q", values="margin")
    pivot_winner = df.pivot(index="p", columns="q", values="winner")

    fig = go.Figure(data=go.Heatmap(
        z=pivot_margin.values,
        x=[f"q = {col}" for col in pivot_margin.columns],
        y=[f"p = {idx}" for idx in pivot_margin.index],
        text=pivot_winner.values,
        texttemplate="%{text}<br>Margin: %{z:.3f}",
        colorscale="Blues", colorbar=dict(title="Victory Margin")
    ))

    fig.update_layout(
        title="<b>Fuzzy PROMETHEE</b>: (q, p) Decision Margin & Winner Space",
        xaxis_title="Indifference Threshold (q)", yaxis_title="Preference Threshold (p)",
        template="plotly_white", margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_category_tornado(tornado_data: dict, is_flat: bool) -> go.Figure:
    entries = tornado_data.get("tornado_entries", [])
    meta = tornado_data.get("metadata", {})
    winner = meta.get("baseline_winner", "")
    method = meta.get("method_name", "")
    pct_str = f"±{int(meta.get('perturbation_fraction', 0.5)*100)}%"

    sorted_entries = list(reversed(entries))
    # Use safe fallback to support both domain_name and criterion_name
    labels = [e.get("domain_name", e.get("criterion_name", "Factor")) for e in sorted_entries]
    deltas_low = [e["delta_low"] for e in sorted_entries]
    deltas_high = [e["delta_high"] for e in sorted_entries]

    item_label = "Criterion" if is_flat else "Category"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=deltas_low, orientation='h',
        name=f"Decrease Weight ({pct_str})", marker=dict(color="#ef4444"),
        hovertemplate=f"{item_label}: %{{y}}<br>Δ Score: %{{x:.4f}}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=labels, x=deltas_high, orientation='h',
        name=f"Increase Weight ({pct_str})", marker=dict(color="#22c55e"),
        hovertemplate=f"{item_label}: %{{y}}<br>Δ Score: %{{x:.4f}}<extra></extra>"
    ))
    fig.add_vline(x=0.0, line_dash="solid", line_color="#333333", line_width=1.5)
    fig.update_layout(
        barmode='relative',
        title=f"<b>{item_label} Leverage Tornado Chart</b>: Impact on <b>{winner}</b> ({method})",
        xaxis_title=f"Δ Score Shift from Baseline (under {pct_str} weight shift)",
        yaxis=dict(title=item_label, automargin=True),
        template="plotly_white", height=max(400, len(labels) * 45),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_boundary_range(thresh: dict, alternatives: list, is_flat: bool) -> go.Figure:
    base_w = thresh["baseline_weight"] * 100
    safe_min = thresh["safe_stability_range"][0] * 100
    safe_max = thresh["safe_stability_range"][1] * 100
    # Use safe fallback for domain vs criterion name keys
    domain = thresh.get("target_domain_name", thresh.get("target_criterion_name", "Factor"))
    winner = thresh["baseline_winner"]
    method = thresh["method"]
    
    other_alternatives = [a for a in alternatives if a != winner]
    other_country = other_alternatives[0] if other_alternatives else "Runner-up"
    
    alt_color_map = {alt: get_alternative_color(alt, alternatives) for alt in alternatives}
    winner_color = alt_color_map.get(winner, "#38bdf8")
    other_color = alt_color_map.get(other_country, "#ff4b4b")

    item_label = "Criterion" if is_flat else "Category"

    fig = go.Figure()

    if safe_min > 0.0:
        fig.add_trace(go.Bar(
            x=[safe_min], base=[0.0], y=[domain], orientation='h',
            marker=dict(color=other_color, opacity=0.85),
            name=f"Winner: {other_country} (0%–{safe_min:.1f}%)",
            text=f"{other_country} wins", textposition="inside"
        ))

    fig.add_trace(go.Bar(
        x=[safe_max - safe_min], base=[safe_min], y=[domain], orientation='h',
        marker=dict(color=winner_color, opacity=0.9),
        name=f"Safe Zone / Winner: {winner} ({safe_min:.1f}%–{safe_max:.1f}%)",
        text=f"{winner} wins (Safe Zone)", textposition="inside"
    ))

    if safe_max < 100.0:
        fig.add_trace(go.Bar(
            x=[100.0 - safe_max], base=[safe_max], y=[domain], orientation='h',
            marker=dict(color=other_color, opacity=0.85),
            name=f"Winner: {other_country} ({safe_max:.1f}%–100%)",
            text=f"{other_country} wins (Flipped)", textposition="inside"
        ))

    fig.add_trace(go.Scatter(
        x=[base_w], y=[domain], mode='markers+text',
        marker=dict(size=16, color="#ffffff", symbol="diamond", line=dict(width=2, color="#333333")),
        text=[f"  Current Baseline ({base_w:.1f}%)"], textposition="top center",
        name="Baseline Weight", textfont=dict(size=12, color="black")
    ))

    if safe_max < 100.0:
        fig.add_vline(
            x=safe_max, line_dash="dash", line_color="#111111", line_width=2,
            annotation_text=f"Flip Threshold: {safe_max:.1f}%", annotation_position="top right"
        )

    fig.update_layout(
        barmode='overlay',
        title=f"<b>Decision Stability Zones:</b> {domain} ({method})",
        xaxis=dict(title=f"{item_label} Weight (%)", range=[0, 100], ticksuffix="%"),
        yaxis=dict(showticklabels=False),
        template="plotly_white", height=240, margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="right", x=1)
    )
    return fig


def plot_epistemic_leverage_chart(criteria_results: list, countries: list, method_name: str = "") -> go.Figure:
    top_25 = criteria_results[:25]
    sorted_entries = list(reversed(top_25))
    names = [f"{e['short_name']} ({e['category_name'][:14]})" for e in sorted_entries]
    shifts = [e["max_abs_advantage_shift"] for e in sorted_entries]
    
    color_map = {"Winner Flip": "#ef4444", "Material Shift": "#f59e0b", "Negligible": "#94a3b8"}
    bar_colors = [color_map.get(e["classification"], "#94a3b8") for e in sorted_entries]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=shifts, orientation='h', marker=dict(color=bar_colors),
        hovertemplate="Criterion: %{y}<br>Max Advantage Shift: %{x:.4f}<extra></extra>"
    ))

    c0 = countries[0] if countries else "C1"
    c1 = countries[1] if len(countries) > 1 else "C2"
    m_title = f" [{method_name}]" if method_name else ""

    fig.update_layout(
        title=f"<b>Ranked Epistemic Leverage{m_title}:</b> Max Decision Advantage Shift ({c0} vs {c1})",
        xaxis_title="Max Absolute Shift in Decision Advantage (|Δ Advantage|)",
        yaxis=dict(title="Criterion", automargin=True),
        template="plotly_white", height=max(400, len(names) * 28), margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_epistemic_trajectory(crit_entry: dict, countries: list) -> go.Figure:
    trajs = crit_entry.get("trajectories", [])
    c_name = crit_entry["criterion_name"]
    base_adv = crit_entry["baseline_advantage"]

    fig = go.Figure()
    for c in countries:
        c_pts = [t for t in trajs if t["perturbed_country"] == c]
        if c_pts:
            x_vals = [t["tested_rating"] for t in c_pts]
            y_vals = [t["advantage"] for t in c_pts]
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='lines+markers',
                name=f"Varying {c} Rating (E={crit_entry['evaluations'].get(c, {}).get('E', 0)})", marker=dict(size=8)
            ))

    fig.add_hline(y=0.0, line_dash="solid", line_color="#ef4444", line_width=2, annotation_text="Decision Flip Boundary (Advantage = 0)", annotation_position="bottom right")
    fig.add_hline(y=base_adv, line_dash="dash", line_color="gray", annotation_text=f"Baseline Adv ({base_adv:.4f})", annotation_position="top left")

    c0 = countries[0] if countries else "C1"
    c1 = countries[1] if len(countries) > 1 else "C2"

    fig.update_layout(
        title=f"<b>Epistemic Trajectory:</b> {c_name}",
        xaxis_title="Realized Point Rating (r ∈ [r - Ke*E, r + Ke*E])",
        yaxis_title=f"Decision Advantage (Positive Favors {c0}, Negative Favors {c1})",
        template="plotly_white", hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_combined_scenarios_chart(scenarios: list, countries: list, method_name: str = "") -> go.Figure:
    names = [s["scenario_name"] for s in scenarios]
    advs = [s["decision_advantage"] for s in scenarios]
    winners = [s["winner"] for s in scenarios]

    c0 = countries[0] if countries else "C1"
    c1 = countries[1] if len(countries) > 1 else "C2"
    
    alt_color_map = {alt: get_alternative_color(alt, countries) for alt in countries}
    colors = [alt_color_map.get(w, "#38bdf8") for w in winners]
    m_title = f" [{method_name}]" if method_name else ""

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=advs, marker=dict(color=colors),
        text=[f"Winner: {w}<br>Adv: {a:+.4f}" for w, a in zip(winners, advs)],
        textposition="outside", hovertemplate="Scenario: %{x}<br>Decision Advantage: %{y:.4f}<extra></extra>"
    ))

    fig.add_hline(y=0.0, line_dash="solid", line_color="#333333", line_width=1.5)
    fig.update_layout(
        title=f"<b>Combined Epistemic Scenarios{m_title}:</b> Decision Advantage ({c0} vs {c1})",
        xaxis=dict(title="Scenario", tickangle=-15),
        yaxis_title=f"Decision Advantage (Positive = {c0}, Negative = {c1})",
        template="plotly_white", height=480, margin=dict(l=40, r=40, t=60, b=80)
    )
    return fig


def plot_mc_win_rates(country_stats: dict, alternatives: list) -> go.Figure:
    labels = list(country_stats.keys())
    values = [country_stats[c]["win_percentage"] for c in labels]
    alt_color_map = [get_alternative_color(c, alternatives) for c in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=.45,
        textinfo='label+percent', marker=dict(colors=alt_color_map)
    )])
    fig.update_layout(
        title=f"<b>Probabilistic Win Rate Distribution</b>",
        template="plotly_white", height=350, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_mc_advantage_histogram(advantage_subsample: list, countries: list, method_name: str) -> go.Figure:
    c0 = countries[0] if countries else "C1"
    c1 = countries[1] if len(countries) > 1 else "C2"

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=advantage_subsample, nbinsx=40,
        marker=dict(color="#38bdf8", line=dict(width=1, color="black")),
        opacity=0.75, name="Simulation Runs"
    ))
    fig.add_vline(x=0.0, line_dash="solid", line_color="#ef4444", line_width=2, annotation_text="Flip Boundary (Advantage = 0)", annotation_position="top right")
    
    fig.update_layout(
        title=f"<b>Decision Advantage Distribution:</b> {method_name} (Positive = {c0}, Negative = {c1})",
        xaxis_title=f"Decision Advantage ({c0} Score - {c1} Score)",
        yaxis_title="Simulated Frequency",
        template="plotly_white", height=350, margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


# =========================================================================
# 1. BASELINE RUN SELECTOR
# =========================================================================
runs = AnalysisDispatcher.list_saved_runs()
if not runs:
    st.warning("⚠️ No saved runs found in this project's runs directory. Please execute and save a baseline run first.")
    st.stop()

run_options = {
    f"{r['name']} ({r['timestamp'][:16].replace('T', ' ')}) — [ID: {r['run_id'][:8]}]": r['run_id']
    for r in runs
}

default_idx = 0
if "target_baseline_run_id" in st.session_state:
    target_id = st.session_state["target_baseline_run_id"]
    for idx, (label, rid) in enumerate(run_options.items()):
        if rid == target_id:
            default_idx = idx
            break

st.sidebar.header("📁 Baseline Selection")
selected_run_label = st.sidebar.selectbox("Select Baseline MCDM Run:", list(run_options.keys()), index=default_idx)
selected_run_id = run_options[selected_run_label]

baseline_run = AnalysisDispatcher.load_baseline_run(selected_run_id)
ctx = AnalysisDispatcher.build_in_memory_context()

countries = baseline_run.get("countries", [])
executed_methods = baseline_run.get("methods_executed", [])
baseline_results = baseline_run.get("results", {})

# Summary Header Cards
with st.container():
    st.markdown("### 📌 Baseline Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Baseline Name", baseline_run.get("name", "Unnamed"))
    with c2:
        st.metric("Alternatives Pool", f"{len(countries)} Alternatives")
    with c3:
        st.metric("Methods Executed", f"{len(executed_methods)} Models")
    with c4:
        win_counts = {}
        for m, r in baseline_results.items():
            if r.get("status") == "success":
                w_idx = int(np.argmin(r.get("ranking", [0])))
                w_name = countries[w_idx]
                win_counts[w_name] = win_counts.get(w_name, 0) + 1
        top_winner = max(win_counts.keys(), key=lambda k: win_counts[k]) if win_counts else "N/A"
        st.metric("Consensus Winner", top_winner)

st.markdown("---")

# =========================================================================
# 2. MAIN TABS (4 CONSOLIDATED TABS)
# =========================================================================
tab_sens, tab_rob, tab_epistemic, tab_mc = st.tabs([
    "📊 Sensitivity Analysis", 
    "🛡️ Robustness & Decision Boundaries",
    "🧬 Epistemic Uncertainty Propagation",
    "🎲 Monte Carlo Simulation"
])

# -------------------------------------------------------------------------
# TAB 1: SENSITIVITY ANALYSIS
# -------------------------------------------------------------------------
with tab_sens:
    saved_sens_exps = AnalysisDispatcher.list_saved_analysis_experiments(filter_type="sensitivity")
    if saved_sens_exps:
        with st.expander("📂 Load a Saved Sensitivity Experiment", expanded=False):
            s_col1, s_col2, s_col3 = st.columns([3, 1, 1])
            with s_col1:
                sens_map = {f"{e['saved_name']} ({e['saved_timestamp'][:16].replace('T', ' ')})": e["analysis_id"] for e in saved_sens_exps}
                sel_sens_lbl = st.selectbox("Select Saved Sensitivity Run:", list(sens_map.keys()), key="load_sens_sel")
                sel_sens_id = sens_map[sel_sens_lbl]
            with s_col2:
                st.write("")
                st.write("")
                if st.button("📖 Load", key="load_sens_btn", type="primary"):
                    st.session_state["last_sensitivity_result"] = AnalysisDispatcher.load_saved_analysis_experiment(sel_sens_id)
                    st.success("Loaded successfully!")
                    st.rerun()
            with s_col3:
                st.write("")
                st.write("")
                if st.button("🗑️ Delete", key="del_sens_btn"):
                    AnalysisDispatcher.delete_saved_analysis_experiment(sel_sens_id)
                    st.success("Deleted!")
                    st.rerun()

    st.subheader("1. Experiment Configuration")
    st.info(
        "💡 **What is Sensitivity Analysis?** "
        "It tests how vulnerable your decision ranking is to changes in weights, parameters, or mathematical assumptions. "
        "Use parameter sweeps to track exact score trajectories, or generate Tornado charts to see which factors exert the most leverage over the winning alternative."
    )

    col_dim, col_methods = st.columns([1, 1])
    with col_dim:
        dim_label_1 = "1. Criteria Weights (Sweep & Tornado)" if is_flat_mode else "1. Category Weights (Sweep & Tornado)"
        dimension = st.selectbox(
            "Sensitivity Dimension:",
            [
                dim_label_1,
                "2. Fuzzy Component Weights",
                "3. Fuzzy PROMETHEE (q / p)",
                "4. WASPAS Lambda (λ)",
                "5. Trapezoid Multipliers (Kv / Ke)",
                "6. Bias Coefficient (Kb)",
                "⚡ ALL Dimensions (Full Diagnostic Suite)"
            ]
        )

    # --- INFORMATIONAL DESCRIPTIONS FOR EACH SENSITIVITY DIMENSION ---
    if dimension.startswith("1."):
        if is_flat_mode:
            st.markdown(
                "> ℹ️ **Measurement Guide (Criteria Weights Sweep & Tornado):** "
                "Systematically sweeps the weight of an individual criterion (Single Flat Mode) or generates a Tornado chart "
                "to measure the total score swing ($\Delta$) caused by ± perturbation. "
                "**Measures:** Factor leverage and rank stability against weight bias on specific criteria."
            )
        else:
            st.markdown(
                "> ℹ️ **Measurement Guide (Category Weights Sweep & Tornado):** "
                "Systematically sweeps a high-level category's priority weight or generates a Tornado chart across all domains. "
                "**Measures:** Which category's importance shift can cause a decision flip or alter the winner's margin."
            )
    elif dimension.startswith("2."):
        st.markdown(
            "> ℹ️ **Measurement Guide (Fuzzy Component Weights):** "
            "Tests decision models across different defuzzification weighting schemes (e.g., Centroid GMIR, Optimistic, Pessimistic) "
            "that dictate how trapezoidal bounds are converted into crisp evaluation scores. "
            "**Measures:** Sensitivity to defuzzification voting power distribution."
        )
    elif dimension.startswith("3."):
        st.markdown(
            "> ℹ️ **Measurement Guide (Fuzzy PROMETHEE q / p):** "
            "Performs a 2D parameter sweep across indifference threshold ($q$) and strict preference threshold ($p$). "
            "**Measures:** Outranking stability, noise filtering resilience, and winner victory margin space."
        )
    elif dimension.startswith("4."):
        st.markdown(
            "> ℹ️ **Measurement Guide (WASPAS Lambda λ):** "
            "Sweeps the hybrid tuning parameter $\lambda$ from 0.0 (pure Weighted Product Model) to 1.0 (pure Weighted Sum Model). "
            "**Measures:** Whether multiplicative compounding vs. additive utility changes your optimal alternative."
        )
    elif dimension.startswith("5."):
        st.markdown(
            "> ℹ️ **Measurement Guide (Trapezoid Multipliers Kv / Ke):** "
            "Scales volatility ($Kv$) and epistemic uncertainty ($Ke$) multipliers to test how widening or narrowing fuzzy bounds affects performance. "
            "**Measures:** Decision resilience under volatile or low-confidence rating environments."
        )
    elif dimension.startswith("6."):
        st.markdown(
            "> ℹ️ **Measurement Guide (Bias Coefficient Kb):** "
            "Varies the directional bias shift multiplier ($Kb$) which shapes optimistic or pessimistic bound tilting. "
            "**Measures:** Impact of Bias coefficient value on boundary skew on final rankings."
        )
    else:
        st.markdown(
            "> ℹ️ **Measurement Guide (Full Diagnostic Suite):** "
            "Automatically executes all sensitivity dimensions and tornado leverage analyses in batch. "
            "**Measures:** Comprehensive stress-testing report across all mathematical assumptions."
        )

    st.markdown("#### Parameter Values Setup")
    
    if dimension.startswith("1."):
        mode_label = "Criterion Analysis Mode:" if is_flat_mode else "Category Analysis Mode:"
        sweep_label = "Single Criterion Parameter Sweep" if is_flat_mode else "Single Category Parameter Sweep"
        tornado_label = "All-Criteria Tornado Leverage Chart" if is_flat_mode else "All-Categories Tornado Leverage Chart"

        cat_mode = st.radio(mode_label, [sweep_label, tornado_label], horizontal=True)
        if cat_mode == sweep_label:
            d1, d2 = st.columns([1, 1])
            with d1:
                if is_flat_mode:
                    item_dict = {f"{f['name']} ({f['id']})": f["id"] for f in ctx["factors"]}
                    target_label = "Target Criterion:"
                else:
                    item_dict = {d["name"]: d["id"] for d in ctx["domains"]}
                    target_label = "Target Category:"

                target_dom_name = st.selectbox(target_label, list(item_dict.keys()))
                target_dom_id = item_dict[target_dom_name]
                
                if is_flat_mode:
                    base_w = ctx["global_weights"].get(target_dom_id, 0.0)
                else:
                    base_w = ctx["category_weights"].get(target_dom_id, 0.0)
                st.info(f"Baseline Weight: **{base_w * 100:.1f}%**")
            with d2:
                sweep_mode = st.radio("Sweep Range:", ["Standard (0% to 50% in steps of 5%)", "Dense (0% to 100% in steps of 5%)", "Custom Range (Start, End, Step)", "Custom List"])
        
            if sweep_mode.startswith("Standard"):
                test_weights = [round(x, 2) for x in np.linspace(0.0, 0.50, 11).tolist()]
            elif sweep_mode.startswith("Dense"):
                test_weights = [round(x, 2) for x in np.linspace(0.0, 1.0, 21).tolist()]
            elif sweep_mode.startswith("Custom Range"):
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    start_val = st.number_input("Start (e.g. 0.0)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
                with rc2:
                    end_val = st.number_input("End (e.g. 0.5)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
                with rc3:
                    step_val = st.number_input("Step Size (e.g. 0.02)", min_value=0.001, max_value=0.5, value=0.02, step=0.01)
                
                if step_val > 0 and end_val >= start_val:
                    raw_weights = np.arange(start_val, end_val + (step_val / 2.0), step_val)
                    test_weights = [round(float(x), 4) for x in raw_weights if x <= 1.0]
                else:
                    test_weights = [0.0, 0.5]
                st.caption(f"Generated Weights Preview: `{test_weights}`")
            else:
                custom_str = st.text_input("Enter weights separated by commas:", "0.05, 0.10, 0.15, 0.20, 0.25, 0.30")
                test_weights = [float(x.strip()) for x in custom_str.split(",") if x.strip()]

            with col_methods:
                available_methods = [m for m in executed_methods if m in METHOD_REGISTRY]
                selected_methods = st.multiselect("Methods to Evaluate:", available_methods, default=available_methods)

            btn_label = "🚀 Run Criterion Sweep" if is_flat_mode else "🚀 Run Category Sweep"
            if st.button(btn_label, type="primary"):
                with st.spinner("Running in-memory sensitivity sweep..."):
                    if is_flat_mode:
                        sens_res = SensitivityEngine.analyze_criteria_weights(
                            baseline_run_id=selected_run_id, target_criterion_id=target_dom_id,
                            test_weights=test_weights, methods_to_run=selected_methods
                        )
                    else:
                        sens_res = SensitivityEngine.analyze_category_weights(
                            baseline_run_id=selected_run_id, target_domain_id=target_dom_id,
                            test_weights=test_weights, methods_to_run=selected_methods
                        )
                    st.session_state["last_sensitivity_result"] = {"mode": "single_sweep", "data": sens_res}
                    st.success("Sensitivity sweep complete!")

        else:
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                available_methods = [m for m in executed_methods if m in METHOD_REGISTRY]
                tornado_method = st.selectbox("Model for Tornado Analysis:", available_methods, key="t_meth_cat")
            with t_col2:
                tornado_pct = st.slider("Perturbation Range (±%):", min_value=10, max_value=80, value=50, step=5) / 100.0

            with col_methods:
                st.info("🌪️ Systematically perturbs all factors to evaluate total score leverage on the baseline winner.")

            tornado_btn_label = "🌪️ Generate Criteria Tornado Chart" if is_flat_mode else "🌪️ Generate Category Tornado Chart"
            if st.button(tornado_btn_label, type="primary"):
                if tornado_method is None:
                    st.warning("Select a model before generating the tornado chart.")
                else:
                    with st.spinner("Calculating leverage across all factors..."):
                        if is_flat_mode:
                            t_data = SensitivityEngine.analyze_criteria_tornado(
                                baseline_run_id=selected_run_id, method_name=tornado_method,
                                perturbation_fraction=tornado_pct
                            )
                        else:
                            t_data = SensitivityEngine.analyze_category_tornado(
                                baseline_run_id=selected_run_id, method_name=tornado_method,
                                perturbation_fraction=tornado_pct
                            )
                        st.session_state["last_sensitivity_result"] = {"mode": "tornado", "data": t_data}
                        st.success("Tornado leverage analysis complete!")

    elif dimension.startswith("2."):
        st.write("Evaluate fuzzy method net flows across different trapezoid component (a, b, c, d) weight allocations:")
        selected_schemes = {s_name: s_weights for s_name, s_weights in DEFUZZ_WEIGHT_SCHEMES.items() if st.checkbox(s_name, value=True)}
        with col_methods:
            fuzzy_methods = [m for m in executed_methods if "fuzzy" in m.lower() or m == "Fuzzy PROMETHEE"]
            selected_methods = st.multiselect("Fuzzy Methods:", fuzzy_methods or ["Fuzzy PROMETHEE"], default=fuzzy_methods or ["Fuzzy PROMETHEE"])

        if st.button("🚀 Run Fuzzy Component Sensitivity", type="primary"):
            with st.spinner("Running sweep across defuzzification schemes..."):
                sens_res = SensitivityEngine.analyze_fuzzy_component_weights(
                    baseline_run_id=selected_run_id, schemes=selected_schemes, methods_to_run=selected_methods
                )
                st.session_state["last_sensitivity_result"] = {"mode": "single_sweep", "data": sens_res}
                st.success("Fuzzy component sensitivity complete!")

    elif dimension.startswith("3."):
        qp1, qp2 = st.columns(2)
        with qp1:
            q_vals = st.multiselect(
                "Indifference Thresholds (q):", 
                [0.0, 0.2, 0.5, 0.8, 1.0, 1.5], 
                default=[0.2, 0.5, 1.0],
                key="sens_qp_q_vals"  # Explicit key prevents state caching conflicts
            )
        with qp2:
            p_vals = st.multiselect(
                "Strict Preference Thresholds (p):", 
                [1.0, 2.0, 3.0, 3.5, 4.0, 5.0], 
                default=[2.0, 3.5, 5.0],
                key="sens_qp_p_vals"  # Explicit key forces fresh defaults
            )

        qp_pairs = [(q, p) for q in q_vals for p in p_vals if p > q]
        st.caption(f"Valid (q, p) coordinate pairs: **{len(qp_pairs)} combinations**")
        with col_methods:
            st.info("🎯 Scope strictly isolated to: **Fuzzy PROMETHEE**")

        if st.button("🚀 Run PROMETHEE (q, p) Sensitivity", type="primary"):
            with st.spinner("Running PROMETHEE 2D parameter sweep..."):
                sens_res = SensitivityEngine.analyze_promethee_qp(baseline_run_id=selected_run_id, qp_pairs=qp_pairs)
                st.session_state["last_sensitivity_result"] = {"mode": "single_sweep", "data": sens_res}
                st.success("PROMETHEE parameter sweep complete!")

    elif dimension.startswith("4."):
        lmbd_choices = st.multiselect("WASPAS Lambda (λ) Test Values:", [0.0, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0], default=[0.0, 0.25, 0.5, 0.75, 1.0])
        with col_methods:
            st.info("🎯 Scope strictly isolated to: **WASPAS** (0.0 = WPM, 1.0 = WSM)")

        if st.button("🚀 Run WASPAS Lambda Sensitivity", type="primary"):
            with st.spinner("Running WASPAS lambda sweep..."):
                sens_res = SensitivityEngine.analyze_waspas_lambda(baseline_run_id=selected_run_id, lambda_values=lmbd_choices)
                st.session_state["last_sensitivity_result"] = {"mode": "single_sweep", "data": sens_res}
                st.success("WASPAS sweep complete!")

    elif dimension.startswith("5."):
        grid_pairs = [(0.5, 0.5), (1.0, 0.5), (0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (1.0, 1.5)]
        selected_pairs = st.multiselect(
            "Select (Kv, Ke) Coordinate Multipliers:", grid_pairs, default=[(0.5, 0.5), (1.0, 0.5), (0.5, 1.0), (1.0, 1.0)],
            format_func=lambda x: f"Kv = {x[0]:.1f} (Volatility), Ke = {x[1]:.1f} (Uncertainty)"
        )
        with col_methods:
            fuzzy_methods = [m for m in executed_methods if "fuzzy" in m.lower() or m == "Fuzzy PROMETHEE"]
            selected_methods = st.multiselect("Fuzzy Methods:", fuzzy_methods or ["Fuzzy PROMETHEE"], default=fuzzy_methods or ["Fuzzy PROMETHEE"])

        if st.button("🚀 Run (Kv, Ke) Multiplier Sensitivity", type="primary"):
            with st.spinner("Regenerating trapezoids in-memory and running fuzzy methods..."):
                sens_res = SensitivityEngine.analyze_kv_ke(baseline_run_id=selected_run_id, kv_ke_pairs=selected_pairs, methods_to_run=selected_methods)
                st.session_state["last_sensitivity_result"] = {"mode": "single_sweep", "data": sens_res}
                st.success("Kv/Ke sensitivity complete!")

    elif dimension.startswith("6."):
        kb_choices = st.multiselect("Bias Multiplier (Kb) Test Values:", [0.5, 1.0, 1.5, 2.0, 2.5], default=[0.5, 1.0, 1.5, 2.0])
        with col_methods:
            fuzzy_methods = [m for m in executed_methods if "fuzzy" in m.lower() or m == "Fuzzy PROMETHEE"]
            selected_methods = st.multiselect("Fuzzy Methods:", fuzzy_methods or ["Fuzzy PROMETHEE"], default=fuzzy_methods or ["Fuzzy PROMETHEE"])

        if st.button("🚀 Run Bias Multiplier Sensitivity", type="primary"):
            with st.spinner("Regenerating biased trapezoids in-memory and running fuzzy methods..."):
                sens_res = SensitivityEngine.analyze_bias_coefficient(baseline_run_id=selected_run_id, kb_values=kb_choices, methods_to_run=selected_methods)
                st.session_state["last_sensitivity_result"] = {"mode": "single_sweep", "data": sens_res}
                st.success("Bias multiplier sensitivity complete!")

    else:
        st.info("⚡ This will execute all 6 sensitivity dimensions plus Tornado factor leverage in a unified batch.")
        with col_methods:
            available_methods = [m for m in executed_methods if m in METHOD_REGISTRY]
            selected_methods = st.multiselect("Models to Include:", available_methods, default=available_methods)

        if st.button("⚡ Run Full Sensitivity Diagnostic Suite", type="primary"):
            with st.spinner("Executing full sensitivity diagnostic suite..."):
                suite_bundle = {"metadata": {"baseline_run_id": selected_run_id, "dimension": "full_suite", "baseline_run_name": baseline_run.get("name", "")}, "results": {}}
                if is_flat_mode:
                    suite_bundle["results"]["tornado"] = SensitivityEngine.analyze_criteria_tornado(selected_run_id, selected_methods[0] if selected_methods else "TOPSIS")
                else:
                    suite_bundle["results"]["tornado"] = SensitivityEngine.analyze_category_tornado(selected_run_id, selected_methods[0] if selected_methods else "TOPSIS")
                suite_bundle["results"]["fuzzy_components"] = SensitivityEngine.analyze_fuzzy_component_weights(selected_run_id)
                if "WASPAS" in executed_methods:
                    suite_bundle["results"]["waspas"] = SensitivityEngine.analyze_waspas_lambda(selected_run_id, [0.0, 0.25, 0.5, 0.75, 1.0])
                if "Fuzzy PROMETHEE" in executed_methods:
                    suite_bundle["results"]["promethee_qp"] = SensitivityEngine.analyze_promethee_qp(selected_run_id, [(0.2, 2.0), (0.5, 3.5), (1.0, 4.0)])
                suite_bundle["results"]["kv_ke"] = SensitivityEngine.analyze_kv_ke(selected_run_id, [(0.5, 0.5), (1.0, 0.5), (0.5, 1.0), (1.0, 1.0)])
                suite_bundle["results"]["bias_kb"] = SensitivityEngine.analyze_bias_coefficient(selected_run_id, [0.5, 1.0, 1.5, 2.0])

                st.session_state["last_sensitivity_result"] = {"mode": "full_suite", "data": suite_bundle}
                st.success("Full diagnostic suite execution complete!")

    # DISPLAY SENSITIVITY RESULTS
    if "last_sensitivity_result" in st.session_state:
        res_container = st.session_state["last_sensitivity_result"]
        res_mode = res_container.get("mode", "single_sweep")
        res_data = res_container.get("data", {})

        st.markdown("---")
        st.subheader("2. Sensitivity Visualizations & Results")

        if res_mode == "tornado" or ("tornado_entries" in res_data):
            st.plotly_chart(plot_category_tornado(res_data, is_flat_mode), use_container_width=True)
            st.info(
                "📊 **Graph Guide — Tornado Leverage Chart:**\n"
                "* **X-Axis:** Score Shift ($\Delta$ Score) from baseline for the winning alternative.\n"
                "* **Y-Axis:** Evaluated categories or criteria.\n"
                "* **How to Interpret:** Bars extending to the right (green) show score gains when that factor's weight increases. Bars extending to the left (red) show score losses when weight decreases. The wider the total span, the higher that factor's leverage over the decision."
            )
            with st.expander("📋 View Factor Leverage Table", expanded=True):
                t_df = pd.DataFrame([{
                    "Factor": e.get("domain_name", e.get("criterion_name", "Unknown")), 
                    "Baseline Weight": f"{e['baseline_weight']*100:.1f}%",
                    "Test Range": f"{e['w_low']*100:.1f}% → {e['w_high']*100:.1f}%",
                    "Score Swing (Δ)": f"{e['total_swing']:.5f}",
                    "Impact when Decreased": f"{e['delta_low']:+.5f}", "Impact when Increased": f"{e['delta_high']:+.5f}"
                } for e in res_data.get("tornado_entries", [])])
                st.dataframe(t_df, use_container_width=True)

        elif res_mode == "full_suite":
            suite = res_data.get("results", {})
            if "tornado" in suite:
                with st.expander("🌪️ Factor Leverage Tornado Chart", expanded=True):
                    st.plotly_chart(plot_category_tornado(suite["tornado"], is_flat_mode), use_container_width=True)
            if "waspas" in suite:
                with st.expander("📐 WASPAS Lambda (λ) Trajectory"):
                    st.plotly_chart(plot_sensitivity_curves(suite["waspas"], "WASPAS", countries), use_container_width=True)
                    st.info("📊 **Graph Guide — WASPAS Lambda Trajectory:** X-axis represents $\lambda$ (0.0 to 1.0). Y-axis represents synthesized scores. Shows how blending multiplicative and additive utility alters alternative rankings.")
            if "promethee_qp" in suite:
                with st.expander("🎯 Fuzzy PROMETHEE (q, p) Decision Space"):
                    st.plotly_chart(plot_qp_heatmap(suite["promethee_qp"]), use_container_width=True)
                    st.info("📊 **Graph Guide — PROMETHEE (q, p) Heatmap:** X-axis represents indifference threshold $q$. Y-axis represents strict preference threshold $p$. Color shading and text indicate victory margins and winner stability across parameter space.")
            if "fuzzy_components" in suite:
                with st.expander("📊 Trapezoid Defuzzification Schemes"):
                    st.plotly_chart(plot_scheme_bar_chart(suite["fuzzy_components"], "Fuzzy PROMETHEE", countries), use_container_width=True)
                    st.info("📊 **Graph Guide — Defuzzification Schemes:** Compares how different centroid voting weights (GMIR, optimistic, pessimistic) affect outranking net flows.")
            if "kv_ke" in suite:
                with st.expander("🔬 Volatility / Uncertainty Multipliers (Kv, Ke)"):
                    st.plotly_chart(plot_sensitivity_curves(suite["kv_ke"], "Fuzzy PROMETHEE", countries), use_container_width=True)
            if "bias_kb" in suite:
                with st.expander("⚖️ Directional Bias Multiplier (Kb)"):
                    st.plotly_chart(plot_sensitivity_curves(suite["bias_kb"], "Fuzzy PROMETHEE", countries), use_container_width=True)

        else:
            meta = res_data.get("metadata", {})
            iterations = res_data.get("iterations", [])
            methods_tested = meta.get("methods_evaluated", [])
            dim = meta.get("dimension", "")

            if dim == "fuzzy_promethee_qp":
                st.plotly_chart(plot_qp_heatmap(res_data), use_container_width=True)
                st.info("📊 **Graph Guide — PROMETHEE (q, p) Decision Space:** Shows victory margins and winner stability across $(q, p)$ threshold combinations.")

            for idx, m_name in enumerate(methods_tested):
                with st.expander(f"📊 Model: **{m_name}**", expanded=(idx == 0)):
                    if dim == "fuzzy_component_weights":
                        st.plotly_chart(plot_scheme_bar_chart(res_data, m_name, countries), use_container_width=True)
                        st.info("📊 **Graph Guide — Defuzzification Scheme Bar Chart:** Compares performance bars for each alternative across different centroid weighting schemes.")
                    elif dim != "fuzzy_promethee_qp":
                        st.plotly_chart(plot_sensitivity_curves(res_data, m_name, countries), use_container_width=True)
                        st.info(
                            "📊 **Graph Guide — Score Sensitivity Trajectory:**\n"
                            "* **X-Axis:** Parameter value or weight fraction being swept.\n"
                            "* **Y-Axis:** Resulting score or net preference flow.\n"
                            "* **How to Interpret:** Each colored line tracks an alternative's score trajectory. Vertical dashed lines show your baseline setting. Crossing lines indicate rank flips."
                        )

                    table_rows = []
                    for it in iterations:
                        p_val = it["param_value"]
                        m_res = it.get("method_results", {}).get(m_name, {})
                        if m_res.get("status") == "success":
                            row = {
                                "Parameter Value": p_val if not isinstance(p_val, float) else f"{p_val*100:.1f}%" if "weight" in dim else f"{p_val:.3f}",
                                "Winner": m_res.get("winner", "-"), "Winner Changed?": "🔴 YES" if m_res.get("winner_changed") else "🟢 No"
                            }
                            for c in countries:
                                score = m_res["scores"].get(c, 0.0)
                                delta = m_res["score_deltas"].get(c, 0.0)
                                rank = m_res["ranking"].get(c, 0)
                                row[f"{c} (Score / Δ)"] = f"{score:.4f} ({delta:+.4f}) [R{rank}]"
                            table_rows.append(row)
                    if table_rows:
                        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            save_name_input = st.text_input("Name this Experiment:", value=f"Sensitivity_{res_mode}_{selected_run_id[:8]}", key="sens_save_name")
        with col_s2:
            st.write("")
            st.write("")
            if st.button("💾 Save Experiment", key="save_sens_btn", type="secondary"):
                exp_id = AnalysisDispatcher.save_analysis_experiment(res_data, save_name_input)
                st.success(f"Saved as '{save_name_input}' (ID: {exp_id[:12]})")

        st.download_button(
            label="⬇️ Download Analysis JSON", data=json.dumps(res_data, indent=4),
            file_name=f"sensitivity_{res_mode}_{selected_run_id[:8]}.json", mime="application/json", key="dl_sens_btn"
        )

# -------------------------------------------------------------------------
# TAB 2: ROBUSTNESS & DECISION BOUNDARIES
# -------------------------------------------------------------------------
with tab_rob:
    st.subheader("1. Robustness Evaluation on Active Experiment")
    st.info(
        "🛡️ **Understanding Robustness & Stability:** "
        "Robustness measures how well your decision holds up under perturbation. "
        "The **Safe Stability Range** identifies the exact weight percentages where the winner remains secure before a rank flip occurs."
    )
    
    if "last_sensitivity_result" not in st.session_state:
        st.info("ℹ️ Run or load a sensitivity experiment in Tab 1 to automatically view its robustness diagnostics here.")
    else:
        res_container = st.session_state["last_sensitivity_result"]
        res_data = res_container.get("data", {})
        
        if "iterations" in res_data:
            rob_data = RobustnessEngine.evaluate_sensitivity_robustness(res_data)
            summary = rob_data.get("overall_summary", {})
            c_r1, c_r2, c_r3 = st.columns(3)
            with c_r1:
                st.metric("Consensus Winner Stable?", "✅ YES" if summary.get("is_strictly_stable_all_methods") else "⚠️ NO (Flips Occurred)")
            with c_r2:
                st.metric("Mean Winner Stability", f"{summary.get('mean_stability_pct', 0.0):.1f}%")
            with c_r3:
                st.metric("Iterations Evaluated", summary.get("total_iterations_tested", 0))

            st.markdown("#### Method Stability & Margin Diagnostics")
            rob_table = []
            for m_name, m_info in rob_data.get("method_details", {}).items():
                margins = m_info.get("margin_stats", {})
                rob_table.append({
                    "Method": m_name, "Baseline Winner": m_info.get("baseline_winner"),
                    "Strictly Stable?": "✅ Yes" if m_info.get("is_winner_strictly_stable") else "🔴 Flips",
                    "Winner Stability %": f"{m_info.get('winner_stability_pct', 0.0):.1f}%",
                    "Flip Count": m_info.get("flip_count", 0),
                    "Min Margin (1st - 2nd)": f"{margins.get('min_margin', 0.0):.4f}",
                    "Avg Margin": f"{margins.get('avg_margin', 0.0):.4f}", "Max Margin": f"{margins.get('max_margin', 0.0):.4f}"
                })
            if rob_table:
                st.dataframe(pd.DataFrame(rob_table), use_container_width=True)
        else:
            st.info("ℹ️ Run a single-parameter sensitivity sweep in Tab 1 to generate detailed robustness & margin tables.")

    st.markdown("---")
    boundary_label = "2. Critical Criterion Weight Boundary Scanner" if is_flat_mode else "2. Critical Category Weight Boundary Scanner"
    st.subheader(boundary_label)
    st.caption("Performs a high-resolution scan across 0% to 100% weight to find exact thresholds where the decision flips.")

    b1, b2, b3 = st.columns(3)
    with b1:
        if is_flat_mode:
            item_dict = {f"{f['name']} ({f['id']})": f["id"] for f in ctx["factors"]}
            scan_label = "Criterion to Scan:"
        else:
            item_dict = {d["name"]: d["id"] for d in ctx["domains"]}
            scan_label = "Category to Scan:"

        scan_dom_name = st.selectbox(scan_label, list(item_dict.keys()), key="scan_dom")
        scan_dom_id = item_dict[scan_dom_name]
    with b2:
        scan_method = st.selectbox("MCDM Method to Scan:", executed_methods, key="scan_meth")
    with b3:
        scan_res = st.slider("Scan Resolution (Steps):", min_value=21, max_value=101, value=51, step=10)

    if st.button("🎯 Scan Critical Decision Boundary", type="primary"):
        with st.spinner(f"Scanning decision boundaries for '{scan_dom_name}' under {scan_method}..."):
            if is_flat_mode:
                thresh = RobustnessEngine.find_criterion_weight_threshold(
                    baseline_run_id=selected_run_id, target_criterion_id=scan_dom_id, method_name=scan_method, resolution=scan_res
                )
            else:
                thresh = RobustnessEngine.find_category_weight_threshold(
                    baseline_run_id=selected_run_id, target_domain_id=scan_dom_id, method_name=scan_method, resolution=scan_res
                )
            st.plotly_chart(plot_boundary_range(thresh, countries, is_flat_mode), use_container_width=True)
            st.info(
                "📊 **Graph Guide — Decision Stability Zones:**\n"
                "* **X-Axis:** Weight percentage (0% to 100%) assigned to the factor.\n"
                "* **Y-Axis:** Target factor name.\n"
                "* **How to Interpret:** Colored bands show which alternative wins at each weight level. The diamond marker shows your current baseline weight. The dashed line marks the exact 'Flip Threshold' where the winner changes."
            )
            st.markdown("#### Boundary Scan Summary")
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Baseline Factor Weight", f"{thresh['baseline_weight']*100:.1f}%")
            with k2:
                st.metric("Safe Stability Range", f"{thresh['safe_stability_range'][0]*100:.1f}% → {thresh['safe_stability_range'][1]*100:.1f}%")
            with k3:
                st.metric("Strictly Stable (0–100%)?", "✅ YES" if thresh['is_strictly_stable_across_entire_range'] else "⚠️ Switch Point Found")

# -------------------------------------------------------------------------
# TAB 3: EPISTEMIC UNCERTAINTY PROPAGATION
# -------------------------------------------------------------------------
with tab_epistemic:
    saved_ep_exps = AnalysisDispatcher.list_saved_analysis_experiments(filter_type="epistemic")
    if saved_ep_exps:
        with st.expander("📂 Load a Saved Epistemic Experiment", expanded=False):
            ep_col1, ep_col2, ep_col3 = st.columns([3, 1, 1])
            with ep_col1:
                ep_map = {f"{e['saved_name']} ({e['saved_timestamp'][:16].replace('T', ' ')})": e["analysis_id"] for e in saved_ep_exps}
                sel_ep_lbl = st.selectbox("Select Epistemic Experiment:", list(ep_map.keys()), key="load_ep_sel")
                sel_ep_id = ep_map[sel_ep_lbl]
            with ep_col2:
                st.write("")
                st.write("")
                if st.button("📖 Load", key="load_ep_btn", type="primary"):
                    st.session_state["last_epistemic_result"] = AnalysisDispatcher.load_saved_analysis_experiment(sel_ep_id)
                    st.success("Loaded successfully!")
                    st.rerun()
            with ep_col3:
                st.write("")
                st.write("")
                if st.button("🗑️ Delete", key="del_ep_btn"):
                    AnalysisDispatcher.delete_saved_analysis_experiment(sel_ep_id)
                    st.success("Deleted!")
                    st.rerun()

    st.subheader("🧬 Epistemic Uncertainty (E) Realization in Deterministic Models")
    st.info(
        "🧬 **Understanding Epistemic Uncertainty Propagation:** "
        "Fuzzy models capture confidence through epistemic uncertainty intervals $[r - Ke \\cdot E, r + Ke \\cdot E]$. "
        "This module propagates those bounds into deterministic central ratings using One-At-A-Time (OAT) and scenario-based perturbations "
        "to reveal which specific criteria carry enough uncertainty to flip the final decision."
    )

    det_methods = [m for m in executed_methods if m in ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR"]]
    if not det_methods:
        det_methods = ["WASPAS", "WSM", "TOPSIS"]

    e_col1, e_col2, e_col3 = st.columns(3)
    with e_col1:
        model_selection_options = ["⚡ ALL Deterministic Models"] + det_methods
        ep_models_choice = st.selectbox("Deterministic MCDM Model(s):", model_selection_options, key="ep_meth")
    with e_col2:
        ep_level = st.selectbox("Analysis Level:", ["ALL Levels", "1. Individual Criteria (OAT)", "2. Category Representatives", "3. Combined Scenarios"], key="ep_lvl")
    with e_col3:
        rep_strategy = st.selectbox(
            "Representative Selection Strategy:",
            ["Highest Weight × E", "Highest Global Weight", "Median Global Weight"],
            index=0, help="Criteria selection rule used for representatives and combined scenarios."
        )

    strat_code_map = {"Highest Weight × E": "highest_weight_x_e", "Highest Global Weight": "highest_weight", "Median Global Weight": "median_weight"}
    selected_strat_code = strat_code_map[rep_strategy]

    if st.button("🚀 Run Epistemic Uncertainty Propagation", type="primary"):
        target_models = det_methods if ep_models_choice.startswith("⚡") else [ep_models_choice]
        
        with st.spinner(f"Evaluating discrete epistemic realizations across {len(target_models)} model(s)..."):
            epistemic_bundle = {
                "metadata": {
                    "baseline_run_id": selected_run_id, "baseline_run_name": baseline_run.get("name", ""),
                    "models_evaluated": target_models, "analysis_type": "epistemic_propagation",
                    "dimension": "epistemic_propagation", "countries": countries, "representative_strategy": rep_strategy
                },
                "models": {}
            }

            shared_reps = None
            if ep_level in ["ALL Levels", "2. Category Representatives", "3. Combined Scenarios"]:
                shared_reps = EpistemicEngine.select_category_representatives(strategy=selected_strat_code)

            for m_name in target_models:
                m_bundle = {}
                if ep_level in ["ALL Levels", "1. Individual Criteria (OAT)"]:
                    m_bundle["level1"] = EpistemicEngine.analyze_individual_criteria(baseline_run_id=selected_run_id, method_name=m_name)
                if shared_reps is not None:
                    m_bundle["level2_representatives"] = shared_reps
                if ep_level in ["ALL Levels", "3. Combined Scenarios"]:
                    if shared_reps is None:
                        raise RuntimeError("Representatives are required for combined scenarios.")
                    m_bundle["level3"] = EpistemicEngine.analyze_combined_scenarios(baseline_run_id=selected_run_id, method_name=m_name, representatives=shared_reps)
                epistemic_bundle["models"][m_name] = m_bundle

            st.session_state["last_epistemic_result"] = epistemic_bundle
            st.success(f"Epistemic propagation completed for {len(target_models)} model(s)!")

    # DISPLAY EPISTEMIC RESULTS
    if "last_epistemic_result" in st.session_state:
        ep_res = st.session_state["last_epistemic_result"]
        models_dict = {}
        if "models" in ep_res:
            models_dict = ep_res["models"]
        elif "level1" in ep_res or "level3" in ep_res:
            single_m = ep_res.get("metadata", {}).get("method_name", "Selected Model")
            models_dict = {single_m: ep_res}

        if models_dict:
            st.markdown("---")
            st.subheader("2. Epistemic Propagation Visualizations & Results")

            for idx, (m_name, m_data) in enumerate(models_dict.items()):
                with st.expander(f"📊 Model: **{m_name}**", expanded=(idx == 0)):
                    if "level1" in m_data:
                        l1 = m_data["level1"]
                        crit_list = l1.get("criteria_results", [])
                        meta1 = l1.get("metadata", {})
                        
                        st.markdown(f"#### 1. Level 1: Individual Criteria Epistemic Leverage (OAT)")
                        st.caption(f"Tested **{len(crit_list)}** criteria. Baseline Winner: **{meta1.get('baseline_winner', '')}** (Advantage: **{meta1.get('baseline_advantage', 0.0):+.4f}**).")

                        flip_count = sum(1 for c in crit_list if c.get("winner_flipped"))
                        mat_count = sum(1 for c in crit_list if c.get("classification") == "Material Shift")
                        neg_count = sum(1 for c in crit_list if c.get("classification") == "Negligible")

                        em1, em2, em3, em4 = st.columns(4)
                        with em1: st.metric("Total Uncertain Criteria", len(crit_list))
                        with em2: st.metric("🔴 Winner Flip Criteria", flip_count)
                        with em3: st.metric("🟠 Material Shift Criteria", mat_count)
                        with em4: st.metric("⚪ Negligible Shift", neg_count)

                        st.plotly_chart(plot_epistemic_leverage_chart(crit_list, meta1.get("countries", countries), m_name), use_container_width=True)
                        st.info(
                            "📊 **Graph Guide — Ranked Epistemic Leverage:**\n"
                            "* **X-Axis:** Maximum absolute shift in decision advantage ($|\\Delta\\text{Advantage}|$).\n"
                            "* **Y-Axis:** Evaluated criteria.\n"
                            "* **How to Interpret:** Measures how much pushing a criterion to its uncertainty bounds shifts the score gap between top alternatives. Red bars indicate criteria whose uncertainty is wide enough to cause a complete winner flip."
                        )

                        st.markdown("##### 🔍 Single Criterion Decision Trajectory")
                        c_options = {f"[{c['classification']}] {c['criterion_name']} ({c['category_name']})": c for c in crit_list}
                        if c_options:
                            sel_c_label = st.selectbox("Select Criterion to View Trajectory:", list(c_options.keys()), key=f"ep_crit_{m_name}_{idx}")
                            chosen_crit = c_options[sel_c_label]
                            st.plotly_chart(plot_epistemic_trajectory(chosen_crit, meta1.get("countries", countries)), use_container_width=True)
                            st.info(
                                "📊 **Graph Guide — Epistemic Trajectory:**\n"
                                "* **X-Axis:** Realized point rating across the epistemic interval $[r - Ke \\cdot E, r + Ke \\cdot E]$.\n"
                                "* **Y-Axis:** Decision Advantage (Positive favors top alternative, negative favors runner-up).\n"
                                "* **How to Interpret:** Tracks how the score advantage crosses the red 'Flip Boundary' ($=0$) as the criterion rating varies within its confidence bounds."
                            )

                        with st.expander(f"📋 View Complete Level 1 Criteria Results Table ({m_name})"):
                            tbl_rows = []
                            for c in crit_list:
                                e_str = " | ".join([f"{k}: r={v['r']:.1f}, E={v['E']}" for k, v in c.get("evaluations", {}).items()])
                                tbl_rows.append({
                                    "Criterion": c.get("criterion_name"), "Category": c.get("category_name"),
                                    "Global %": f"{c.get('global_weight', 0.0)*100:.2f}%", "Ratings & E": e_str,
                                    "Advantage Range": f"[{c.get('min_advantage', 0.0):+.4f} → {c.get('max_advantage', 0.0):+.4f}]",
                                    "Max Shift (|Δ|)": f"{c.get('max_abs_advantage_shift', 0.0):.4f}",
                                    "Winner Flip?": "🔴 YES" if c.get("winner_flipped") else "🟢 No", "Classification": c.get("classification")
                                })
                            st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True)

                    if "level2_representatives" in m_data:
                        reps = m_data["level2_representatives"]
                        st.markdown(f"#### 2. Level 2: Selected Category Representatives")
                        rep_df = pd.DataFrame([{
                            "Category": r.get("domain_name"), "Representative Criterion": r.get("criterion_name"),
                            "Global Weight": f"{r.get('global_weight', 0.0)*100:.2f}%", "Max Epistemic E": r.get("max_uncertainty_E"), "Weight × E": f"{r.get('weight_x_e', 0.0):.4f}"
                        } for r in reps])
                        st.dataframe(rep_df, use_container_width=True)

                    if "level3" in m_data:
                        l3 = m_data["level3"]
                        scens = l3.get("scenarios", [])
                        st.markdown(f"#### 3. Level 3: Combined Multi-Category Epistemic Scenarios")
                        st.plotly_chart(plot_combined_scenarios_chart(scens, countries, m_name), use_container_width=True)
                        st.info(
                            "📊 **Graph Guide — Combined Scenarios:** Compares baseline advantage against worst-case, best-case, and combined compounding uncertainty scenarios across top criteria."
                        )

            col_ep_s1, col_ep_s2 = st.columns([2, 1])
            with col_ep_s1:
                default_name = f"Epistemic_{len(models_dict)}Models_{selected_run_id[:8]}"
                ep_save_name = st.text_input("Name this Epistemic Experiment:", value=default_name, key="ep_save_name_input")
            with col_ep_s2:
                st.write("")
                st.write("")
                if st.button("💾 Save Epistemic Run", key="save_ep_btn", type="secondary"):
                    exp_id = AnalysisDispatcher.save_analysis_experiment(ep_res, ep_save_name)
                    st.success(f"Saved as '{ep_save_name}' (ID: {exp_id[:12]})")

            st.download_button(
                label="⬇️ Download Epistemic Analysis JSON", data=json.dumps(ep_res, indent=4),
                file_name=f"epistemic_{len(models_dict)}models_{selected_run_id[:8]}.json", mime="application/json", key="dl_ep_btn"
            )

# -------------------------------------------------------------------------
# TAB 4: MONTE CARLO UNCERTAINTY SIMULATION
# -------------------------------------------------------------------------
with tab_mc:
    saved_mc_exps = AnalysisDispatcher.list_saved_analysis_experiments(filter_type="monte_carlo")
    if saved_mc_exps:
        with st.expander("📂 Load a Saved Monte Carlo Simulation", expanded=False):
            mc_col1, mc_col2, mc_col3 = st.columns([3, 1, 1])
            with mc_col1:
                mc_map = {f"{e['saved_name']} ({e['saved_timestamp'][:16].replace('T', ' ')})": e["analysis_id"] for e in saved_mc_exps}
                sel_mc_lbl = st.selectbox("Select Monte Carlo Experiment:", list(mc_map.keys()), key="load_mc_sel")
                sel_mc_id = mc_map[sel_mc_lbl]
            with mc_col2:
                st.write("")
                st.write("")
                if st.button("📖 Load", key="load_mc_btn", type="primary"):
                    st.session_state["last_monte_carlo_result"] = AnalysisDispatcher.load_saved_analysis_experiment(sel_mc_id)
                    st.success("Loaded successfully!")
                    st.rerun()
            with mc_col3:
                st.write("")
                st.write("")
                if st.button("🗑️ Delete", key="del_mc_btn"):
                    AnalysisDispatcher.delete_saved_analysis_experiment(sel_mc_id)
                    st.success("Deleted!")
                    st.rerun()

    st.subheader("🎲 Monte Carlo Uncertainty Simulation")
    st.info(
        "🎲 **Understanding Monte Carlo Simulations:** "
        "Instead of testing single worst-case scenarios, Monte Carlo simulation stochastically samples thousands of randomized ratings "
        "across each criterion's uncertainty bounds ($N$ iterations). "
        "This outputs precise **Probabilistic Win Rates** and score distribution bell curves, quantifying your overall confidence in the decision."
    )

    mc_c1, mc_c2, mc_c3 = st.columns(3)
    with mc_c1:
        det_methods = [m for m in executed_methods if m in ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR"]] or ["WASPAS", "WSM", "TOPSIS"]
        mc_model_choices = ["⚡ ALL Deterministic Models"] + det_methods
        mc_selected_models = st.selectbox("Model(s) to Simulate:", mc_model_choices, key="mc_model_sel")
    with mc_c2:
        mc_mode = st.selectbox(
            "Uncertainty Range Mode:",
            ["Full Uncertainty (V + E Combined)", "Epistemic Uncertainty Only (E)"],
            index=0,
            help="Full Uncertainty samples across r ± (V*Kv + E*Ke). Epistemic Only samples across r ± (E*Ke)."
        )
        mode_code = "full_uncertainty" if "Full" in mc_mode else "epistemic_only"
    with mc_c3:
        mc_iterations = st.select_slider("Simulation Iterations (N):", options=[1000, 5000, 10000, 20000, 50000], value=10000)

    mc_adv_c1, mc_adv_c2, mc_adv_c3 = st.columns(3)
    with mc_adv_c1:
        mc_dist = st.radio(
            "Probability Distribution:",
            ["Trapezoidal / Triangular (Fuzzy Geometry)", "Normal (Bell Curve)", "Uniform (Flat over interval)"],
            horizontal=True
        )
        if "Trapezoidal" in mc_dist:
            dist_code = "trapezoidal"
        elif "Normal" in mc_dist:
            dist_code = "normal"
        else:
            dist_code = "uniform"
            
    with mc_adv_c2:
        if dist_code == "normal":
            mc_coverage = st.slider(
                "Uncertainty Bounds Coverage (%):",
                min_value=60, max_value=99, value=75, step=5,
                help="75% (Recommended): Heavier tail swings where V and E actively pull the decision. 95%: Conservative bell curve centered tightly at r."
            )
        elif dist_code == "trapezoidal":
            st.info("📐 Uses exact fuzzy geometry: triangles when $E=0$ ($V>0$), rectangles when $V=0$ ($E>0$), and trapezoids when both are active. Natively preserves bias shifts and asymmetry!")
            mc_coverage = 75.0
        else:
            st.info("📏 Uniform distribution samples with equal probability across 100% of the interval.")
            mc_coverage = 100.0
            
    with mc_adv_c3:
        mc_quantize = st.radio("Grid Quantization:", ["Snap to 0.5 Intervals", "Continuous (No discretization)"], horizontal=True)
        step_val = 0.5 if "0.5" in mc_quantize else None

    if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
        target_mc_models = det_methods if mc_selected_models.startswith("⚡") else [mc_selected_models]
        
        with st.spinner(f"Running {mc_iterations:,} Monte Carlo simulations ({mc_coverage:.0f}% spread coverage) across {len(target_mc_models)} model(s)..."):
            mc_result = MonteCarloEngine.run_simulation(
                baseline_run_id=selected_run_id,
                method_names=target_mc_models,
                num_iterations=mc_iterations,
                mode=mode_code,
                distribution=dist_code,
                coverage_pct=float(mc_coverage),
                discrete_step=step_val
            )
            st.session_state["last_monte_carlo_result"] = mc_result
            st.success(f"Monte Carlo simulation completed successfully for {len(target_mc_models)} model(s)!")

    # DISPLAY MONTE CARLO RESULTS
    if "last_monte_carlo_result" in st.session_state:
        mc_data = st.session_state["last_monte_carlo_result"]
        meta = mc_data.get("metadata", {})
        model_results = mc_data.get("model_results", {})

        st.markdown("---")
        st.subheader("2. Monte Carlo Probabilistic Results")
        st.caption(f"Evaluated **{meta.get('num_iterations', 10000):,}** stochastic runs per model under **{meta.get('uncertainty_mode', 'full_uncertainty').replace('_', ' ').title()}** ({meta.get('distribution', 'normal').title()} distribution).")

        for idx, (m_name, m_res) in enumerate(model_results.items()):
            top_winner = m_res.get("top_probabilistic_winner", "N/A")
            top_pct = m_res.get("top_winner_pct", 0.0)
            c_stats = m_res.get("country_stats", {})
            adv_stats = m_res.get("advantage_stats", {})

            with st.expander(f"🎲 Model: **{m_name}** — Probabilistic Winner: **{top_winner}** ({top_pct:.1f}% win rate)", expanded=(idx == 0)):
                
                col_mc_m1, col_mc_m2, col_mc_m3 = st.columns(3)
                with col_mc_m1:
                    st.metric("Probabilistic Winner", f"🏆 {top_winner}")
                with col_mc_m2:
                    st.metric("Win Probability", f"{top_pct:.2f}%")
                with col_mc_m3:
                    st.metric("Mean Decision Advantage", f"{adv_stats.get('mean_advantage', 0.0):+.4f}")

                v_col1, v_col2 = st.columns([1, 1])
                with v_col1:
                    st.plotly_chart(plot_mc_win_rates(c_stats, countries), use_container_width=True)
                    st.info(
                        "📊 **Graph Guide — Win Rate Distribution (Donut Chart):** "
                        "Shows the percentage of total simulation runs ($N$) won by each alternative. "
                        "A win rate of 100% indicates absolute dominance, whereas split percentages indicate decision vulnerability."
                    )
                with v_col2:
                    st.plotly_chart(plot_mc_advantage_histogram(m_res.get("advantage_subsample", []), meta.get("countries", countries), m_name), use_container_width=True)
                    st.info(
                        "📊 **Graph Guide — Advantage Histogram:**\n"
                        "* **X-Axis:** Score difference between top alternatives ($Score_{A} - Score_{B}$).\n"
                        "* **Y-Axis:** Frequency of simulation runs.\n"
                        "* **How to Interpret:** The red 'Flip Boundary' at zero marks where the winner changes. If the entire histogram sits on one side of zero, your decision is extremely robust."
                    )

                st.markdown("##### 📊 Performance Statistics (Mean & 95% Confidence Interval)")
                mc_table_rows = []
                for c, stats in c_stats.items():
                    mc_table_rows.append({
                        "Alternative": c,
                        "Win Rate (%)": f"{stats['win_percentage']:.2f}%",
                        "Total Wins": f"{stats['total_wins']:,}",
                        "Mean Score": f"{stats['mean_score']:.4f}",
                        "Score StdDev": f"{stats['score_std']:.4f}",
                        "95% Score Range": f"[{stats['ci_95_score'][0]:.4f}, {stats['ci_95_score'][1]:.4f}]",
                        "Mean Rank": f"{stats['mean_rank']:.2f}"
                    })
                st.dataframe(pd.DataFrame(mc_table_rows), use_container_width=True)

        st.markdown("---")
        col_mc_s1, col_mc_s2 = st.columns([2, 1])
        with col_mc_s1:
            default_mc_name = f"MonteCarlo_{len(model_results)}Models_{meta.get('num_iterations', 10000)}iter_{selected_run_id[:8]}"
            mc_save_name = st.text_input("Name this Simulation Experiment:", value=default_mc_name, key="mc_save_name_input")
        with col_mc_s2:
            st.write("")
            st.write("")
            if st.button("💾 Save Monte Carlo Run", key="save_mc_btn", type="secondary"):
                exp_id = AnalysisDispatcher.save_analysis_experiment(mc_data, mc_save_name)
                st.success(f"Saved as '{mc_save_name}' (ID: {exp_id[:12]})")

        st.download_button(
            label="⬇️ Download Monte Carlo JSON",
            data=json.dumps(mc_data, indent=4),
            file_name=f"monte_carlo_{len(model_results)}models_{selected_run_id[:8]}.json",
            mime="application/json",
            key="dl_mc_btn"
        )