"""
Decision Support System - Executive Landing Dashboard (Decision Home)
Interactive home dashboard summarizing system status, live decision telemetry,
clickable end-to-end pipeline cards, and mathematical architecture.
"""

import os
import json
import streamlit as st

# =========================================================================
# CONFIG & METADATA LOADER
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_FILE = os.path.join(DATA_DIR, 'rating_config.json')

# Default generalized metadata
default_title = "Multi-Criteria Decision Support System"
default_caption = "A generalized evaluation architecture integrating workspace management, AHP weighting, deterministic MCDM models, Fuzzy intervals, and uncertainty propagation."

project_title = default_title
project_caption = default_caption

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            project_title = cfg.get("project_title", default_title)
            project_caption = cfg.get("project_caption", default_caption)
    except Exception:
        pass

st.set_page_config(
    page_title=project_title,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Custom CSS for sleek glassmorphic card styling, clickable card links, and badges[cite: 1]
st.markdown("""
<style>
    /* Clickable Container Card Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
        padding: 18px 22px !important;
        margin-bottom: 12px !important;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(74, 144, 226, 0.8) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
    }
    
    /* Sleek primary button styling inside cards */
    div[data-testid="stVerticalBlockBorderWrapper"] a[data-testid="stPageLink-NavLink"] {
        background-color: rgba(74, 144, 226, 0.15) !important;
        border: 1px solid rgba(74, 144, 226, 0.4) !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        text-decoration: none !important;
        display: inline-flex !important;
        width: 100% !important;
        justify-content: space-between !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] a[data-testid="stPageLink-NavLink"]:hover {
        background-color: rgba(74, 144, 226, 0.4) !important;
        border-color: #4A90E2 !important;
        color: #FFFFFF !important;
    }
    
    .pipeline-step {
        font-weight: 800;
        color: #4A90E2;
        text-transform: uppercase;
        font-size: 0.76rem;
        letter-spacing: 1.2px;
        margin-bottom: 2px;
    }
    .card-desc {
        color: #A0AEC0;
        font-size: 0.90rem;
        line-height: 1.5;
        margin-top: 6px;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 4px;
        background-color: rgba(74, 144, 226, 0.12);
        color: #4A90E2;
        border: 1px solid rgba(74, 144, 226, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# HEADER & HERO SECTION
# =========================================================================
st.title(f"⚖️ {project_title}")
st.caption(project_caption)

st.markdown("---")

# =========================================================================
# CLICKABLE DECISION PIPELINE ARCHITECTURE[cite: 1]
# =========================================================================
st.subheader("🧭 Interactive Decision Pipeline")
st.caption("Click on any step card below to open its module directly.")

c1, c2 = st.columns(2)

with c1:
    # STEP 0
    with st.container(border=True):
        st.markdown('<div class="pipeline-step">Step 0 — Workspace Control</div>', unsafe_allow_html=True)
        st.page_link("pages/0_Decision_Hub.py", label="🗂️ 0. Decision Hub (Workspaces) ➔", use_container_width=True)
        st.markdown("""
        <div class="card-desc">
            Manage your multi-project workspace collection. Switch active projects, create new independent decision models, clone workspaces, or backup/restore via ZIP archives.
        </div>
        <div>
            <span class="badge">Multi-Project Isolation</span>
            <span class="badge">ZIP Archive Backup</span>
            <span class="badge">Active Workspace</span>
        </div>
        """, unsafe_allow_html=True)

    # STEP 1
    with st.container(border=True):
        st.markdown('<div class="pipeline-step">Step 1 — Foundation</div>', unsafe_allow_html=True)
        st.page_link("pages/2_Factors_Overview.py", label="🗂️ 1. Factors & Category Hierarchy ➔", use_container_width=True)
        st.markdown("""
        <div class="card-desc">
            Explore and structure the decision hierarchy across high-level domains and granular criteria with designated Benefit (+1) and Cost (-1) properties.
        </div>
        <div>
            <span class="badge">Category Hierarchy</span>
            <span class="badge">Benefit/Cost Types</span>
            <span class="badge">Cascading IDs</span>
        </div>
        """, unsafe_allow_html=True)

    # STEP 2
    with st.container(border=True):
        st.markdown('<div class="pipeline-step">Step 2 — Preference Calibration</div>', unsafe_allow_html=True)
        st.page_link("pages/3_Weights_Engine.py", label="⚖️ 2. Hybrid Weights Engine ➔", use_container_width=True)
        st.markdown("""
        <div class="card-desc">
            Calibrate high-level domain priorities using PyMCDM AHP pairwise comparisons with automated Consistency Ratio (CR &lt; 0.10) diagnostics and local scoring.
        </div>
        <div>
            <span class="badge">PyMCDM AHP</span>
            <span class="badge">Consistency Ratio</span>
            <span class="badge">Rank-Locked Auto-Tune</span>
        </div>
        """, unsafe_allow_html=True)

    # STEP 3
    with st.container(border=True):
        st.markdown('<div class="pipeline-step">Step 3 — Rating & Uncertainty</div>', unsafe_allow_html=True)
        st.page_link("pages/4_Evaluations.py", label="📝 3. Decision Matrix & Evaluations ➔", use_container_width=True)
        st.markdown("""
        <div class="card-desc">
            Score alternatives on central ratings (r) while modeling volatility (V) and epistemic uncertainty (E) as 4-point trapezoidal fuzzy numbers [a, b, c, d].
        </div>
        <div>
            <span class="badge">Rating (1–10)</span>
            <span class="badge">Volatility (V)</span>
            <span class="badge">Epistemic (E)</span>
            <span class="badge">Trapezoids</span>
        </div>
        """, unsafe_allow_html=True)

with c2:
    # STEP 4
    with st.container(border=True):
        st.markdown('<div class="pipeline-step">Step 4 — Multi-Model Synthesis</div>', unsafe_allow_html=True)
        st.page_link("pages/5_MCDM_Engine.py", label="🧮 4. Multi-Model MCDM Engine ➔", use_container_width=True)
        st.markdown("""
        <div class="card-desc">
            Execute deterministic models (WSM, WPM, WASPAS, TOPSIS, VIKOR) alongside Fuzzy PROMETHEE to generate consensus winners and immutable historical snapshots.
        </div>
        <div>
            <span class="badge">PyMCDM Deterministic</span>
            <span class="badge">Fuzzy PROMETHEE</span>
            <span class="badge">Full Snapshots</span>
        </div>
        """, unsafe_allow_html=True)

    # STEP 5
    with st.container(border=True):
        st.markdown('<div class="pipeline-step">Step 5 — Exploration & Diagnostics</div>', unsafe_allow_html=True)
        st.page_link("pages/6_Analytics_Dashboard.py", label="📊 5. Analytics Dashboard ➔", use_container_width=True)
        st.markdown("""
        <div class="card-desc">
            Deep-dive into multi-model score distributions, ranking agreements, category contribution decompositions, and comparative alternative profiles.
        </div>
        <div>
            <span class="badge">Score Decompositions</span>
            <span class="badge">Rank Agreement</span>
            <span class="badge">Comparative Charts</span>
        </div>
        """, unsafe_allow_html=True)

    # STEP 6
    with st.container(border=True):
        st.markdown('<div class="pipeline-step">Step 6 — Stress-Testing & Verification</div>', unsafe_allow_html=True)
        st.page_link("pages/7_Sensitivity_and_Robustness.py", label="🔬 6. Sensitivity, Robustness & Monte Carlo ➔", use_container_width=True)
        st.markdown("""
        <div class="card-desc">
            Stress-test decisions across sensitivity dimensions, Tornado leverage rankings, critical decision boundaries, multi-level epistemic propagation, and Monte Carlo simulations.
        </div>
        <div>
            <span class="badge">Tornado Leverage</span>
            <span class="badge">Decision Boundaries</span>
            <span class="badge">Monte Carlo Simulations</span>
        </div>
        """, unsafe_allow_html=True)

# =========================================================================
# RETRACTABLE MATHEMATICAL SPECIFICATIONS
# =========================================================================
st.markdown("---")

with st.expander("📐 Mathematical Framework & Method Specifications", expanded=False):
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("**Deterministic Models (PyMCDM)**")
        st.markdown("""
        * **WSM & WPM:** Additive and multiplicative utility synthesis.
        * **WASPAS:** Joint WSM/WPM optimization with dynamic $\\lambda$ weighting.
        * **TOPSIS:** Geometric Euclidean distance to positive-ideal ($A^+$) and negative-ideal ($A^-$) solutions.
        * **VIKOR:** Compromise ranking based on maximum group utility ($S$) and individual regret ($R$).
        """)

    with m2:
        st.markdown("**Fuzzy Mathematics & PROMETHEE**")
        st.markdown("""
        * **Trapezoidal Realization:** $a = r - K_v V - K_e E$, $d = r + K_v V + K_e E$.
        * **Directional Asymmetry:** Skewed core points ($b, c$) driven by bias coefficient ($K_b$).
        * **Type V Preference Function:** Symmetric linear preference flow with indifference ($q$) and strict preference ($p$) thresholds.
        """)

    with m3:
        st.markdown("**Stochastic Simulation (Monte Carlo)**")
        st.markdown("""
        * **Vectorized Matrix Sampling:** Simultaneous perturbation across all criteria and alternatives.
        * **Controlled Dispersion:** Truncated normal and uniform distributions over $[r - \\Delta, r + \\Delta]$.
        * **Aggregate Diagnostics:** Empirical win rates, confidence intervals, and decision advantage distributions.
        """)

st.caption("Use the sidebar on the left or the interactive cards above to navigate through the modules.")