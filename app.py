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
    .kpi-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: left;
    }
    .kpi-label {
        font-size: 0.80rem;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .flag-img {
        width: 22px;
        height: 16px;
        border-radius: 3px;
        vertical-align: middle;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3);
        margin-right: 4px;
        margin-bottom: 2px;
    }
    
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
# LIVE SYSTEM DATA TELEMETRY & OPTIONAL FLAG/ENTITY PARSER
# =========================================================================
FLAG_CODE_MAP = {
    "usa": "us", "united states": "us", "germany": "de",
    "uk": "gb", "united kingdom": "gb", "france": "fr", "canada": "ca",
    "japan": "jp", "australia": "au", "spain": "es", "italy": "it"
}

def get_entity_html(entity_name: str) -> str:
    c_lower = entity_name.strip().lower()
    code = FLAG_CODE_MAP.get(c_lower, "")
    if code:
        img_tag = f'<img src="https://flagcdn.com/28x21/{code}.png" class="flag-img" alt="{entity_name}">'
        return f'{img_tag}<span>{entity_name}</span>'
    return f'<span>{entity_name}</span>'

def get_system_telemetry():
    domains_cnt = 0
    factors_cnt = 0
    alternatives_set = set()
    runs_cnt = 0
    analyses_cnt = 0

    f_path = os.path.join(DATA_DIR, 'factors_config.json')
    if os.path.exists(f_path):
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                f_data = json.load(f)
                domains_cnt = len(f_data.get("domains", []))
                factors_cnt = len(f_data.get("factors", []))
        except Exception:
            pass

    e_path = os.path.join(DATA_DIR, 'evaluations.json')
    if os.path.exists(e_path):
        try:
            with open(e_path, 'r', encoding='utf-8') as f:
                e_data = json.load(f)
                for e in e_data:
                    alt_key = next((k for k in ["country", "alternative", "option"] if k in e), None)
                    if alt_key:
                        alternatives_set.add(e[alt_key])
        except Exception:
            pass

    r_dir = os.path.join(DATA_DIR, 'runs')
    if os.path.exists(r_dir):
        runs_cnt = len([f for f in os.listdir(r_dir) if f.endswith('.json')])

    a_dir = os.path.join(DATA_DIR, 'analysis_runs')
    if os.path.exists(a_dir):
        analyses_cnt = len([f for f in os.listdir(a_dir) if f.endswith('.json')])

    return domains_cnt, factors_cnt, sorted(list(alternatives_set)), runs_cnt, analyses_cnt

domains_n, factors_n, raw_alternatives, runs_n, analyses_n = get_system_telemetry()

if raw_alternatives:
    arena_html = " <span style='color: #718096; margin: 0 4px;'>vs</span> ".join([get_entity_html(alt) for alt in raw_alternatives])
else:
    arena_html = "<span>Alternative A</span> <span style='color: #718096; margin: 0 4px;'>vs</span> <span>Alternative B</span>"

# =========================================================================
# HEADER & HERO SECTION
# =========================================================================
st.title(f"⚖️ {project_title}")
st.caption(project_caption)

st.markdown("---")

# Telemetry KPI Strip
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns([1, 1, 1.5, 1, 1])

with kpi1:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Categories</div>
        <div class="kpi-value">{domains_n} Domains</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Criteria Pool</div>
        <div class="kpi-value">{factors_n} Factors</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Decision Scope</div>
        <div class="kpi-value">{arena_html}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Baseline Runs</div>
        <div class="kpi-value">{runs_n} Runs</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Stress Archive</div>
        <div class="kpi-value">{analyses_n} Tests</div>
    </div>
    """, unsafe_allow_html=True)

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