<div align="center">

# ⚡ Decision Forge
### Enterprise Multi-Criteria Decision Support System (DSS)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-Interactive-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/MCDM-Engine-green.svg" alt="MCDM">
  <img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="License">
</p>

*A powerful, modular decision-making platform featuring multi-model evaluation, hierarchical AHP weighting, epistemic uncertainty modeling, and rigorous Monte Carlo robustness simulations.*

</div>

---

## 🌟 Key Features & Capabilities

* **Dual Weighting Architectures:** Seamlessly toggle between **Dual Hybrid Mode** (structured hierarchical categories and criteria evaluated via AHP pairwise matrices) and **Single Flat Weighting Mode** (unified direct criteria pool).
* **Multi-Model MCDM Orchestrator:** Execute and compare leading decision algorithms side-by-side: **WSM**, **WPM**, **WASPAS**, **TOPSIS**, and **Fuzzy PROMETHEE (Net Flow)**.
* **Advanced AHP Diagnostics & Inverse Tuning:** Features Consistency Ratio (CR) calculation, circular judgment loop detection, automated minimum-revision matrix auto-tuning, and a direct weight slider tuner (Inverse AHP).
* **Fuzzy Trapezoidal Modeling:** Capture decision uncertainty through volatility ($V$), epistemic uncertainty ($E$), and risk-averse bias shifts ($bias$) mapped into crisp trapezoidal bounds $[a, b, c, d]$.
* **Comprehensive Analytics Dashboard:** Visualize multi-model rank agreement, consensus winners, model robustness percentages, and alternative-agnostic performance breakdowns.
* **Robustness & Sensitivity Suite:** Probe model stability across 6 sensitivity dimensions, generate factor leverage Tornado charts, scan critical decision boundaries, propagate epistemic uncertainty intervals, and run stochastic Monte Carlo simulations (< 50,000 iterations).

---

## 📁 System Architecture & Modules

Decision Forge is structured as a multi-page Streamlit application backed by robust, modular backend engines:

| Page / Module | Path | Description |
| :--- | :--- | :--- |
| **Global Settings** | `pages/1_Global_Settings.py` | Configures fuzzy parameters ($Kv, Ke, Kb$), defuzzification weights, PROMETHEE thresholds ($q, p$), and weighting modes. |
| **Weights Engine** | `pages/2_Weights_Engine.py` | Manages AHP pairwise comparisons, consistency ratios (CR), inverse tuning, and preset structures. |
| **Factors Overview** | `pages/1_Factors_Overview.py` | Universal workspace for managing categories, criteria, optimization directions, and automated ID cascading. |
| **MCDM Engine** | `pages/4_MCDM_Engine.py` | Validates decision matrices, runs multi-model algorithms, and archives immutable historical snapshots. |
| **Analytics Dashboard** | `pages/5_Analytics_Dashboard.py` | Renders dynamic consensus metrics, score comparisons, and category/criteria performance decompositions. |
| **Sensitivity & Robustness** | `pages/6_Sensitivity_and_Robustness.py` | Advanced analytical suite for parameter sweeps, Tornado charts, stability bounds, and Monte Carlo sims. |

---

## 🧮 Supported Decision Models

Decision Forge integrates rigorous mathematical implementations via `pymcdm` and custom fuzzy analytics engines:

* **Weighted Sum Model (WSM):** Linear additive utility model.
* **Weighted Product Model (WPM):** Multiplicative compounding utility model.
* **WASPAS:** Hybrid weighted aggregate sum product assessment controlled by tuning parameter $\lambda$.
* **TOPSIS:** Technique for Order Preference by Similarity to Ideal Solution (geometric distance metric).
* **Fuzzy PROMETHEE:** Advanced outranking method utilizing Type V preference functions and net preference flows across trapezoidal bounds.

---

## 🔬 Sensitivity & Robustness Analysis Framework

> 💡 **Built for Defensible Decisions:** Decision Forge includes deep analytical instruments to stress-test your decision model against parameter bias and data uncertainty.

1. **Parameter Sweeps:** Track exact score trajectories and rank reversals across weights, defuzzification schemes, PROMETHEE thresholds ($q, p$), and WASPAS $\lambda$.
2. **Leverage Tornado Charts:** Visually isolate which factors exert the strongest leverage over the winning alternative.
3. **Decision Boundary Scanners:** High-resolution scans from 0% to 100% weight to find exact flip thresholds and safe stability zones.
4. **Epistemic Uncertainty Propagation:** Propagate confidence intervals into deterministic models using One-At-A-Time (OAT) and compounding scenarios.
5. **Monte Carlo Simulations:** Stochastically sample ratings across uncertainty bounds to output probabilistic win rates and 95% confidence intervals.

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository

### 2. Create a virtual environment 

        python -m venv .venv

####    Windows:

        .venv\Scripts\activate

####    macOS / Linux:

        source .venv/bin/activate

### 3. Install dependencies

        pip install -r requirements.txt

### 4. Run the application

    Option 1:
    Run decision_maker.bat

    Option 2:
    
        streamlit run app.py