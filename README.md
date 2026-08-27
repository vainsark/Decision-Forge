<div align="center">

# ⚖️ Decision Forge

### Multi-Criteria Decision Support System (DSS)

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-Interactive-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/PyMCDM-MCDM%20Engine-2F855A" alt="PyMCDM">
  <img src="https://img.shields.io/badge/Fuzzy%20MCDM-Trapezoidal-805AD5" alt="Fuzzy MCDM">
  <img src="https://img.shields.io/badge/License-MIT-6B46C1" alt="License">
</p>

<p>
  <strong>Decision Forge turns complex, uncertain decisions into structured, testable, and explainable decision models.</strong>
</p>

<p>
  Build a decision hierarchy → assign weights → evaluate alternatives → compare MCDM models → analyze robustness → inspect the result.
</p>

</div>

---

## 🧭 What is Decision Forge?

**Decision Forge** is a modular decision-support application for problems where there is no single obvious "best" answer.

Instead of reducing a decision to one subjective score, the system separates the decision into:

- **Domains and criteria** — what matters
- **Weights** — how important each criterion is
- **Evaluations** — how alternatives perform
- **Uncertainty** — how confident we are in those evaluations
- **MCDM models** — how the evidence is aggregated
- **Sensitivity & robustness analysis** — how stable the conclusion actually is

The application is built around a Streamlit interface with a modular Python backend and supports both deterministic and fuzzy decision models.

---

## ✨ Core Capabilities

### 🗂️ Project & Workspace Management

Decision Forge supports independent decision-model workspaces, allowing separate projects to maintain their own:

- Factors and category hierarchies
- Evaluations
- AHP weights
- Rating configuration
- MCDM runs
- Analysis runs

Projects can be switched, cloned, and backed up/restored as ZIP archives.

### 🌳 Decision Hierarchy

Define a structured hierarchy of:

```text
Domain / Category
        │
        ├── Criterion
        ├── Criterion
        └── Criterion
```

Each criterion carries an optimization direction:

- **Benefit (+1)** — higher is better
- **Cost (-1)** — lower is better

Criterion IDs are managed consistently across the application.

### ⚖️ AHP Weighting

The Weights Engine supports hierarchical weighting through pairwise comparisons and provides:

- Category-level AHP
- Local criterion weights
- Global criterion weights
- Consistency Ratio diagnostics
- Inverse / direct weight tuning
- Automated matrix adjustment utilities

The resulting global weights feed the MCDM engines.

### 📝 Evaluation & Uncertainty Modeling

Alternatives are evaluated using a central rating together with uncertainty parameters:

| Parameter | Meaning |
|---|---|
| **r** | Central evaluation / rating |
| **V** | Volatility / variability |
| **E** | Epistemic uncertainty / confidence uncertainty |
| **bias** | Optimistic, neutral, or pessimistic directional bias |

For fuzzy evaluation, these values are represented as a trapezoidal fuzzy number:

```text
[a, b, c, d]
```

The evaluation layer supports both:

- **Scale / binary criteria** using the absolute decision-rating scale
- **Numeric criteria** using percentage-relative uncertainty bounds

The canonical trapezoid construction is implemented in `src/evaluations.py`.

### 🧮 Multi-Model MCDM

Decision Forge can execute multiple MCDM approaches and compare their rankings rather than relying on a single algorithm.

Current active models include:

- **WSM** — Weighted Sum Model
- **WPM** — Weighted Product Model
- **WASPAS** — Weighted Aggregated Sum Product Assessment
- **TOPSIS** — Technique for Order Preference by Similarity to Ideal Solution
- **PROMETHEE II** — deterministic outranking model
- **Fuzzy PROMETHEE** — trapezoidal fuzzy net-flow analysis

The codebase also contains a **VIKOR** implementation, although it is currently not enabled in the active method registry.

### 🔬 Sensitivity, Robustness & Monte Carlo

Decision Forge is designed to test whether a decision is robust or merely the result of a fragile set of assumptions.

Available analysis includes:

- Parameter sensitivity sweeps
- Weight sensitivity
- Category / criterion leverage analysis
- Tornado-style visualizations
- Decision-boundary scans
- Epistemic uncertainty propagation
- Monte Carlo simulations
- Rank reversals and stability analysis
- Cross-model agreement / consensus analysis

---

## 🧠 Decision Pipeline

The application follows a structured end-to-end workflow:

```text
┌──────────────────────┐
│  0. Decision Hub     │
│  Project / Workspace │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  1. Factors          │
│  Domains + Criteria  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  2. Weights Engine   │
│  AHP + Global Weights│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  3. Evaluations      │
│  r + V + E + Bias    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  4. MCDM Engine      │
│  Deterministic/Fuzzy │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  5. Analytics        │
│  Scores + Rankings   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  6. Robustness       │
│  Sensitivity + MC    │
└──────────────────────┘
```

The goal is not simply to produce a winner. The goal is to understand **why it wins and how stable that conclusion is**.

---

## 🧮 Mathematical Models

### Deterministic Models

| Model | Role |
|---|---|
| **WSM** | Additive utility aggregation |
| **WPM** | Multiplicative utility aggregation |
| **WASPAS** | Hybrid WSM/WPM aggregation controlled by λ |
| **TOPSIS** | Distance from positive and negative ideal solutions |
| **PROMETHEE II** | Outranking based on preference functions and net flows |
| **VIKOR** | Compromise-ranking method; implementation present but currently disabled in the active registry |

### Fuzzy Model

**Fuzzy PROMETHEE** operates on trapezoidal fuzzy evaluations:

```text
(a, b, c, d)
```

and applies:

- Criterion weights
- Benefit / cost directions
- Trapezoidal preference comparison
- Type-V preference behavior
- Indifference threshold `q`
- Preference threshold `p`
- Defuzzification weights
- PROMETHEE net-flow ranking

---

## 📐 Fuzzy Evaluation Model

The evaluation layer is responsible for converting the recorded evaluation parameters into a trapezoidal representation.

For scale / binary criteria, uncertainty is modeled on the absolute rating scale using:

```text
a = r - (E × Ke) - (V × Kv)
b = r - (E × Ke)
c = r + (E × Ke)
d = r + (E × Ke) + (V × Kv)
```

Bias then shifts the appropriate side of the trapezoid toward the central rating.

For numeric criteria, the uncertainty coefficients are interpreted as **relative percentages of the measured value**, allowing the same evaluation framework to represent quantities such as costs, salaries, distances, or other numeric measures.

The implementation also enforces valid trapezoid ordering and scale bounds where applicable.

> **Important:** the exact fuzzy realization is centralized in `calculate_trapezoid()` in `src/evaluations.py`. Analysis components should consume that representation rather than independently redefining the fuzzy geometry.

---

## 📊 Analytics & Robustness

A high MCDM score does not automatically mean a robust decision.

Decision Forge therefore separates the **base decision** from the **analysis of the decision**.

### Sensitivity Analysis

Explore how results change when individual parameters are varied, including:

- Criterion weights
- Category weights
- Defuzzification parameters
- PROMETHEE `q` / `p`
- WASPAS `λ`
- Uncertainty parameters
- Other model-specific assumptions

### Leverage Analysis

Identify which criteria have the greatest influence on the final result and visualize their contribution to the decision.

### Decision Boundaries

Search for the point at which changing a weight or parameter causes the preferred alternative to change.

This makes it possible to distinguish between:

```text
"Alternative A wins."
```

and:

```text
"Alternative A wins, and the result remains stable
across a wide range of plausible assumptions."
```

### Monte Carlo

Monte Carlo analysis samples uncertain evaluations repeatedly and reports probabilistic outcomes such as:

- Win rates
- Ranking distributions
- Score distributions
- Confidence intervals
- Decision advantage

This provides a stochastic view of decision robustness rather than relying only on one deterministic evaluation.

---

## 💾 Historical Runs & Reproducibility

MCDM runs are persisted as JSON snapshots inside the active project workspace.

A run can contain:

- Run metadata
- Alternatives
- Category weights
- Executed methods
- Model parameters
- Results
- Evaluations
- Factor configuration
- Criterion types
- Weight configuration

This provides a foundation for comparing decisions over time and analyzing previously executed models without modifying the underlying project data.

---

## 🏗️ Architecture

Decision Forge separates the user-facing application from the mathematical and data layers.

```text
Decision-Forge/
│
├── Home.py
├── main.py
├── decision_maker.bat
├── requirements.txt
├── README.md
│
├── pages/
│   ├── 0_Decision_Hub.py
│   ├── 1_Global_Settings.py
│   ├── 2_Factors_Overview.py
│   ├── 3_Weights_Engine.py
│   ├── 4_Evaluations.py
│   ├── 5_MCDM_Engine.py
│   ├── 6_Analytics_Dashboard.py
│   └── 7_Sensitivity_and_Robustness.py
│
└── src/
    ├── analysis/
    │   ├── dispatcher.py
    │   ├── epistemic.py
    │   ├── monte_carlo.py
    │   └── sensitivity.py
    │
    ├── evaluations.py
    ├── factors_manager.py
    ├── mcdm_engine.py
    ├── mcdm_fuzzy_engine.py
    ├── mcdm_methods.py
    ├── project_manager.py
    └── weights.py
```

### Main Backend Components

| Module | Responsibility |
|---|---|
| `evaluations.py` | Rating configuration, evaluation data, fuzzy trapezoid construction and persistence |
| `factors_manager.py` | Factor/category configuration and hierarchy management |
| `weights.py` | AHP weighting and weight calculations |
| `mcdm_methods.py` | Deterministic and fuzzy MCDM method implementations |
| `mcdm_engine.py` | Deterministic MCDM orchestration |
| `mcdm_fuzzy_engine.py` | Fuzzy matrix construction, fuzzy execution and run snapshots |
| `project_manager.py` | Multi-project workspace management |
| `analysis/dispatcher.py` | In-memory analysis and historical baseline context |
| `analysis/sensitivity.py` | Sensitivity and leverage analysis |
| `analysis/epistemic.py` | Epistemic uncertainty analysis |
| `analysis/monte_carlo.py` | Stochastic robustness analysis |

---

## 🚀 Installation

### Requirements

- Python **3.10+**
- `pip`
- A virtual environment is recommended

### 1. Clone the repository

```bash
git clone https://github.com/vainsark/Decision-Forge.git
cd Decision-Forge
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The current dependency set includes:

- Streamlit
- pandas
- NumPy
- tabulate
- PyMCDM

---

## ▶️ Running the Application

Decision Forge is primarily presented as a multi-page Streamlit application.

From the repository root:

```bash
streamlit run Home.py
```

### Windows shortcut

You can also use:

```text
decision_maker.bat
```

The repository additionally contains `main.py`, which provides the project's command-line entry point for the underlying decision-support components.

---

## ⚙️ Configuration

Decision Forge stores project-specific configuration in the active workspace.

Key configuration areas include:

| Configuration | Purpose |
|---|---|
| `factors_config.json` | Domains, criteria and optimization directions |
| `weights.json` | Category, local and global weights |
| `evaluations.json` | Alternative evaluations and fuzzy representations |
| `rating_config.json` | Rating coefficients and MCDM parameters |
| `runs/` | Historical MCDM run snapshots |
| `analysis_runs/` | Saved analysis results |

Important rating parameters include:

```text
Kv            Volatility coefficient
Ke            Epistemic uncertainty coefficient
Kb            Bias coefficient

Kv_numeric    Relative volatility coefficient for numeric criteria
Ke_numeric    Relative epistemic coefficient for numeric criteria
Kb_numeric    Relative bias coefficient for numeric criteria

q             PROMETHEE indifference threshold
p             PROMETHEE preference threshold
λ             WASPAS weighting parameter
```

---

## 🔍 Design Principles

Decision Forge is built around a few core principles:

### 1. Separate the decision from the method

A decision should not depend on one MCDM algorithm.

Different models can expose different aspects of the same data.

### 2. Treat uncertainty as data

Uncertainty is explicitly represented rather than hidden inside a single subjective score.

### 3. Preserve the decision context

Weights, evaluations, configurations and model parameters should travel with historical runs.

### 4. Stress-test the conclusion

A result is more useful when you know how sensitive it is to reasonable changes in assumptions.

### 5. Keep the mathematical layers modular

Evaluation, weighting, MCDM execution and robustness analysis are separated so that individual components can evolve without redesigning the entire application.

---

## 🛠️ Development Status

Decision Forge is an actively evolving decision-support framework.

The current repository already provides the complete high-level workflow:

```text
Workspace
   ↓
Factors
   ↓
AHP Weights
   ↓
Evaluations
   ↓
MCDM
   ↓
Analytics
   ↓
Sensitivity / Robustness
```

Some advanced MCDM and fuzzy-analysis components are implemented while others remain intentionally staged for further development. In particular, the repository contains VIKOR and additional fuzzy-method scaffolding that is not currently enabled in the active registry.

---

## 📜 License

Decision Forge is released under the **MIT License**.

See [`LICENSE`](LICENSE) for the full license text.

---

<div align="center">

**⚖️ Decision Forge**

*Structure the decision. Quantify the uncertainty. Stress-test the result.*

</div>
