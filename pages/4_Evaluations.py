"""
Decision Support System - Criteria Evaluations Page
Universal evaluation workspace for scoring alternatives across criteria, 
modeling uncertainty intervals as trapezoidal fuzzy numbers, and visualizing matrices.
"""

import streamlit as st
import pandas as pd
import os
import sys
import base64
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.factors_manager import load_factors_config
from src.evaluations import load_rating_config, load_evaluations, calculate_trapezoid, save_evaluations
from src.project_manager import get_active_project_dir
from src.project_manager import get_active_project_id


# Dynamic path resolution functions for active project workspace isolation
def _get_project_data_dir() -> str:
    """Returns the active project directory, asserting it is not None."""
    proj_dir = get_active_project_dir()
    assert proj_dir is not None, "Active project directory is required."
    return proj_dir

def get_evaluations_filepath() -> str:
    return os.path.join(_get_project_data_dir(), 'evaluations.json')

def get_rating_config_filepath() -> str:
    return os.path.join(_get_project_data_dir(), 'rating_config.json')

def get_evaluation_presets_dir() -> str:
    d = os.path.join(_get_project_data_dir(), 'evaluation_presets')
    if not os.path.exists(d):
        os.makedirs(d)
    return d

st.set_page_config(page_title="Criteria Evaluations", page_icon="📝", layout="wide")

# ==========================================
# ACTIVE PROJECT GUARD (PASTE HERE)
# ==========================================
active_proj_id = get_active_project_id()
if not active_proj_id:
    st.warning("⚠️ **No Active Project Selected.** Please go to the **Decision Hub** to create or open a project workspace.")
    if st.button("🗂️ Go to Decision Hub", type="primary"):
        st.switch_page("pages/0_Decision_Hub.py")
    st.stop()

# ==========================================
# EVALUATION PRESET MANAGER (SIDEBAR)
# ==========================================
PRESET_EVALS_DIR = get_evaluation_presets_dir()

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Evaluation Presets")

eval_presets = [f.replace(".json", "") for f in os.listdir(PRESET_EVALS_DIR) if f.endswith(".json")]

save_mode = st.sidebar.radio("Save Action:", ["Save as New", "Overwrite Existing"], key="eval_save_mode", horizontal=True)

if save_mode == "Save as New" or not eval_presets:
    new_eval_name = st.sidebar.text_input("New Preset Name:", value="My_Evaluation_Set", key="eval_preset_input")
    if st.sidebar.button("💾 Save New Preset", key="save_eval_preset"):
        if new_eval_name.strip():
            preset_path = os.path.join(PRESET_EVALS_DIR, f"{new_eval_name.strip()}.json")
            current_evs = load_evaluations(load_rating_config(), load_factors_config())
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(current_evs, f, indent=4)
            st.sidebar.success(f"Saved new preset '{new_eval_name}'!")
            st.rerun()
        else:
            st.sidebar.error("Enter a valid name.")
else:
    target_eval_preset = st.sidebar.selectbox("Select Preset to Overwrite:", eval_presets, key="overwrite_eval_sel")
    if st.sidebar.button("⚠️ Overwrite Preset", key="overwrite_eval_btn"):
        preset_path = os.path.join(PRESET_EVALS_DIR, f"{target_eval_preset}.json")
        current_evs = load_evaluations(load_rating_config(), load_factors_config())
        with open(preset_path, 'w', encoding='utf-8') as f:
            json.dump(current_evs, f, indent=4)
        st.sidebar.success(f"Overwrote preset '{target_eval_preset}' successfully!")
        st.rerun()

if eval_presets:
    st.sidebar.markdown("---")
    selected_eval = st.sidebar.selectbox("Load Preset:", eval_presets, key="load_eval_sel")
    if st.sidebar.button("📂 Load Preset into App", key="load_eval_btn"):
        preset_path = os.path.join(PRESET_EVALS_DIR, f"{selected_eval}.json")
        with open(preset_path, 'r', encoding='utf-8') as f:
            loaded_evs = json.load(f)
        save_evaluations(loaded_evs)
        st.sidebar.success(f"Loaded '{selected_eval}' successfully!")
        st.rerun()

# ==============================================================================
# NATIVE DATAFRAME SVG GENERATOR WITH PROFESSIONAL GRID & TICKS
# ==============================================================================
def generate_mini_trapezoid_svg(a: float, b: float, c: float, d: float, r: float, color="#38bdf8", width=500, height=60, eval_type="scale", min_val=0.0, max_val=10.0, in_cell=False) -> str:
    pad = 16  
    w = width - (2 * pad)
    span = max(0.001, max_val - min_val)
    def scale(val): 
        clamped = max(min_val, min(max_val, float(val)))
        return pad + ((clamped - min_val) / span) * w
        
    x_a, x_b, x_c, x_d, x_r = scale(a), scale(b), scale(c), scale(d), scale(r)
    strip_w = 5 if in_cell else 4
    
    grid_elements = ""
    
    if in_cell:
        # Optimized 5-tick layout with large, high-contrast text for table cells
        height = 52
        for i in range(7):
            val = min_val + (i / 6.0) * span
            x_pos = scale(val)
            if eval_type == "scale":
                label = f"{int(val)}"
            elif eval_type == "binary":
                label = f"{int(val)}"
            else:
                if abs(val) >= 1000000:
                    label = f"{val/1000000:.1f}M" if val % 1000000 != 0 else f"{int(val/1000000)}M"
                elif abs(val) >= 1000:
                    label = f"{val/1000:.1f}k" if val % 1000 != 0 else f"{int(val/1000)}k"
                elif abs(val) >= 10 or val == 0:
                    label = f"{val:.0f}"
                else:
                    label = f"{val:.1f}"
            grid_elements += f'<line x1="{x_pos}" y1="2" x2="{x_pos}" y2="25" stroke="#777" stroke-width="1.2" />\n'
            grid_elements += f'<text x="{x_pos}" y="46" font-family="sans-serif" font-size="18" font-weight="bold" fill="#ffffff" text-anchor="middle">{label}</text>\n'
    else:
        # Detailed standard layout for editor view
        if eval_type == "scale":
            for i in range(11):
                val = float(i)
                x_pos = scale(val)
                grid_elements += f'<line x1="{x_pos}" y1="2" x2="{x_pos}" y2="22" stroke="#444" stroke-width="0.8" />\n'
                grid_elements += f'<text x="{x_pos}" y="38" font-family="sans-serif" font-size="10" fill="#888" text-anchor="middle">{int(val)}</text>\n'
        elif eval_type == "binary":
            for val in [0.0, 1.0]:
                x_pos = scale(val)
                grid_elements += f'<line x1="{x_pos}" y1="2" x2="{x_pos}" y2="22" stroke="#444" stroke-width="0.8" />\n'
                grid_elements += f'<text x="{x_pos}" y="38" font-family="sans-serif" font-size="10" fill="#888" text-anchor="middle">{int(val)}</text>\n'
        else:
            for i in range(11):
                val = min_val + (i / 10.0) * span
                x_pos = scale(val)
                if abs(val) >= 1000000:
                    label = f"{val/1000000:.1f}M" if val % 1000000 != 0 else f"{int(val/1000000)}M"
                elif abs(val) >= 1000:
                    label = f"{val/1000:.1f}k" if val % 1000 != 0 else f"{int(val/1000)}k"
                elif abs(val) >= 10 or val == 0:
                    label = f"{val:.0f}"
                else:
                    label = f"{val:.1f}"
                grid_elements += f'<line x1="{x_pos}" y1="2" x2="{x_pos}" y2="22" stroke="#444" stroke-width="0.8" />\n'
                grid_elements += f'<text x="{x_pos}" y="38" font-family="sans-serif" font-size="8.5" fill="#888" text-anchor="middle">{label}</text>\n'

    rect_h = 26 if in_cell else 20
    poly_y = 26 if in_cell else 21
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
        <rect x="0" y="2" width="{width}" height="{rect_h}" rx="10" fill="#2b2b36" />
        {grid_elements}
        <polygon points="{x_a},{poly_y} {x_b},3 {x_c},3 {x_d},{poly_y}" fill="{color}" opacity="0.9" />
        <rect x="{x_r - strip_w/2}" y="3" width="{strip_w}" height="{rect_h - 3}" fill="#ffffff" rx="1.5" />
    </svg>'''
    
    b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"

def get_metric_icon(val: float, metric_type: str) -> str:
    if metric_type == "rating":
        if val >= 7.0: return "🟢"
        elif val >= 4.0: return "🟠"
        else: return "🔴"
    elif metric_type in ["volatility", "uncertainty"]:
        if val <= 1.0: return "🟢"
        elif val <= 3.0: return "🟠"
        else: return "🔴"
    return ""

BIAS_FORMAT_MAP = {"opt": "OPTIMISTIC 🟢", "neutral": "NEUTRAL ⚪", "pes": "PESSIMISTIC 🔴"}

ALTERNATIVE_COLORS = ["#38bdf8", "#ff4b4b", "#a855f7", "#22c55e", "#eab308", "#ec4899", "#06b6d4"]

def get_alternative_color(alt_name: str, alternatives_list: list) -> str:
    try:
        idx = alternatives_list.index(alt_name)
        return ALTERNATIVE_COLORS[idx % len(ALTERNATIVE_COLORS)]
    except ValueError:
        return "#38bdf8"

# ==========================================
# LOAD & VALIDATE DATA
# ==========================================
config = load_factors_config()
rating_config = load_rating_config()
evaluations = load_evaluations(rating_config, config)

alternatives = rating_config.get("alternatives", rating_config.get("countries", []))
factors = config.get("factors", [])
domains = config.get("domains", [])
category_map = {d["id"]: d["name"] for d in domains}
coeffs = rating_config.get("coefficients", {"Kv": 0.5, "Ke": 0.5, "Kb": 1.0})

if not alternatives:
    st.warning("No alternatives found. Please add decision alternatives in Global Settings first.")
    st.stop()
if not factors:
    st.warning("No criteria found. Please add criteria in the Weights Engine first.")
    st.stop()

for ev in evaluations:
    if "alternative" not in ev and "country" in ev:
        ev["alternative"] = ev["country"]

expected_pairs = [(f["id"], alt) for f in factors for alt in alternatives]
existing_pairs = [(e["criterion_id"], e.get("alternative")) for e in evaluations]
missing_pairs = [p for p in expected_pairs if p not in existing_pairs]

# ==========================================
# REAL-TIME UI RENDERER (MIXED TYPES WITH GRIDS)
# ==========================================
def render_interactive_rating_ui(alternative, factor, current_val=None):
    base_key = f"{alternative}_{factor['id']}"
    eval_type = factor.get("evaluation_type", "scale")
    unit = factor.get("unit", "")
    unit_str = f" ({unit})" if unit else ""
    
    default_r = float(current_val["rating"]) if current_val else (5.0 if eval_type == "scale" else (0.0 if eval_type == "numeric" else 1.0))
    default_v = int(current_val["volatility"]) if current_val else 0
    default_u = int(current_val["uncertainty"]) if current_val else 0
    
    bias_opts = {"Neutral": "neutral", "Optimistic": "opt", "Pessimistic": "pes"}
    default_b_idx = list(bias_opts.values()).index(current_val["bias"]) if current_val and current_val["bias"] in bias_opts.values() else 0

    c1, c2, c3, c4 = st.columns(4)
    
    if eval_type == "numeric":
        r = c1.number_input(
            f"Base Value{unit_str}", 
            value=default_r, 
            step=1.0, 
            format="%.2f",
            key=f"r_{base_key}",
            help=f"Enter the absolute numeric value for {factor['name']} {unit_str}."
        )
    elif eval_type == "binary":
        bin_options = ["Yes (1.0)", "No (0.0)"]
        default_bin_idx = 0 if default_r >= 0.5 else 1
        bin_choice = c1.selectbox(
            "Binary Choice", 
            bin_options, 
            index=default_bin_idx,
            key=f"r_{base_key}",
            help="Select Yes or No for this binary criterion."
        )
        r = 1.0 if "Yes" in bin_choice else 0.0
    else:
        r = c1.slider(
            "Base Rating (0–10)", 0.0, 10.0, default_r, 0.5, 
            key=f"r_{base_key}",
            help="Your core expected score. Click anywhere along the slider track to jump directly to a value."
        )

    v = c2.slider(
        "Volatility (0–5)", 0, 5, default_v, 1, 
        key=f"v_{base_key}",
        help="How much this score fluctuates or swings under different real-world conditions (risk level)."
    )
    u = c3.slider(
        "Uncertainty (0–5)", 0, 5, default_u, 1, 
        key=f"u_{base_key}",
        help="How confident you are in your data. Higher uncertainty widens safety margins."
    )
    b_label = c4.selectbox(
        "Bias Shift", list(bias_opts.keys()), index=default_b_idx, 
        key=f"b_{base_key}",
        help="Evaluates this alternative leaning toward Optimistic, Pessimistic, or Neutral expectations."
    )
    b = bias_opts[b_label]

    trap = calculate_trapezoid(r, v, u, b, coeffs, criterion_id=factor["id"])
    alt_color = get_alternative_color(alternative, alternatives)
    
    # Self-adjusting range formula: r * 0.5 to r * 1.5
    if eval_type == "numeric":
        if r <= 0:
            min_val, max_val = 0.0, 100.0
        else:
            min_val = r * 0.5
            max_val = r * 1.5
    elif eval_type == "binary":
        min_val, max_val = 0.0, 1.0
    else:
        min_val, max_val = 0.0, 10.0

    st.markdown("**Real-Time Geometry (Fuzzy Interval):**")
    svg_b64 = generate_mini_trapezoid_svg(trap[0], trap[1], trap[2], trap[3], r, color=alt_color, width=600, height=60, eval_type=eval_type, min_val=min_val, max_val=max_val)
    st.markdown(f'<img src="{svg_b64}" width="100%">', unsafe_allow_html=True)
    st.caption(f"Calculated Bounds [Worst, Lower Core, Upper Core, Best]: [{trap[0]:.2f}, {trap[1]:.2f}, {trap[2]:.2f}, {trap[3]:.2f}]")
    
    return r, v, u, b, trap

# ==========================================
# TRIGGER: INITIALIZATION MODE
# ==========================================
st.title("📊 Criteria Evaluations & Visual Matrix")
st.caption("Score decision alternatives across criteria while modeling volatility, epistemic uncertainty, and psychological bias.")

if missing_pairs:
    st.markdown("---")
    
    target_fid, target_alt = missing_pairs[0]
    target_factor = next(f for f in factors if f["id"] == target_fid)
    cat_name = category_map.get(target_factor["domain_id"], "Unknown")
    
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(56, 189, 248, 0.02) 100%); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 22px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">
                    📁 Category: {cat_name}
                </span>
                <span style="font-size: 0.85rem; color: #38bdf8; font-weight: 600; background: rgba(56, 189, 248, 0.15); padding: 3px 10px; border-radius: 20px;">
                    {len(missing_pairs)} evaluation(s) remaining
                </span>
            </div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #38bdf8; margin-bottom: 8px;">
                🎯 Alternative: {target_alt}
            </div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #ffffff; margin-bottom: 10px;">
                📄 Criterion: {target_factor['name']}
            </div>
            <div style="font-size: 0.95rem; color: #cbd5e1; line-height: 1.4; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px; margin-top: 6px;">
                {target_factor.get('description', 'No description provided.')}
            </div>
        </div>
    """, unsafe_allow_html=True)
            
    r, v, u, b, trap = render_interactive_rating_ui(target_alt, target_factor)
    
    if st.button("💾 Save & Next", type="primary"):
        new_ev = {
            "criterion_id": target_fid, "alternative": target_alt, "country": target_alt,
            "rating": r, "volatility": v, "uncertainty": u,
            "bias": b, "coefficients": coeffs, "trapezoid": trap
        }
        evaluations.append(new_ev)
        save_evaluations(evaluations)
        st.rerun()

# ==========================================
# NORMAL DASHBOARD
# ==========================================
else:
    tab_matrix, tab_edit, tab_danger = st.tabs(["📊 Evaluation Matrix", "✏️ Targeted Editor", "⚠️ Danger Zone"])
    
    with tab_matrix:
        category_options = ["All Categories"] + [d["name"] for d in domains]
        selected_view = st.selectbox("🔍 Filter Table by Category:", category_options)

        if selected_view == "All Categories": 
            display_factors = factors
        else:
            sel_id = next(d["id"] for d in domains if d["name"] == selected_view)
            display_factors = [f for f in factors if f["domain_id"] == sel_id]

        table_data = []
        for f in display_factors:
            eval_type = f.get("evaluation_type", "scale")
            unit = f.get("unit", "")
            row = {"ID": f["id"], "Category": category_map.get(f.get("domain_id"), "Unknown"), "Criterion": f["name"]}
            for alt in alternatives:
                ev = next((e for e in evaluations if e["criterion_id"] == f["id"] and e.get("alternative") == alt), None)
                if ev:
                    a, b_val, c_val, d = ev["trapezoid"]
                    r, v, u = float(ev["rating"]), int(ev["volatility"]), int(ev["uncertainty"])
                    alt_color = get_alternative_color(alt, alternatives)
                    
                    bias_text = BIAS_FORMAT_MAP.get(ev['bias'], "NEUTRAL 🟡")
                    val_str = f"{r:.2f}" if eval_type == "numeric" else f"{r:.1f}"
                    unit_suffix = f" {unit}" if eval_type == "numeric" and unit else ""
                    
                    row[f"{alt} Data"] = f"Val: {val_str}{unit_suffix} | V: {v} | U: {u} | {bias_text}"
                    
                    if eval_type == "numeric":
                        min_v = 0.0 if r <= 0 else r * 0.5
                        max_v = 100.0 if r <= 0 else r * 1.5
                    elif eval_type == "binary":
                        min_v, max_v = 0.0, 1.0
                    else:
                        min_v, max_v = 0.0, 10.0
                        
                    row[f"{alt} Visual"] = generate_mini_trapezoid_svg(a, b_val, c_val, d, r, color=alt_color, eval_type=eval_type, min_val=min_v, max_val=max_v, in_cell=True)
                else:
                    row[f"{alt} Data"] = "N/A"
                    row[f"{alt} Visual"] = None
                    
            table_data.append(row)

        column_configuration = {
            "ID": st.column_config.TextColumn("ID", width=40),
            "Category": st.column_config.TextColumn("Category", width="medium"),
            "Criterion": st.column_config.TextColumn("Criterion", width="medium"),
        }
        for alt in alternatives:
            column_configuration[f"{alt} Data"] = st.column_config.TextColumn(f"{alt} Data", width=240)
            column_configuration[f"{alt} Visual"] = st.column_config.ImageColumn(f"{alt} (Fuzzy Interval)", width=290)

        matrix_height = min(900, max(480, len(table_data) * 55 + 60))

        st.dataframe(
            pd.DataFrame(table_data), 
            column_config=column_configuration, 
            hide_index=True, 
            use_container_width=True,
            height=matrix_height
        )

    with tab_edit:
        st.header("✏️ Targeted Rating Editor")
        st.caption("Select an alternative, category, and criterion to fine-tune its performance score and uncertainty parameters.")
        
        col_cat, col_crit, col_alt = st.columns(3)
        with col_cat:
            sel_cat_name = st.selectbox("1. Select Category:", [d["name"] for d in domains])
            sel_cat_id = next(d["id"] for d in domains if d["name"] == sel_cat_name)
            
        with col_crit:
            cat_factors = sorted([f for f in factors if f["domain_id"] == sel_cat_id], key=lambda x: x["id"])
            if not cat_factors:
                st.warning("No criteria exist in this category.")
                st.stop()
                
            crit_opts = {f"[{f['id']}] {f['name']}": f for f in cat_factors}
            sel_crit_label = st.selectbox("2. Select Criterion:", list(crit_opts.keys()))
            selected_factor = crit_opts[sel_crit_label]
            
        with col_alt:
            edit_alternative = st.selectbox("3. Select Alternative:", alternatives)
            
        st.info(f"**Description:** {selected_factor.get('description', 'No description provided.')}")
        
        current_ev = next((e for e in evaluations if e["criterion_id"] == selected_factor["id"] and e.get("alternative") == edit_alternative), None)
        
        r, v, u, b, trap = render_interactive_rating_ui(edit_alternative, selected_factor, current_ev)
        
        if st.button("💾 Update Rating", type="primary"):
            updated = False
            for i, e in enumerate(evaluations):
                if e["criterion_id"] == selected_factor["id"] and e.get("alternative") == edit_alternative:
                    evaluations[i].update({
                        "rating": r, "volatility": v, "uncertainty": u, 
                        "bias": b, "coefficients": coeffs, "trapezoid": trap,
                        "alternative": edit_alternative, "country": edit_alternative
                    })
                    updated = True
                    break
            
            if not updated:
                evaluations.append({
                    "criterion_id": selected_factor["id"], "alternative": edit_alternative, "country": edit_alternative,
                    "rating": r, "volatility": v, "uncertainty": u, 
                    "bias": b, "coefficients": coeffs, "trapezoid": trap
                })
                
            save_evaluations(evaluations)
            st.success(f"Rating for '{edit_alternative}' updated successfully!")
            st.rerun()

    with tab_danger:
        st.header("⚠️ Danger Zone")
        st.warning("This will permanently delete all evaluations for all decision alternatives.")
        if st.button("🗑️ Wipe All Ratings", type="primary"):
            save_evaluations([])
            st.success("All ratings deleted. Reinitializing...")
            st.rerun()