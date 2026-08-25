"""
Decision Support System - Categories & Criteria Overview Page
Universal workspace for exploring, expanding, editing, and managing decision criteria. 
Dynamically adapts between Dual Hybrid (Categories & Criteria) and Single Flat Weighting modes.
"""

import streamlit as st
import pandas as pd
import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.factors_manager import (
    load_factors_config, 
    save_factors_config, 
    add_domain, 
    add_criterion, 
    delete_domain, 
    delete_criterion, 
    ensure_ghost_category, 
    remove_ghost_category
)

RATING_CONFIG_FILE = os.path.join(BASE_DIR, 'data', 'rating_config.json')

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
is_flat_mode = (weight_system_mode == "Single Flat Weighting (Direct Criteria Pool)")

# ==========================================
# AUTO-ADAPT GHOST CATEGORY ON PAGE LOAD
# ==========================================
if is_flat_mode:
    ensure_ghost_category()
else:
    remove_ghost_category()

st.set_page_config(page_title="Criteria Overview", page_icon="📋", layout="wide")
st.title("📋 Decision Criteria & Hierarchy")

if is_flat_mode:
    st.caption("Active Architecture: **Single Flat Weighting**. Criteria are managed in a unified flat pool.")
else:
    st.caption("Active Architecture: **Dual Hybrid**. Structure your decision model across high-level categories and criteria.")

# Load data after potential auto-adjustments
config = load_factors_config()
domains = [d for d in config.get("domains", []) if not (is_flat_mode and d["id"] == "d01")]
factors = config.get("factors", [])

if is_flat_mode:
    tab_overview, tab_manage, tab_edit, tab_danger = st.tabs([
        "👁️ Criteria Overview", 
        "➕ Add Criterion", 
        "✏️ Edit Criterion", 
        "⚠️ Danger Zone"
    ])
else:
    tab_overview, tab_manage, tab_edit, tab_danger = st.tabs([
        "👁️ Hierarchy Overview", 
        "➕ Add Category or Criterion", 
        "✏️ Edit Category or Criterion", 
        "⚠️ Danger Zone"
    ])

# ------------------------------------------
# TAB 1: VISUAL OVERVIEW
# ------------------------------------------
with tab_overview:
    if is_flat_mode:
        st.write("Unified flat list of all active decision criteria.")
        if not factors:
            st.info("No criteria exist yet. Go to 'Add Criterion' to build your decision model!")
        else:
            table_data = []
            for f in factors:
                table_data.append({
                    "ID": f["id"],
                    "Criterion": f["name"],
                    "Short Name": f.get("short_name", ""),
                    "Optimization": "Benefit (+1) [Higher is better]" if f.get("type", 1) == 1 else "Cost (-1) [Lower is better]",
                    "Description": f.get("description", "")
                })
            st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
    else:
        st.write("A complete overview of your hierarchical decision-making structure and categories.")
        if not domains:
            st.info("No categories exist yet. Go to 'Add Category or Criterion' to build your decision model!")
        else:
            for d in domains:
                d_factors = [f for f in factors if f["domain_id"] == d["id"]]
                with st.expander(f"📁 **{d['name']}** (`{d['id']}`) — {len(d_factors)} criteria", expanded=True):
                    st.caption(f"_{d.get('description', 'No description provided.')}_")
                    if d_factors:
                        table_data = []
                        for f in d_factors:
                            table_data.append({
                                "ID": f["id"],
                                "Criterion": f["name"],
                                "Short Name": f.get("short_name", ""),
                                "Optimization": "Benefit (+1) [Higher is better]" if f.get("type", 1) == 1 else "Cost (-1) [Lower is better]",
                                "Description": f.get("description", "")
                            })
                        st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
                    else:
                        st.warning("No criteria assigned to this category yet.")

# ------------------------------------------
# TAB 2: ADD (CATEGORY / CRITERION)
# ------------------------------------------
with tab_manage:
    if is_flat_mode:
        st.subheader("Add New Criterion")
        with st.form("add_criterion_flat_form", clear_on_submit=True):
            st.write("### 📄 New Criterion")
            raw_cr_name = st.text_input("Criterion Name", placeholder="e.g., Battery Life, Price, Speed")
            raw_cr_short = st.text_input("Short Name (Optional)", max_chars=30, placeholder="e.g., BAT")
            raw_cr_desc = st.text_area("Description", placeholder="Describe how this criterion is measured...")
            c_type = st.radio(
                "Optimization Direction:", 
                ["Benefit (+1) - Higher score is better", "Cost (-1) - Lower score is better"],
                help="Benefit criteria reward higher values. Cost criteria reward lower values."
            )
            
            if st.form_submit_button("Add Criterion", type="primary"):
                cr_name = (raw_cr_name or "").strip()
                if cr_name:
                    type_val = 1 if "Benefit" in c_type else -1
                    new_id = add_criterion("d01", cr_name, (raw_cr_desc or "").strip(), type_val, (raw_cr_short or "").strip())
                    st.success(f"Criterion '{cr_name}' ({new_id}) added successfully!")
                    st.rerun()
                else:
                    st.error("Criterion name is required.")
    else:
        st.subheader("Expand Your Decision Model")
        col_cat, col_crit = st.columns(2)
        
        with col_cat:
            with st.form("add_category_form", clear_on_submit=True):
                st.write("### 📁 New Category")
                raw_c_name = st.text_input("Category Name", placeholder="e.g., Performance, Cost, Usability")
                raw_c_short = st.text_input("Short Name (Optional)", max_chars=30, placeholder="e.g., PERF")
                raw_c_desc = st.text_area("Description", placeholder="Describe what this category evaluates...")
                
                if st.form_submit_button("Add Category", type="primary"):
                    c_name = (raw_c_name or "").strip()
                    if c_name:
                        new_id = add_domain(c_name, (raw_c_desc or "").strip(), (raw_c_short or "").strip())
                        st.success(f"Category '{c_name}' ({new_id}) added successfully!")
                        st.rerun()
                    else:
                        st.error("Category name is required.")
                        
        with col_crit:
            with st.form("add_criterion_form", clear_on_submit=True):
                st.write("### 📄 New Criterion")
                dom_map_inv = {d["name"]: d["id"] for d in domains}
                
                if dom_map_inv:
                    sel_cat = st.selectbox("Assign to Category:", list(dom_map_inv.keys()))
                    raw_cr_name = st.text_input("Criterion Name", placeholder="e.g., Battery Life, Price, Speed")
                    raw_cr_short = st.text_input("Short Name (Optional)", max_chars=30, placeholder="e.g., BAT")
                    raw_cr_desc = st.text_area("Description", placeholder="Describe how this criterion is measured...")
                    c_type = st.radio(
                        "Optimization Direction:", 
                        ["Benefit (+1) - Higher score is better", "Cost (-1) - Lower score is better"],
                        help="Benefit criteria reward higher values. Cost criteria reward lower values."
                    )
                    
                    if st.form_submit_button("Add Criterion", type="primary"):
                        cr_name = (raw_cr_name or "").strip()
                        if cr_name:
                            type_val = 1 if "Benefit" in c_type else -1
                            new_id = add_criterion(dom_map_inv[sel_cat], cr_name, (raw_cr_desc or "").strip(), type_val, (raw_cr_short or "").strip())
                            st.success(f"Criterion '{cr_name}' ({new_id}) added successfully!")
                            st.rerun()
                        else:
                            st.error("Criterion name is required.")
                else:
                    st.warning("You must create at least one Category before adding criteria.")

# ------------------------------------------
# TAB 3: EDIT
# ------------------------------------------
with tab_edit:
    if is_flat_mode:
        st.subheader("Edit Criterion")
        st.caption("Fix typos, update descriptions, change short names, or adjust optimization direction.")
        
        crit_edit_map = {f"[{f['id']}] {f['name']}": f for f in factors}
        if crit_edit_map:
            sel_edit_crit_label = st.selectbox("Select Criterion to Edit:", list(crit_edit_map.keys()), key="edit_flat_crit_sel")
            target_crit = crit_edit_map[sel_edit_crit_label]
            
            with st.form("edit_flat_criterion_form"):
                ed_cr_name = st.text_input("Criterion Name", value=target_crit.get("name") or "")
                ed_cr_short = st.text_input("Short Name", value=target_crit.get("short_name") or "", max_chars=30)
                ed_cr_desc = st.text_area("Description", value=target_crit.get("description") or "")
                
                current_type = target_crit.get("type", 1)
                type_options = ["Benefit (+1) - Higher score is better", "Cost (-1) - Lower score is better"]
                default_type_idx = 0 if current_type == 1 else 1
                
                ed_c_type = st.selectbox("Optimization Direction:", type_options, index=default_type_idx)
                
                if st.form_submit_button("💾 Save Criterion Changes", type="primary"):
                    cr_name = (ed_cr_name or "").strip()
                    if cr_name:
                        type_val = 1 if "Benefit" in ed_c_type else -1
                        for f in factors:
                            if f["id"] == target_crit["id"]:
                                f["name"] = cr_name
                                f["short_name"] = (ed_cr_short or "").strip()
                                f["description"] = (ed_cr_desc or "").strip()
                                f["type"] = type_val
                                break
                        save_factors_config(config)
                        st.success(f"Criterion '{cr_name}' updated successfully!")
                        st.rerun()
                    else:
                        st.error("Criterion name cannot be empty.")
        else:
            st.info("No criteria available to edit.")
    else:
        st.subheader("Modify Existing Hierarchy Elements")
        st.caption("Fix typos, update descriptions, change short names, or adjust optimization direction without losing data.")
        
        edit_cat_col, edit_crit_col = st.columns(2)
        
        with edit_cat_col:
            st.write("### 📁 Edit Category")
            dom_edit_map = {f"[{d['id']}] {d['name']}": d for d in domains}
            if dom_edit_map:
                sel_edit_dom_label = st.selectbox("Select Category to Edit:", list(dom_edit_map.keys()), key="edit_dom_sel")
                target_dom = dom_edit_map[sel_edit_dom_label]
                
                with st.form("edit_category_form"):
                    ed_c_name = st.text_input("Category Name", value=target_dom.get("name") or "")
                    ed_c_short = st.text_input("Short Name", value=target_dom.get("short_name") or "", max_chars=30)
                    ed_c_desc = st.text_area("Description", value=target_dom.get("description") or "")
                    
                    if st.form_submit_button("💾 Save Category Changes", type="primary"):
                        c_name = (ed_c_name or "").strip()
                        if c_name:
                            for d in domains:
                                if d["id"] == target_dom["id"]:
                                    d["name"] = c_name
                                    d["short_name"] = (ed_c_short or "").strip()
                                    d["description"] = (ed_c_desc or "").strip()
                                    break
                            save_factors_config(config)
                            st.success(f"Category '{c_name}' updated successfully!")
                            st.rerun()
                        else:
                            st.error("Category name cannot be empty.")
            else:
                st.info("No categories available to edit.")
                
        with edit_crit_col:
            st.write("### 📄 Edit Criterion")
            if dom_edit_map:
                sel_crit_dom_label = st.selectbox("1. Select Category:", list(dom_edit_map.keys()), key="edit_crit_dom_sel")
                target_dom_id = dom_edit_map[sel_crit_dom_label]["id"]
                
                cat_factors = [f for f in factors if f["domain_id"] == target_dom_id]
                crit_edit_map = {f"[{f['id']}] {f['name']}": f for f in cat_factors}
                
                if crit_edit_map:
                    sel_edit_crit_label = st.selectbox("2. Select Criterion to Edit:", list(crit_edit_map.keys()), key="edit_crit_sel")
                    target_crit = crit_edit_map[sel_edit_crit_label]
                    
                    with st.form("edit_criterion_form"):
                        ed_cr_name = st.text_input("Criterion Name", value=target_crit.get("name") or "")
                        ed_cr_short = st.text_input("Short Name", value=target_crit.get("short_name") or "", max_chars=30)
                        ed_cr_desc = st.text_area("Description", value=target_crit.get("description") or "")
                        
                        current_type = target_crit.get("type", 1)
                        type_options = ["Benefit (+1) - Higher score is better", "Cost (-1) - Lower score is better"]
                        default_type_idx = 0 if current_type == 1 else 1
                        
                        ed_c_type = st.selectbox("Optimization Direction:", type_options, index=default_type_idx)
                        
                        if st.form_submit_button("💾 Save Criterion Changes", type="primary"):
                            cr_name = (ed_cr_name or "").strip()
                            if cr_name:
                                type_val = 1 if "Benefit" in ed_c_type else -1
                                for f in factors:
                                    if f["id"] == target_crit["id"]:
                                        f["name"] = cr_name
                                        f["short_name"] = (ed_cr_short or "").strip()
                                        f["description"] = (ed_cr_desc or "").strip()
                                        f["type"] = type_val
                                        break
                                save_factors_config(config)
                                st.success(f"Criterion '{cr_name}' updated successfully!")
                                st.rerun()
                            else:
                                st.error("Criterion name cannot be empty.")
                else:
                    st.info("No criteria exist in this category to edit.")
            else:
                st.info("Please select a category first.")

# ------------------------------------------
# TAB 4: DANGER ZONE
# ------------------------------------------
with tab_danger:
    if is_flat_mode:
        st.subheader("Delete Criteria")
        st.error("Deleting a criterion will permanently remove it and its associated ratings. This action cannot be undone.")
        
        del_crit_map = {f"[{f['id']}] {f['name']}": f['id'] for f in factors}
        if del_crit_map:
            with st.form("delete_flat_criterion_form"):
                sel_del_crit = st.selectbox("Select Criterion to Delete:", list(del_crit_map.keys()), index=None, placeholder="Choose a criterion to delete...")
                if st.form_submit_button("🗑️ Delete Criterion", type="primary"):
                    if not sel_del_crit:
                        st.warning("⚠️ Please select a criterion first.")
                    else:
                        delete_criterion(del_crit_map[sel_del_crit])
                        st.success("Criterion deleted successfully!")
                        st.rerun()
        else:
            st.info("No criteria available to delete.")
    else:
        st.subheader("Manage & Delete Hierarchy Elements")
        st.error("Deleting a category or criterion will safely cascade ID re-sequencing and preserve your existing evaluations. This action cannot be undone.")
        
        del_cat, del_crit = st.columns(2)
        
        with del_cat:
            with st.form("delete_category_form"):
                st.write("**Delete Category**")
                del_dom_map = {f"[{d['id']}] {d['name']}": d['id'] for d in domains}
                if del_dom_map:
                    sel_del_dom = st.selectbox("Select Category to Delete:", list(del_dom_map.keys()), index=None, placeholder="Choose a category to delete...")
                    if st.form_submit_button("🗑️ Delete Category"):
                        if not sel_del_dom:
                            st.warning("⚠️ Please select a category first.")
                        else:
                            try:
                                delete_domain(del_dom_map[sel_del_dom])
                                st.success("Category deleted and IDs cascaded safely!")
                                st.rerun()
                            except ValueError as e:
                                st.error(str(e))
                else:
                    st.info("No categories available to delete.")
                    
        with del_crit:
            st.write("**Delete Criterion**")
            del_dom_map = {f"[{d['id']}] {d['name']}": d['id'] for d in domains}
            
            if del_dom_map:
                sel_del_dom_label = st.selectbox("1. Select Category:", list(del_dom_map.keys()), index=None, placeholder="Choose a category first...", key="del_crit_dom_sel")
                
                if sel_del_dom_label:
                    target_dom_id = del_dom_map[sel_del_dom_label]
                    cat_factors = [f for f in factors if f["domain_id"] == target_dom_id]
                    del_crit_map = {f"[{f['id']}] {f['name']}": f['id'] for f in cat_factors}
                    
                    if del_crit_map:
                        with st.form("delete_criterion_form"):
                            sel_del_crit = st.selectbox("2. Select Criterion to Delete:", list(del_crit_map.keys()), index=None, placeholder="Choose a criterion to delete...")
                            if st.form_submit_button("🗑️ Delete Criterion", type="primary"):
                                if not sel_del_crit:
                                    st.warning("⚠️ Please select a criterion first.")
                                else:
                                    delete_criterion(del_crit_map[sel_del_crit])
                                    st.success("Criterion deleted and IDs cascaded safely!")
                                    st.rerun()
                    else:
                        st.info("No criteria exist in this category.")
                else:
                    st.info("Please select a category above first.")
            else:
                st.info("No criteria available to delete.")