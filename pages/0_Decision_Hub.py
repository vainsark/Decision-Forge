"""
Decision Support System - Decision Hub (Project Workspace Manager)
Provides a dedicated UI page for creating, switching, duplicating, deleting,
and importing/exporting isolated project workspaces.
"""

import os
import shutil
import zipfile
import io
import json
from datetime import datetime
import streamlit as st

from src.project_manager import (
    list_projects,
    get_active_project_id,
    set_active_project_id,
    create_project,
    delete_project,
    duplicate_project,
    update_project_metadata,
    PROJECTS_DIR
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Decision Hub — MCDM Suite",
    page_icon="🗂️",
    layout="wide"
)

# ==========================================
# HELPER: PROJECT TELEMETRY SNAPSHOT
# ==========================================
def get_project_telemetry(proj_folder: str):
    domains_cnt = 0
    factors_cnt = 0
    alternatives_set = set()
    runs_cnt = 0
    analyses_cnt = 0

    f_path = os.path.join(proj_folder, 'factors_config.json')
    if os.path.exists(f_path):
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                f_data = json.load(f)
                domains_cnt = len(f_data.get("domains", []))
                factors_cnt = len(f_data.get("factors", []))
        except Exception:
            pass

    e_path = os.path.join(proj_folder, 'evaluations.json')
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

    r_dir = os.path.join(proj_folder, 'runs')
    if os.path.exists(r_dir):
        runs_cnt = len([f for f in os.listdir(r_dir) if f.endswith('.json')])

    a_dir = os.path.join(proj_folder, 'analysis_runs')
    if os.path.exists(a_dir):
        analyses_cnt = len([f for f in os.listdir(a_dir) if f.endswith('.json')])

    return domains_cnt, factors_cnt, sorted(list(alternatives_set)), runs_cnt, analyses_cnt

# ==========================================
# SESSION STATE & SIDEBAR WIDGET SYNC
# ==========================================
active_id = get_active_project_id()

st.sidebar.markdown("---")
st.sidebar.subheader("🗂️ Active Workspace")
projects = list_projects()
project_map = {p["name"]: p["project_id"] for p in projects}
inverse_project_map = {p["project_id"]: p["name"] for p in projects}

current_name = inverse_project_map.get(active_id, active_id)
selected_name = st.sidebar.selectbox(
    "Switch Project",
    options=list(project_map.keys()),
    index=list(project_map.keys()).index(current_name) if current_name in project_map else 0
)

if project_map.get(selected_name) != active_id:
    set_active_project_id(project_map[selected_name])
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Go to **Factors Overview** or **Weights Engine** next after switching projects.")

# ==========================================
# MAIN PAGE CONTENT: DECISION HUB
# ==========================================
st.title("🗂️ Decision Hub — Project Workspaces")
st.markdown(
    "Welcome to the **Multi-Project Workspace Manager**. "
    "Select an active decision project below, create a new analysis workspace, or manage existing project files."
)

st.markdown("---")

# ==========================================
# TABBED LAYOUT: WORKSPACES VS CREATE/IMPORT
# ==========================================
tab_list, tab_create, tab_import = st.tabs(["📂 Active Projects", "➕ Create New Project", "📦 Import Project (ZIP)"])

# ------------------------------------------
# TAB 1: LIST & MANAGE PROJECTS
# ------------------------------------------
with tab_list:
    st.subheader("Available Decision Projects")
    
    if not projects:
        st.warning("No projects found. Create one using the 'Create New Project' tab.")
    else:
        for p in projects:
            p_id = p["project_id"]
            is_active = (p_id == active_id)
            
            with st.container(border=True):
                col1, col2, col3 = st.columns([3, 2, 2])
                
                with col1:
                    if is_active:
                        st.markdown(f"### 🟢 **{p.get('name', p_id)}** `[Active Workspace]`")
                    else:
                        st.markdown(f"### **{p.get('name', p_id)}**")
                        
                    st.caption(p.get("description", "No description provided."))

                    # Project mini snapshot telemetry beneath description
                    proj_folder = os.path.join(PROJECTS_DIR, p_id)
                    domains_n, factors_n, raw_alts, runs_n, analyses_n = get_project_telemetry(proj_folder)
                    if raw_alts:
                        arena_str = " vs ".join(raw_alts)
                    else:
                        arena_str = "Alternative A vs Alternative B"
                    
                    st.markdown(
                        f"<div style='margin-top: 6px; font-size: 1rem; color: #A0AEC0;'>"
                        f"📊 <b>{domains_n}</b> Categories &nbsp;|&nbsp; "
                        f"📋 <b>{factors_n}</b> Factors &nbsp;|&nbsp; "
                        f"🎯 <b>{arena_str}</b> &nbsp;|&nbsp; "
                        f"📈 <b>{runs_n}</b> Runs &nbsp;|&nbsp; "
                        f"🔬 <b>{analyses_n}</b> Stress Tests"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.text(f"ID: {p_id} | Last Modified: {p.get('updated_at', 'N/A')[:16].replace('T', ' ')}")
                    
                with col2:
                    # Action Buttons
                    if not is_active:
                        if st.button("🚀 Switch to Project", key=f"switch_{p_id}", use_container_width=True):
                            set_active_project_id(p_id)
                            st.success(f"Switched to {p.get('name')}!")
                            st.rerun()
                            
                    # Export Project as ZIP
                    proj_folder = os.path.join(PROJECTS_DIR, p_id)
                    if os.path.exists(proj_folder):
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for root, _, files in os.walk(proj_folder):
                                for file in files:
                                    abs_path = os.path.join(root, file)
                                    rel_path = os.path.relpath(abs_path, proj_folder)
                                    zip_file.write(abs_path, rel_path)
                        zip_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Export Project (.zip)",
                            data=zip_buffer,
                            file_name=f"{p_id}_backup.zip",
                            mime="application/zip",
                            key=f"export_{p_id}",
                            use_container_width=True
                        )

                with col3:
                    # Edit Metadata Expandable
                    with st.expander("✏️ Edit & Duplicate"):
                        new_title = st.text_input("Title", value=p.get("name", ""), key=f"name_{p_id}")
                        new_desc = st.text_area("Description", value=p.get("description", ""), key=f"desc_{p_id}")
                        if st.button("Save Changes", key=f"save_{p_id}"):
                            update_project_metadata(p_id, str(new_title or ""), str(new_desc or ""))
                            st.success("Updated successfully!")
                            st.rerun()
                            
                        st.markdown("---")
                        dup_key = st.text_input("New ID for Clone", value=f"{p_id}_copy", key=f"dup_id_{p_id}")
                        dup_name = st.text_input("New Name for Clone", value=f"{p.get('name')} (Copy)", key=f"dup_name_{p_id}")
                        if st.button("Clone Project", key=f"clone_btn_{p_id}"):
                            try:
                                duplicate_project(p_id, dup_key, dup_name)
                                st.success("Project duplicated successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

                    # Delete Warning Dialog / Button
                    if not is_active:
                        with st.expander("🗑️ Danger Zone"):
                            confirm_del = st.checkbox("Confirm deletion", key=f"chk_del_{p_id}")
                            if st.button("Delete Project", key=f"del_{p_id}", type="primary"):
                                if confirm_del:
                                    delete_project(p_id)
                                    st.warning("Project deleted.")
                                    st.rerun()
                                else:
                                    st.error("Please check the confirmation box first.")
                    else:
                        st.info("ℹ️ Active project cannot be deleted. Switch away first.")

# ------------------------------------------
# TAB 2: CREATE NEW PROJECT
# ------------------------------------------
with tab_create:
    st.subheader("Create a New Decision Workspace")
    st.markdown("Spin up a pristine, isolated workspace with default configuration templates.")
    
    with st.form("create_project_form"):
        new_proj_id = st.text_input("Project Folder ID (lowercase, no spaces, e.g., cloud_migration)").strip()
        new_proj_name = st.text_input("Project Display Name (e.g., Enterprise Cloud Provider Selection)").strip()
        new_proj_desc = st.text_area("Description / Objective", placeholder="Evaluating AWS vs Azure vs GCP based on cost, security, and uptime.")
        
        submitted = st.form_submit_button("✨ Initialize Project Workspace")
        
        if submitted:
            if not new_proj_id or not new_proj_name:
                st.error("Project ID and Display Name are required.")
            else:
                try:
                    clean_id = create_project(new_proj_id, new_proj_name, new_proj_desc)
                    set_active_project_id(clean_id)
                    st.success(f"Project '{new_proj_name}' created and set as active workspace!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Creation failed: {e}")

# ------------------------------------------
# TAB 3: IMPORT PROJECT VIA ZIP
# ------------------------------------------
with tab_import:
    st.subheader("Import Project Archive")
    st.markdown("Upload a previously exported `.zip` project archive to restore it into your workspace collection.")
    
    uploaded_zip = st.file_uploader("Upload Project ZIP", type=["zip"])
    import_id_override = st.text_input("Override Project Folder ID (Optional)").strip()
    
    if uploaded_zip and st.button("📥 Import & Extract Workspace"):
        try:
            with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as zf:
                base_name = import_id_override or os.path.splitext(uploaded_zip.name)[0].replace("_backup", "")
                clean_id = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in base_name).lower()
                target_dir = os.path.join(PROJECTS_DIR, clean_id)
                
                if os.path.exists(target_dir):
                    clean_id = f"{clean_id}_{datetime.now().strftime('%H%M%S')}"
                    target_dir = os.path.join(PROJECTS_DIR, clean_id)
                    
                os.makedirs(target_dir, exist_ok=True)
                zf.extractall(target_dir)
                
                meta_file = os.path.join(target_dir, "project_meta.json")
                if not os.path.exists(meta_file):
                    meta = {
                        "name": clean_id.replace("_", " ").title(),
                        "description": "Imported via ZIP archive.",
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                    with open(meta_file, 'w', encoding='utf-8') as f:
                        json.dump(meta, f, indent=4)
                        
                set_active_project_id(clean_id)
                st.success(f"Successfully imported and activated project '{clean_id}'!")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to extract ZIP archive: {e}")