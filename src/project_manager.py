"""
Decision Support System - Project Manager Backend
Handles multi-project isolation, directory scanning, metadata tracking, and path routing.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
import streamlit as st

# ==========================================
# PATH CONSTANTS
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROJECTS_DIR = os.path.join(DATA_DIR, 'projects')

def _ensure_dirs():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(PROJECTS_DIR):
        os.makedirs(PROJECTS_DIR)

# ==========================================
# PROJECT DISCOVERY & METADATA
# ==========================================
def list_projects() -> List[Dict[str, Any]]:
    """
    Scans data/projects/ and returns a list of project metadata dictionaries.
    If no projects exist, automatically ensures at least one default exists.
    """
    _ensure_dirs()
    projects = []
    
    if not os.path.exists(PROJECTS_DIR):
        return projects
        
    for entry in os.listdir(PROJECTS_DIR):
        proj_path = os.path.join(PROJECTS_DIR, entry)
        if os.path.isdir(proj_path):
            meta_file = os.path.join(proj_path, "project_meta.json")
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        meta["project_id"] = entry
                        projects.append(meta)
                except Exception:
                    # Fallback if meta file is corrupted
                    projects.append({
                        "project_id": entry,
                        "name": entry,
                        "description": "Legacy or uninitialized project folder.",
                        "created_at": datetime.now().isoformat()
                    })
            else:
                # Auto-generate meta file if missing inside folder
                meta = {
                    "project_id": entry,
                    "name": entry.replace("_", " ").title(),
                    "description": "Imported or manually created project folder.",
                    "created_at": datetime.now().isoformat()
                }
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=4)
                projects.append(meta)
                
    # Sort projects alphabetically or by creation time
    projects.sort(key=lambda x: x.get("name", ""))
    return projects

def get_active_project_id() -> Optional[str]:
    """
    Retrieves the currently active project ID from Streamlit session state.
    Returns None if no project has been explicitly opened or selected yet.
    """
    if "active_project_id" not in st.session_state:
        st.session_state["active_project_id"] = None
            
    return st.session_state["active_project_id"]

def set_active_project_id(project_id: str):
    """Sets the active project ID in Streamlit session state."""
    st.session_state["active_project_id"] = project_id

def get_active_project_dir() -> Optional[str]:
    """Returns the absolute path to the currently active project directory, or None if unselected."""
    proj_id = get_active_project_id()
    if not proj_id:
        return None
    proj_dir = os.path.join(PROJECTS_DIR, proj_id)
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)
    return proj_dir

# ==========================================
# PROJECT LIFECYCLE OPERATIONS
# ==========================================
def create_project(project_id: str, name: str, description: str) -> str:
    """
    Creates a new project folder with subdirectories and default JSON templates.
    """
    _ensure_dirs()
    # Sanitize project_id
    clean_id = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in project_id).lower()
    proj_dir = os.path.join(PROJECTS_DIR, clean_id)
    
    if os.path.exists(proj_dir):
        raise ValueError(f"Project ID '{clean_id}' already exists. Choose a unique name.")
        
    os.makedirs(proj_dir)
    os.makedirs(os.path.join(proj_dir, "runs"))
    os.makedirs(os.path.join(proj_dir, "analysis_runs"))
    os.makedirs(os.path.join(proj_dir, "weight_presets"))
    os.makedirs(os.path.join(proj_dir, "evaluation_presets"))
    
    # Create project metadata
    meta = {
        "name": name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    with open(os.path.join(proj_dir, "project_meta.json"), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=4)
        
    # Initialize default configuration files inside the project folder
    default_rating_config = {
        "coefficients": {"Kv": 0.5, "Ke": 0.5, "Kb": 1.0},
        "defuzz_weights": [0.15, 0.35, 0.35, 0.15],
        "promethee_q": 0.5,
        "promethee_p": 3.5,
        "weight_init_mode": "Direct Weight Sliders",
        "alternatives": ["Alternative 1", "Alternative 2"],
        "waspas_lambda": 0.5,
        "normalization_mode": "default",
        "normalization_ceiling": 10.0,
        "promethee_pref_func": "vshape_2"
    }
    with open(os.path.join(proj_dir, "rating_config.json"), 'w', encoding='utf-8') as f:
        json.dump(default_rating_config, f, indent=4)
        
    # Create empty baseline files so loaders don't crash
    with open(os.path.join(proj_dir, "factors_config.json"), 'w', encoding='utf-8') as f:
        json.dump({"domains": [], "factors": []}, f, indent=4)
        
    with open(os.path.join(proj_dir, "weights.json"), 'w', encoding='utf-8') as f:
        json.dump({"global_weights": {}, "category_weights": {}}, f, indent=4)
        
    with open(os.path.join(proj_dir, "evaluations.json"), 'w', encoding='utf-8') as f:
        json.dump([], f, indent=4)
        
    return clean_id

def update_project_metadata(project_id: str, new_name: str, new_description: str):
    """
    Updates the display name and description of an existing project workspace.
    """
    proj_dir = os.path.join(PROJECTS_DIR, project_id)
    meta_file = os.path.join(proj_dir, "project_meta.json")
    
    if not os.path.exists(proj_dir):
        raise ValueError(f"Project folder '{project_id}' does not exist.")
        
    meta = {}
    if os.path.exists(meta_file):
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
    meta["name"] = new_name
    meta["description"] = new_description
    meta["updated_at"] = datetime.now().isoformat()
    
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=4)

def delete_project(project_id: str):
    """Safely deletes an entire project folder and its contents."""
    proj_dir = os.path.join(PROJECTS_DIR, project_id)
    if os.path.exists(proj_dir):
        shutil.rmtree(proj_dir)
        
    # Reset active project to None if the active project was deleted
    if get_active_project_id() == project_id:
        st.session_state["active_project_id"] = None

def duplicate_project(source_id: str, new_id: str, new_name: str) -> str:
    """Clones an existing project folder into a new project workspace."""
    source_dir = os.path.join(PROJECTS_DIR, source_id)
    clean_new_id = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in new_id).lower()
    target_dir = os.path.join(PROJECTS_DIR, clean_new_id)
    
    if not os.path.exists(source_dir):
        raise ValueError(f"Source project '{source_id}' does not exist.")
    if os.path.exists(target_dir):
        raise ValueError(f"Project ID '{clean_new_id}' already exists.")
        
    # Copy directory tree
    shutil.copytree(source_dir, target_dir)
    
    # Update metadata file name
    meta_file = os.path.join(target_dir, "project_meta.json")
    if os.path.exists(meta_file):
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        meta["name"] = new_name
        meta["created_at"] = datetime.now().isoformat()
        meta["updated_at"] = datetime.now().isoformat()
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=4)
            
    return clean_new_id