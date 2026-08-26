"""
Decision Support System - Hybrid Weight Engine
Uses PyMCDM native AHP and Normalization tools to handle weights.
Includes interactive CLI initializers, PyMCDM Float Sanitization, and a Consistency Finder.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Tuple, List, Any
from tabulate import tabulate

from pymcdm.weights.subjective import AHP
from pymcdm.normalizations import sum_normalization

# --- NEW PATH FIX: Ensure Python knows where the project root is ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.factors_manager import load_factors_config
from src.project_manager import get_active_project_dir

# ==========================================
# DYNAMIC FILE PATHS & PYMCDM CONSTANTS
# ==========================================
# Dynamic path resolution functions for active project workspace isolation
def _get_project_data_dir() -> str:
    """Returns the active project directory, asserting it is not None."""
    proj_dir = get_active_project_dir()
    assert proj_dir is not None, "Active project directory is required."
    return proj_dir

def get_ahp_filepath() -> str:
    """Returns the path to the AHP matrix file for the active project."""
    return os.path.join(_get_project_data_dir(), 'ahp_matrices.json')

def get_ratings_filepath() -> str:
    """Returns the path to the criteria ratings file for the active project."""
    return os.path.join(_get_project_data_dir(), 'criteria_ratings.json')

def get_weights_filepath() -> str:
    """Returns the path to the weights calculation bundle for the active project."""
    return os.path.join(_get_project_data_dir(), 'weights.json')

# PyMCDM strictly enforces these exact floats for fractions
# Create the exact mathematical floats PyMCDM demands (e.g. 0.3333333333333333 instead of 0.33)
EXACT_AHP_VALUES = [float(x) for x in range(1, 10)] + [1.0 / x for x in range(2, 10)]

def _ensure_data_dir():
    """Ensures that the active project directory exists on the file system."""
    proj_dir = _get_project_data_dir()
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)

def _sanitize_ahp_value(val: float) -> float:
    """Snaps any float to the exact PyMCDM mathematical AHP floats to prevent ValueError crashes."""
    return min(EXACT_AHP_VALUES, key=lambda x: abs(x - val))

def _parse_input_value(val_str: str) -> float:
    """Parses fractions or numbers and strictly sanitizes them for PyMCDM."""
    try:
        # Check if user entered a fraction (e.g., '1/3')
        if '/' in val_str:
            num, den = val_str.split('/')
            val = float(num) / float(den)
        else:
            val = float(val_str)
        # Snap to exact allowed AHP float values
        return _sanitize_ahp_value(val)
    except ValueError:
        raise ValueError("Invalid format. Enter a number (e.g., 5) or fraction (e.g., 1/3).")

# ==========================================
# INTERACTIVE INITIALIZATION & LOADING
# ==========================================
def load_ahp_matrix(domains: List[Dict]) -> np.ndarray:
    """Loads AHP matrix or triggers interactive creation using PyMCDM structures."""
    n = len(domains)
    _ensure_data_dir()
    ahp_file = get_ahp_filepath()
    
    # Check if the active project already has a saved AHP matrix file
    if os.path.exists(ahp_file):
        with open(ahp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "domain_matrix" in data:
                mat = np.array(data["domain_matrix"])
                v_sanitize = np.vectorize(_sanitize_ahp_value)
                mat = v_sanitize(mat)
                if mat.shape == (n, n):
                    return mat
                    
    # --- INTERACTIVE INITIAL CREATION WITH PYMCDM ---
    print("\n\033[96m[!] No existing AHP Matrix found. Starting PyMCDM Interactive Setup.\033[0m")
    domain_names = [d['short_name'] for d in domains]
    
    # Passing object_names tells PyMCDM to launch the interactive prompt
    ahp_model = AHP(object_names=domain_names)
    _ = ahp_model()  # This triggers the interactive CLI questions
    
    # Retrieve the generated matrix from PyMCDM safely
    mat = getattr(ahp_model, 'matrix', getattr(ahp_model, '_matrix', None))
    if mat is None:
        # Failsafe dump to CSV if attributes are locked
        proj_dir = _get_project_data_dir()
        temp_csv = os.path.join(proj_dir, 'temp_ahp.csv')
        ahp_model.to_csv(temp_csv)
        mat = np.loadtxt(temp_csv, delimiter=',')
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            
    # Persist the newly generated matrix to the active project folder
    with open(ahp_file, 'w', encoding='utf-8') as f:
         json.dump({"domain_matrix": mat.tolist()}, f, indent=4)
         
    return mat

def load_criteria_ratings(config: Dict[str, Any]) -> Dict[str, float]:
    """Loads 1-10 criteria ratings. Injects defaults for missing new factors or triggers interactive creation."""
    _ensure_data_dir()
    factors = config.get("factors", [])
    expected_fids = [f["id"] for f in factors]
    ratings_file = get_ratings_filepath()
    
    # Check if criteria ratings file exists for this project
    if os.path.exists(ratings_file):
        with open(ratings_file, 'r', encoding='utf-8') as f:
            ratings = json.load(f)
            
        # Check if any new factors were added to config but are missing from saved ratings
        missing_keys = [fid for fid in expected_fids if fid not in ratings]
        if missing_keys:
            for fid in missing_keys:
                ratings[fid] = 5.0  # Assign a safe default rating of 5.0 for new factors
            # Resave the repaired file
            with open(ratings_file, 'w', encoding='utf-8') as f:
                json.dump(ratings, f, indent=4)
                
        return ratings
            
    # --- INTERACTIVE INITIAL CREATION ---
    print("\n\033[96m[!] No existing criteria ratings found. Starting Interactive Setup.\033[0m")
    ratings = {}
    print("\n\033[94m" + "="*50)
    print(" INITIAL CRITERIA RATINGS SETUP (1-10)")
    print("="*50 + "\033[0m")
    
    # Loop through each domain and prompt user for initial criteria ratings
    for domain in config.get("domains", []):
        print(f"\n--- {domain['name']} ---")
        domain_factors = [f for f in factors if f["domain_id"] == domain["id"]]
        for f in domain_factors:
            print(f"\nCriterion: \033[93m{f['name']} ({f['short_name']})\033[0m")
            print(f"Description: {f['description']}")
            while True:
                val_str = input("Enter rating (1.0 - 10.0): ").strip()
                try:
                    val = float(val_str)
                    if 1.0 <= val <= 10.0:
                        ratings[f["id"]] = val
                        break
                    else:
                        print("Rating must be between 1.0 and 10.0.")
                except ValueError:
                    print("Invalid number format.")
                    
    # Save initialized ratings to active project storage
    with open(ratings_file, 'w', encoding='utf-8') as f:
        json.dump(ratings, f, indent=4)
        
    return ratings

def save_state(matrix: np.ndarray, ratings: Dict[str, float], weights: Dict[str, Any]):
    """Persists the AHP matrix, criteria ratings, and weight bundles to project files."""
    _ensure_data_dir()
    with open(get_ahp_filepath(), 'w', encoding='utf-8') as f:
        json.dump({"domain_matrix": matrix.tolist()}, f, indent=4)
    with open(get_ratings_filepath(), 'w', encoding='utf-8') as f:
        json.dump(ratings, f, indent=4)
    with open(get_weights_filepath(), 'w', encoding='utf-8') as f:
        json.dump(weights, f, indent=4)

# ==========================================
# PYMCDM MATHEMATICAL ENGINE
# ==========================================
def calculate_ahp_weights(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """Uses pymcdm's AHP module to get weights and Consistency Ratio."""
    # Step 1: Snap to PyMCDM's exact allowed floats
    v_sanitize = np.vectorize(_sanitize_ahp_value)
    safe_matrix = v_sanitize(matrix)
    
    # Step 2: Mathematically force perfect reciprocals so PyMCDM's validator doesn't crash
    n = safe_matrix.shape[0]
    for i in range(n):
        safe_matrix[i, i] = 1.0  # Diagonal must be exactly 1.0
        for j in range(i + 1, n):
            safe_matrix[j, i] = 1.0 / safe_matrix[i, j]
            
    # Step 3: Pass perfectly sanitized matrix to PyMCDM
    ahp_model = AHP(matrix=safe_matrix)
    weights = ahp_model()
    try:
        cr = ahp_model.get_cr()
    except ValueError:
        cr = 0.0  # Fallback just in case N <= 2
    return weights, float(cr)

def get_worst_inconsistency(matrix: np.ndarray, weights: np.ndarray) -> Tuple[int, int, float, float]:
    """Manual function to find the exact cell that contradicts the final weights the most."""
    n = len(weights)
    max_dev = 0.0
    worst_i, worst_j = 0, 1
    actual_val, expected_ratio = 1.0, 1.0
    
    # Compare every pair in the matrix to find highest deviation from expected ratio
    for i in range(n):
        for j in range(i+1, n):
            expected = weights[i] / weights[j]
            actual = matrix[i, j]
            dev = max(expected/actual, actual/expected)
            if dev > max_dev:
                max_dev = dev
                worst_i, worst_j = i, j
                actual_val, expected_ratio = actual, expected
                
    return worst_i, worst_j, actual_val, expected_ratio

def calculate_local_weights(ratings: Dict[str, float], config: Dict[str, Any]) -> Dict[str, float]:
    """Uses pymcdm.normalizations to convert 1-10 ratings to local weights."""
    local_weights = {}
    
    for domain in config.get("domains", []):
        domain_id = domain["id"]
        domain_factors = [f["id"] for f in config.get("factors", []) if f["domain_id"] == domain_id]
        
        if not domain_factors:
            continue
            
        ratings_list = [ratings[fid] for fid in domain_factors]
        ratings_array = np.array(ratings_list, dtype=float)
        
        # Normalize ratings using PyMCDM sum normalization
        norm_array = sum_normalization(ratings_array, cost=False)
        
        for idx, fid in enumerate(domain_factors):
            local_weights[fid] = float(norm_array[idx])
            
    return local_weights

def calculate_global_weights(cat_weights: Dict[str, float], local_weights: Dict[str, float], config: Dict[str, Any]) -> Dict[str, float]:
    """Multiplies category weights by local weights to yield global criteria weights."""
    global_weights = {}
    for factor in config.get("factors", []):
        fid = factor["id"]
        did = factor["domain_id"]
        global_weights[fid] = cat_weights.get(did, 0.0) * local_weights.get(fid, 0.0)
    return global_weights

# ==========================================
# PUBLIC API FOR MAIN.PY / UI
# ==========================================
def load_or_initialize_weights() -> Dict[str, Any]:
    """Orchestrates loading factors, AHP matrix, and ratings, then computes final weights bundle."""
    config = load_factors_config()
    domains = config.get("domains", [])
    
    matrix = load_ahp_matrix(domains)
    ratings = load_criteria_ratings(config)
    
    cat_w_array, cr = calculate_ahp_weights(matrix)
    cat_weights = {domains[i]["id"]: float(cat_w_array[i]) for i in range(len(domains))}
    loc_weights = calculate_local_weights(ratings, config)
    glob_weights = calculate_global_weights(cat_weights, loc_weights, config)
    
    weights_bundle = {
        "category_weights": cat_weights,
        "local_weights": loc_weights,
        "global_weights": glob_weights,
        "cr_status": cr
    }
    
    save_state(matrix, ratings, weights_bundle)
    return weights_bundle

def get_global_weights() -> Dict[str, float]:
    """Convenience wrapper to quickly retrieve global weights dictionary."""
    weights = load_or_initialize_weights()
    return weights["global_weights"]

# ==========================================
# INTERACTIVE CLI MENUS
# ==========================================
def view_weights_cli(config: Dict[str, Any], weights: Dict[str, Any]):
    """CLI interactive screen for viewing weights with visual charts and sorting."""
    sort_mode = "default"
    
    while True:
        print("\n\033[94m--- CATEGORY WEIGHTS (W_cat) ---\033[0m")
        
        # --- NEW VISUAL BAR CHART ---
        cat_list = []
        for d in config["domains"]:
            w = weights['category_weights'].get(d['id'], 0.0)
            cat_list.append((d['name'], w * 100))
        
        cat_list.sort(key=lambda x: x[1], reverse=True)
        
        if cat_list:
            max_name_len = max(len(name) for name, _ in cat_list)
            BAR_WIDTH = 30
            for name, pct in cat_list:
                blocks = int((pct / 100.0) * BAR_WIDTH)
                bar = "█" * blocks
                print(f" {name:<{max_name_len}}   \033[92m{bar:<{BAR_WIDTH}}\033[0m  {pct:>4.1f}%")
        
        # 1. Build the base data list for the criteria table
        c_data = []
        for f in config["factors"]:
            lw = weights['local_weights'].get(f['id'], 0.0)
            gw = weights['global_weights'].get(f['id'], 0.0)
            c_data.append({
                "domain_id": f["domain_id"],
                "domain_name": f["domain_short_name"],
                "criterion": f["short_name"],
                "lw": lw,
                "gw": gw
            })
            
        # 2. Apply sorting based on user state
        if sort_mode == "global":
            c_data.sort(key=lambda x: x["gw"], reverse=True)
            sort_title = "Sorted by: Global %"
        elif sort_mode == "domain":
            domain_order = {d["id"]: i for i, d in enumerate(config["domains"])}
            c_data.sort(key=lambda x: domain_order.get(x["domain_id"], 999))
            sort_title = "Sorted by: Domain"
        elif sort_mode == "domain_local":
            domain_order = {d["id"]: i for i, d in enumerate(config["domains"])}
            c_data.sort(key=lambda x: (domain_order.get(x["domain_id"], 999), -x["lw"]))
            sort_title = "Sorted by: Domain & Local %"
        else:
            sort_title = "Default Config Order"
            
        # 3. Format the table and merge Domain cells
        crit_table = []
        prev_domain = None
        domain_jump_indices = []
        
        for i, row in enumerate(c_data):
            if sort_mode in ["domain", "domain_local", "default"]:
                if row["domain_id"] != prev_domain:
                    display_domain = row["domain_name"]
                    if i > 0:
                        domain_jump_indices.append(i) 
                else:
                    display_domain = ""
            else:
                display_domain = row["domain_name"]
                
            prev_domain = row["domain_id"]
            
            crit_table.append([
                display_domain,
                row["criterion"],
                f"{row['lw']*100:.1f}%",
                f"{row['gw']*100:.1f}%"
            ])
            
        print(f"\n\033[94m--- CRITERIA WEIGHTS ({sort_title}) ---\033[0m")
        
        # 4. Generate the raw string table, inject Yellow borders, and print
        raw_table = tabulate(crit_table, headers=["Domain", "Criterion", "Local %", "Global %"], tablefmt="grid")
        
        if sort_mode in ["domain", "domain_local", "default"] and domain_jump_indices:
            lines = raw_table.split('\n')
            for idx in domain_jump_indices:
                border_line_index = 2 + (idx * 2)
                lines[border_line_index] = f"\033[93m{lines[border_line_index]}\033[0m"
            print('\n'.join(lines))
        else:
            print(raw_table)
        
        # 5. Interactive sorting menu options
        print("\nOptions:")
        print(" [1] Sort by Global % (Descending)")
        print(" [2] Sort by Domain (Alphabetical/Grouped)")
        print(" [3] Sort by Domain & Local % (Descending within domains)")
        print(" [b] Back to Weights Menu")
        
        choice = input("\nEnter command: ").strip().lower()
        if choice == 'b':
            break
        elif choice == '1':
            sort_mode = "global"
        elif choice == '2':
            sort_mode = "domain"
        elif choice == '3':
            sort_mode = "domain_local"
        else:
            print("\033[93mInvalid choice. Please select 1, 2, 3, or 'b'.\033[0m")

def edit_ahp_matrix_cli(config: Dict[str, Any]):
    """CLI interactive screen for viewing and modifying pairwise comparison matrices."""
    domains = config["domains"]
    n = len(domains)
    matrix = load_ahp_matrix(domains)
    
    while True:
        cat_w_array, cr = calculate_ahp_weights(matrix)
        
        print("\n\033[94m" + "="*50)
        print(" DOMAIN AHP PAIRWISE MATRIX")
        print("="*50 + "\033[0m")
        
        headers = ["#", "Domain"] + [f"{i+1}. {d['short_name']}" for i, d in enumerate(domains)]
        table = [[i+1, domains[i]['short_name']] + [f"{val:.2f}" for val in row] for i, row in enumerate(matrix)]
        print(tabulate(table, headers=headers, tablefmt="grid"))
        
        print(f"\nCurrent Consistency Ratio (CR): {cr:.4f}")
        if cr > 0.10:
            print(f"\033[91mWARNING: Consistency Ratio CR = {cr:.4f} exceeds threshold (0.10).\033[0m")
            i, j, actual, expected = get_worst_inconsistency(matrix, cat_w_array)
            print("\n\033[93m--- CONTRADICTION FINDER ---\033[0m")
            print(f"Your biggest inconsistency is between Row {i+1} ({domains[i]['short_name']}) and Col {j+1} ({domains[j]['short_name']}).")
            print(f"You inputted: {actual:.2f}. But to match your overall weights, it should be closer to: {_sanitize_ahp_value(expected):.2f}")
            print("\033[93m----------------------------\033[0m")
        
        print("\nOptions:")
        print(" [row col val] - Edit a cell (e.g., '1 2 3')")
        print(" [i]           - Initialize Matrix (PyMCDM Interactive AHP)")
        print(" [r]           - Reset matrix to equal weights")
        print(" [b]           - Back to Weights Menu")
        
        choice = input("Enter command: ").strip().lower()
        if choice == 'b':
            break
        if choice == 'r':
            matrix = np.eye(n)
            continue
            
        if choice == 'i':
            print("\nStarting PyMCDM Interactive AHP Setup...")
            domain_names = [d['short_name'] for d in domains]
            ahp_model = AHP(object_names=domain_names)
            try:
                _ = ahp_model()  # Triggers PyMCDM CLI
                
                # Recover generated matrix
                mat = getattr(ahp_model, 'matrix', getattr(ahp_model, '_matrix', None))
                if mat is None:
                    proj_dir = _get_project_data_dir()
                    temp_csv = os.path.join(proj_dir, 'temp_ahp.csv')
                    ahp_model.to_csv(temp_csv)
                    mat = np.loadtxt(temp_csv, delimiter=',')
                    if os.path.exists(temp_csv):
                        os.remove(temp_csv)
                        
                matrix = mat
                
                # Force full pipeline recalculation & save the new data
                cat_weights = {domains[idx]["id"]: float(calculate_ahp_weights(matrix)[0][idx]) for idx in range(n)}
                ratings = load_criteria_ratings(config)
                loc_weights = calculate_local_weights(ratings, config)
                glob_weights = calculate_global_weights(cat_weights, loc_weights, config)
                
                save_state(matrix, ratings, {"category_weights": cat_weights, "local_weights": loc_weights, "global_weights": glob_weights})
                print("\033[92mMatrix initialized and updated successfully.\033[0m")
            except Exception as e:
                print(f"\033[91mInteractive setup failed or cancelled: {e}\033[0m")
            continue
            
        parts = choice.split()
        if len(parts) == 3:
            try:
                row_idx, col_idx, val = int(parts[0]) - 1, int(parts[1]) - 1, _parse_input_value(parts[2])
                if 0 <= row_idx < n and 0 <= col_idx < n:
                    if row_idx == col_idx:
                        print("\033[93mCannot change diagonal values. Must remain 1.0\033[0m")
                        continue
                    matrix[row_idx, col_idx] = val
                    matrix[col_idx, row_idx] = _sanitize_ahp_value(1.0 / val)
                    
                    # Complete a full recalculation and save
                    cat_weights = {domains[idx]["id"]: float(calculate_ahp_weights(matrix)[0][idx]) for idx in range(n)}
                    ratings = load_criteria_ratings(config)
                    loc_weights = calculate_local_weights(ratings, config)
                    glob_weights = calculate_global_weights(cat_weights, loc_weights, config)
                    
                    save_state(matrix, ratings, {"category_weights": cat_weights, "local_weights": loc_weights, "global_weights": glob_weights})
                    print("\033[92mMatrix updated successfully.\033[0m")
                else:
                    print("Invalid row/col numbers.")
            except ValueError as e:
                print(f"Error parsing input: {e}")
        else:
            print("Invalid format. Please use 'row col value' (e.g., '1 2 5').")

def edit_ratings_cli(config: Dict[str, Any]):
    """CLI interactive screen for managing local criteria ratings."""
    import textwrap
    factors = config["factors"]
    domains = config["domains"]
    ratings = load_criteria_ratings(config)
    
    while True:
        print("\n\033[94m" + "="*50)
        print(" LOCAL CRITERIA RATINGS (CATEGORIES)")
        print("="*50 + "\033[0m")
        
        # 1. Display Categories
        for i, d in enumerate(domains):
            print(f" {i + 1}. {d['name']} ({d['short_name']})")
            
        print("="*50)
        print("Options:")
        print(" [1-X] - Select a category to view/edit its ratings in a table")
        print(" [i]   - Initialize sequentially (Interactive Q&A for a category)")
        print(" [b]   - Back to Weights Menu")
        
        choice = input("\nEnter command: ").strip().lower()
        if choice == 'b':
            break
            
        # --- NEW INITIALIZER FLOW ---
        if choice == 'i':
            cat_choice = input(f"Enter the number of the category to initialize [1-{len(domains)}]: ").strip()
            try:
                domain_idx = int(cat_choice) - 1
                if 0 <= domain_idx < len(domains):
                    selected_domain = domains[domain_idx]
                    domain_factors = [f for f in factors if f["domain_id"] == selected_domain["id"]]
                    
                    print("\n\033[94m" + "="*70)
                    print(f" INITIALIZING RATINGS FOR: {selected_domain['name'].upper()}")
                    print("="*70 + "\033[0m")
                    
                    for f in domain_factors:
                        print(f"\nHow important is \033[93m{f['name']}\033[0m?")
                        
                        # Wrap description for clean terminal reading
                        wrapped_desc = textwrap.fill(f['description'], width=65)
                        print(f"({wrapped_desc})")
                        
                        while True:
                            val_str = input(f"Enter rating (1.0 - 10.0) [Current: {ratings[f['id']]:.1f}]: ").strip()
                            
                            # Allow user to just press Enter to keep the current rating
                            if val_str == "":
                                break
                                
                            try:
                                val = float(val_str)
                                if 1.0 <= val <= 10.0:
                                    ratings[f["id"]] = val
                                    break
                                else:
                                    print("\033[93mRating must be between 1.0 and 10.0.\033[0m")
                            except ValueError:
                                print("\033[93mInvalid number format.\033[0m")
                    
                    # Full pipeline recalculation & save after the loop finishes
                    matrix = load_ahp_matrix(domains)
                    cat_w_array, _ = calculate_ahp_weights(matrix)
                    cat_weights = {domains[i]["id"]: float(cat_w_array[i]) for i in range(len(domains))}
                    loc_weights = calculate_local_weights(ratings, config)
                    glob_weights = calculate_global_weights(cat_weights, loc_weights, config)
                    
                    save_state(matrix, ratings, {"category_weights": cat_weights, "local_weights": loc_weights, "global_weights": glob_weights})
                    print(f"\n\033[92mAll ratings for '{selected_domain['short_name']}' updated and saved successfully!\033[0m")
                else:
                    print("\033[93mInvalid category number.\033[0m")
            except ValueError:
                print("\033[93mInvalid input.\033[0m")
            continue
            
        # --- ORIGINAL SUB-MENU FLOW ---
        try:
            domain_idx = int(choice) - 1
            if 0 <= domain_idx < len(domains):
                selected_domain = domains[domain_idx]
                
                while True:
                    print("\n\033[94m" + "="*70)
                    print(f" RATINGS FOR: {selected_domain['name'].upper()}")
                    print("="*70 + "\033[0m")
                    
                    domain_factors = [f for f in factors if f["domain_id"] == selected_domain["id"]]
                    table = []
                    
                    for f in domain_factors:
                        full_wrapped_desc = textwrap.fill(f['description'], width=55)
                        table.append([f['id'], f['short_name'], f"{ratings[f['id']]:.1f}", full_wrapped_desc])
                        
                    print(tabulate(table, headers=["ID", "Criterion (Short)", "Rating", "Description"], tablefmt="grid"))
                    
                    print("\nOptions:")
                    print(" [id val] - Edit rating (e.g., 'c01 8.5')")
                    print(" [b]      - Back to Categories list")
                    
                    sub_choice = input("Enter command: ").strip().lower()
                    if sub_choice == 'b':
                        break
                        
                    parts = sub_choice.split()
                    if len(parts) == 2:
                        fid, val_str = parts[0], parts[1]
                        
                        valid_fids = [f['id'] for f in domain_factors]
                        if fid in valid_fids:
                            try:
                                val = float(val_str)
                                if 1.0 <= val <= 10.0:
                                    ratings[fid] = val
                                    
                                    matrix = load_ahp_matrix(domains)
                                    cat_w_array, _ = calculate_ahp_weights(matrix)
                                    cat_weights = {domains[i]["id"]: float(cat_w_array[i]) for i in range(len(domains))}
                                    loc_weights = calculate_local_weights(ratings, config)
                                    glob_weights = calculate_global_weights(cat_weights, loc_weights, config)
                                    
                                    save_state(matrix, ratings, {"category_weights": cat_weights, "local_weights": loc_weights, "global_weights": glob_weights})
                                    print(f"\033[92mRating for {fid} successfully updated to {val}.\033[0m")
                                else:
                                    print("\033[93mRating must be between 1.0 and 10.0.\033[0m")
                            except ValueError:
                                print("\033[93mInvalid number format.\033[0m")
                        else:
                            print("\033[93mInvalid ID. Please select an ID from the table above.\033[0m")
                    else:
                        print("\033[93mInvalid format. Use 'ID Value' (e.g., 'c01 9').\033[0m")
            else:
                print("\033[93mInvalid category number.\033[0m")
        except ValueError:
            print("\033[93mInvalid input. Please enter a number or 'b' or 'i'.\033[0m")

def run_weights_cli():
    """Main CLI menu handler for the Hybrid Weight Engine."""
    try:
        config = load_factors_config()
    except FileNotFoundError as e:
        print(f"\033[91m{e}\033[0m")
        return
        
    while True:
        weights = load_or_initialize_weights()
        print("\n\033[94m" + "="*50)
        print("  HYBRID WEIGHT ENGINE (AHP DOMAINS + 1-10 RATINGS)")
        print("="*50 + "\033[0m")
        print("1. View Current Weights (Category, Local, Global)")
        print("2. Manage Domain AHP Matrix (View / Edit / Fix CR)")
        print("3. Manage Criteria 1-10 Ratings (View / Edit Ratings)")
        print("4. Force Recalculate & Export")
        print("5. Return to Main Menu")
        print("="*50)
        
        choice = input("Select an option [1-5]: ").strip()
        
        if choice == '1':
            view_weights_cli(config, weights)
        elif choice == '2':
            edit_ahp_matrix_cli(config)
        elif choice == '3':
            edit_ratings_cli(config)
        elif choice == '4':
            load_or_initialize_weights()
            print("\n\033[92mWeights successfully recalculated and saved!\033[0m")
        elif choice == '5':
            print("Returning to main menu...")
            break
        else:
            print("\033[93mInvalid choice. Please select 1-5.\033[0m")