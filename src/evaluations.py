"""
Decision Support System - Evaluations & Rating Data Layer
Handles collecting, validating, calculating fuzzy trapezoids, and persisting joint evaluations.
"""

import os
import json
import textwrap
from typing import Dict, List, Any, Tuple
from tabulate import tabulate

from src.factors_manager import load_factors_config
from src.project_manager import get_active_project_dir

# ==========================================
# FILE PATHS & CONSTANTS
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dynamic path resolution functions for active project workspace isolation
def _get_project_data_dir() -> str:
    """Returns the active project directory, asserting it is not None."""
    proj_dir = get_active_project_dir()
    assert proj_dir is not None, "Active project directory is required."
    return proj_dir

def get_rating_config_filepath() -> str:
    return os.path.join(_get_project_data_dir(), 'rating_config.json')

def get_evaluations_filepath() -> str:
    return os.path.join(_get_project_data_dir(), 'evaluations.json')

# ANSI Colors for CLI
C_RED = '\033[91m'
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_BLUE = '\033[94m'
C_BOLD = '\033[1m'
C_RESET = '\033[0m'

# ==========================================
# CONFIGURATION MANAGEMENT
# ==========================================
def _ensure_data_dir():
    proj_dir = _get_project_data_dir()
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)

def load_rating_config() -> Dict[str, Any]:
    """Loads the dynamic rating coefficients and countries."""
    _ensure_data_dir()
    rating_config_path = get_rating_config_filepath()
    default_config = {
        "countries": ["Israel", "Prague"],
        "coefficients": {
            "Kv": 0.5,
            "Ke": 0.5,
            "Kb": 1.0
        },
        "promethee_q": 0.5,
        "promethee_p": 3.5,
        "promethee_pref_func": "vshape_2",
        "normalization_mode": "default",
        "normalization_ceiling": 10.0,
        "waspas_lambda": 0.5
    }
    
    if os.path.exists(rating_config_path):
        with open(rating_config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Merge defaults for any missing keys
            default_config.update(data)
            return default_config
            
    with open(rating_config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=4)
    return default_config

def save_rating_config(config: Dict[str, Any]):
    rating_config_path = get_rating_config_filepath()
    with open(rating_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

# ==========================================
# MATHEMATICAL CORE: TRAPEZOID CONSTRUCTION
# ==========================================
def calculate_trapezoid(rating: float, volatility: float, uncertainty: float, bias: str, coeffs: dict) -> tuple:
    """
    Calculates the fuzzy trapezoid (a, b, c, d) for an evaluation.

    Base (Neutral) Construction:
        a = r - E*Ke - V*Kv
        b = r - E*Ke
        c = r + E*Ke
        d = r + E*Ke + V*Kv

    Directional Bias:
        - Optimistic ('opt'): Shifts ONLY the lower bounds [a, b] toward r by Kb.
          Represents a mitigated downside (less lower-end risk).
        - Pessimistic ('pes'): Shifts ONLY the upper bounds [c, d] toward r by Kb.
          Represents a mitigated upside (less higher-end potential).
        * The opposite half remains completely unchanged.
        * The shifted values are capped so they never cross the center rating r.

    Args:
        rating (float): The center rating (r)
        volatility (float): Volatility score (V)
        uncertainty (float): Uncertainty score (E)
        bias (str): 'neutral', 'opt', or 'pes'
        coeffs (dict): Dictionary containing 'Kv', 'Ke', 'Kb' multipliers

    Returns:
        tuple: (a, b, c, d) bounded between 0 and 10.
    """
    r = float(rating)
    v = float(volatility)
    u = float(uncertainty)

    # Extract coefficients (with safe fallbacks)
    kv = coeffs.get('Kv', 0.5)
    ke = coeffs.get('Ke', 0.5)
    kb = coeffs.get('Kb', coeffs.get('bias_coefficient', 1.0))

    # 1. Construct the neutral base trapezoid
    a = r - (u * ke) - (v * kv)
    b = r - (u * ke)
    c = r + (u * ke)
    d = r + (u * ke) + (v * kv)

    # 2. Apply Directional Bias
    if bias == 'opt':
        # Shift lower bounds up, but do not let them cross the center rating `r`
        a = min(a + kb, r)
        b = min(b + kb, r)
    elif bias == 'pes':
        # Shift upper bounds down, but do not let them cross the center rating `r`
        c = max(c - kb, r)
        d = max(d - kb, r)

    # 3. Clip to the absolute 0-10 scale
    a = max(0.0, min(10.0, a))
    b = max(0.0, min(10.0, b))
    c = max(0.0, min(10.0, c))
    d = max(0.0, min(10.0, d))

    # 4. Enforce structural validity (a <= b <= c <= d)
    b = max(a, b)
    c = max(b, c)
    d = max(c, d)

    return (a, b, c, d)

# ==========================================
# EVALUATION I/O & MOCK DATA INJECTION
# ==========================================
def load_evaluations(rating_config: Dict[str, Any], factors_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Loads joint evaluations. If missing, auto-generates neutral (5,0,0) data for testing UI."""
    filepath = get_evaluations_filepath()
    
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    # Auto-generate mock data if file doesn't exist
    evals = []
    countries = rating_config.get("countries", [])
    coeffs = rating_config.get("coefficients", {})
    
    for f in factors_config.get("factors", []):
        for country in countries:
            trap = calculate_trapezoid(5.0, 0, 0, "neutral", coeffs)
            evals.append({
                "country": country,
                "criterion_id": f["id"],
                "rating": 5.0,
                "volatility": 0,
                "uncertainty": 0,
                "bias": "neutral",
                "coefficients": coeffs,
                "trapezoid": trap
            })
            
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(evals, f, indent=4)
    return evals

def save_evaluations(evaluations: List[Dict[str, Any]]):
    filepath = get_evaluations_filepath()
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(evaluations, f, indent=4)

def parse_rating_input(val_str: str) -> Tuple[float, int, int, str]:
    """Parses standard single-line rating string: '6 4 1' or '6 4 1 opt'"""
    parts = val_str.strip().lower().split()
    if len(parts) not in [3, 4]:
        raise ValueError("Expected 3 or 4 values (rating, volatility, uncertainty, [bias]).")
        
    r = float(parts[0])
    v = int(parts[1])
    u = int(parts[2])
    
    if not (0 <= r <= 10): raise ValueError("Rating must be 0-10.")
    if not (0 <= v <= 5): raise ValueError("Volatility must be 0-5.")
    if not (0 <= u <= 5): raise ValueError("Uncertainty must be 0-5.")
    
    bias = "neutral"
    if len(parts) == 4:
        if parts[3] not in ["opt", "pes"]:
            raise ValueError("Bias must be 'opt' or 'pes'.")
        bias = parts[3]
        
    return r, v, u, bias

# ==========================================
# CLI DISPLAY HELPERS
# ==========================================
def _color_val(val: float, is_rating: bool = True) -> str:
    """Helper to apply Green/Yellow/Red coloring based on value."""
    if is_rating:
        if val >= 7.0: return f"{C_BOLD}{C_GREEN}{val:.1f}{C_RESET}"
        if val >= 4.0: return f"{C_BOLD}{C_YELLOW}{val:.1f}{C_RESET}"
        return f"{C_BOLD}{C_RED}{val:.1f}{C_RESET}"
    else: # Volatility / Uncertainty (low is good)
        if val <= 1: return f"{C_GREEN}{int(val)}{C_RESET}"
        if val <= 3: return f"{C_YELLOW}{int(val)}{C_RESET}"
        return f"{C_RED}{int(val)}{C_RESET}"

def _color_bias(bias: str) -> str:
    if bias == "opt": return f"{C_GREEN}opt{C_RESET}"
    if bias == "pes": return f"{C_RED}pes{C_RESET}"
    return "neutral"

# ==========================================
# CLI MENUS
# ==========================================
def run_evaluations_cli():
    try:
        factors_config = load_factors_config()
    except FileNotFoundError as e:
        print(f"{C_RED}{e}{C_RESET}")
        return

    while True:
        rating_config = load_rating_config()
        print(f"\n{C_BLUE}" + "="*50)
        print("  CRITERIA RATING & EVALUATION SYSTEM")
        print("="*50 + f"{C_RESET}")
        print("1. View / Edit Current Ratings")
        print("2. Initialize Ratings (Interactive Questionnaire)")
        print("3. Edit Coefficients (Kv, Ke, Bias)")
        print("4. Back to Main Menu")
        print(f"{C_BLUE}" + "="*50 + f"{C_RESET}")
        
        choice = input("Select an option [1-4]: ").strip()
        
        if choice == '1':
            view_ratings_menu(rating_config, factors_config)
        elif choice == '2':
            initialize_ratings_menu(rating_config, factors_config)
        elif choice == '3':
            edit_coefficients_menu(rating_config)
        elif choice == '4':
            break
        else:
            print(f"{C_YELLOW}Invalid choice.{C_RESET}")

def view_ratings_menu(rating_config: Dict, factors_config: Dict):
    domains = factors_config.get("domains", [])
    factors = factors_config.get("factors", [])
    countries = rating_config.get("countries", [])
    
    if not domains:
        print("Configuration missing domains.")
        return

    while True:
        print(f"\n{C_BLUE}" + "="*50)
        print(" VIEW RATINGS (CATEGORIES)")
        print("="*50 + f"{C_RESET}")
        
        for i, d in enumerate(domains):
            print(f" {i + 1}. {d['name']} ({d['short_name']})")
            
        print(f"{C_BLUE}" + "="*50 + f"{C_RESET}")
        print(" [1-X] Select a category to view")
        print(" [b]   Back")
        
        choice = input("\nEnter command: ").strip().lower()
        if choice == 'b':
            break
            
        try:
            domain_idx = int(choice) - 1
            if 0 <= domain_idx < len(domains):
                selected_domain = domains[domain_idx]
                domain_factors = [f for f in factors if f["domain_id"] == selected_domain["id"]]
                
                # --- CATEGORY VIEW LOOP ---
                while True:
                    evaluations = load_evaluations(rating_config, factors_config)
                    
                    print(f"\n{C_BLUE}" + "="*95)
                    print(f" {C_YELLOW}CRITERIA FOR: {selected_domain['name'].upper()}{C_BLUE}")
                    print("="*95 + f"{C_RESET}")
                    
                    table = []
                    for f in domain_factors:
                        # Build the row base (ID + Description)
                        wrapped_desc = textwrap.fill(f["description"], width=30)
                        display_desc = f"{C_BOLD}{C_YELLOW}{f['short_name']}{C_RESET}\n{wrapped_desc}"
                        
                        row = [f["id"], display_desc]
                        
                        # Add a column for each country dynamically
                        for country in countries:
                            ev = next((e for e in evaluations if e["criterion_id"] == f["id"] and e["country"] == country), None)
                            
                            if ev:
                                r_str = _color_val(ev["rating"], True)
                                v_str = _color_val(ev["volatility"], False)
                                u_str = _color_val(ev["uncertainty"], False)
                                b_str = _color_bias(ev["bias"])
                                
                                # Extract and format the trapezoid array
                                t = ev["trapezoid"]
                                t_str = f"T: ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}, {t[3]:.1f})"
                                
                                cell = f"{C_BOLD}R:{C_RESET} {r_str}\nV: {v_str} | U: {u_str}\nB: {b_str}\n{t_str}"
                            else:
                                cell = "N/A"
                                
                            row.append(cell)
                            
                        table.append(row)
                    
                    # Generate dynamic headers and print
                    headers = ["ID", "Criterion"] + [c.upper() for c in countries]
                    print(tabulate(table, headers=headers, tablefmt="grid"))
                    
                    print("\nOptions:")
                    print(" [e] Edit a rating (e.g., 'e c01 Israel')")
                    print(" [b] Back to Categories")
                    
                    sub = input("Enter command: ").strip().lower()
                    if sub == 'b':
                        break
                    elif sub.startswith('e '):
                        parts = sub.split()
                        if len(parts) >= 3:
                            fid = parts[1]
                            country_name = " ".join(parts[2:]).title() 
                            
                            if fid in [f["id"] for f in domain_factors] and country_name in countries:
                                while True:
                                    val_str = input(f"Enter rating for {fid} in {country_name} (r V U [bias]): ").strip()
                                    if val_str == "": break
                                    try:
                                        r, v, u, bias = parse_rating_input(val_str)
                                        
                                        # Update the evaluation record
                                        coeffs = rating_config.get("coefficients", {})
                                        trap = calculate_trapezoid(r, v, u, bias, coeffs)
                                        
                                        for i, e in enumerate(evaluations):
                                            if e["criterion_id"] == fid and e["country"] == country_name:
                                                evaluations[i].update({
                                                    "rating": r, "volatility": v, "uncertainty": u, 
                                                    "bias": bias, "coefficients": coeffs, "trapezoid": trap
                                                })
                                                break
                                                
                                        save_evaluations(evaluations)
                                        print(f"{C_GREEN}Updated successfully!{C_RESET}")
                                        break
                                    except ValueError as err:
                                        print(f"{C_RED}Error: {err}{C_RESET}")
                            else:
                                print(f"{C_YELLOW}Invalid ID or Country.{C_RESET}")
                        else:
                            print(f"{C_YELLOW}Format must be: e <ID> <Country> (e.g., 'e c01 Israel'){C_RESET}")
                    else:
                        print(f"{C_YELLOW}Invalid command.{C_RESET}")
            else:
                print(f"{C_YELLOW}Invalid category number.{C_RESET}")
        except ValueError:
            print(f"{C_YELLOW}Invalid input.{C_RESET}")

def initialize_ratings_menu(rating_config: Dict, factors_config: Dict):
    countries = rating_config.get("countries", [])
    domains = factors_config.get("domains", [])
    factors = factors_config.get("factors", [])
    
    print("\nWhich country are you rating?")
    for i, c in enumerate(countries): print(f"{i+1}. {c}")
    c_idx = input("Select [1/2]: ").strip()
    if c_idx not in ['1', '2']: return
    country = countries[int(c_idx)-1]
    
    evaluations = load_evaluations(rating_config, factors_config)
    coeffs = rating_config.get("coefficients", {})
    
    for domain in domains:
        print(f"\n{C_BLUE}" + "="*70)
        print(f" INITIALIZING COUNTRY: {country.upper()}")
        print(f" {C_YELLOW}CATEGORY: {domain['name'].upper()}{C_BLUE}")
        print("="*70 + f"{C_RESET}")
        
        domain_factors = [f for f in factors if f["domain_id"] == domain["id"]]
        for idx, f in enumerate(domain_factors):
            if idx > 0:
                print("\n" + "-"*35)  # Separator between questions
                
            print(f"\nHow do you rate '{C_YELLOW}{C_BOLD}{f['name']}{C_RESET}'?")
            wrapped_desc = textwrap.fill(f['description'], width=65)
            print(f"({wrapped_desc})")
            
            while True:
                val_str = input("Enter: Rating(0-10) Volatility(0-5) Uncertainty(0-5) [Bias(opt/pes)] (or press Enter to skip)\n> ").strip()
                if val_str == "":
                    print("Skipped.")
                    break
                    
                try:
                    # Initial parsing
                    r, v, u, bias = parse_rating_input(val_str)
                    
                    # --- NEW VALIDATION LOOP ---
                    while True:
                        trap = calculate_trapezoid(r, v, u, bias, coeffs)
                        
                        print(f"\n  Rating:      {_color_val(r, True)}")
                        print(f"  Volatility:  {_color_val(v, False)}")
                        print(f"  Uncertainty: {_color_val(u, False)}")
                        print(f"  Bias:        {_color_bias(bias)}")
                        print(f"  Trapezoid:   ({trap[0]:.1f}, {trap[1]:.1f}, {trap[2]:.1f}, {trap[3]:.1f})")
                        
                        confirm = input(f"\nPress [{C_GREEN}Enter{C_RESET}] to confirm, or type new values to fix a typo:\n> ").strip()
                        
                        if confirm == "":
                            # Confirmed! Save to DB
                            for i, e in enumerate(evaluations):
                                if e["criterion_id"] == f["id"] and e["country"] == country:
                                    evaluations[i].update({
                                        "rating": r, "volatility": v, "uncertainty": u, 
                                        "bias": bias, "coefficients": coeffs, "trapezoid": trap
                                    })
                                    break
                            break # Break out of validation loop
                        else:
                            # User typed new numbers to fix a typo!
                            try:
                                r, v, u, bias = parse_rating_input(confirm)
                                # The loop continues, recalculates, and previews the fixed numbers
                            except ValueError as err:
                                print(f"{C_RED}Error in new input: {err}{C_RESET}")
                                
                    break # Break out of the main question loop since we successfully saved
                    
                except ValueError as err:
                    print(f"{C_RED}Error: {err}{C_RESET}")
                    
    save_evaluations(evaluations)
    print(f"\n{C_GREEN}Finished initializing {domain['name']} for {country}!{C_RESET}")

def edit_coefficients_menu(rating_config: Dict):
    coeffs = rating_config["coefficients"]
    while True:
        print(f"\n{C_BLUE}" + "="*50)
        print(" EDIT COEFFICIENTS")
        print("="*50 + f"{C_RESET}")
        print(f"1. Kv (Volatility multiplier)   : {coeffs.get('Kv', 0.5)}")
        print(f"2. Ke (Uncertainty multiplier)  : {coeffs.get('Ke', 0.5)}")
        print(f"3. Kb (Bias multiplier)         : {coeffs.get('Kb', coeffs.get('bias_coefficient', 1.0))}")
        print("4. Back")
        
        choice = input("Select an option [1-4]: ").strip()
        if choice == '4':
            break
        
        keys = {'1': 'Kv', '2': 'Ke', '3': 'Kb'} # <--- Updated to Kb
        if choice in keys:
            val_str = input(f"Enter new value for {keys[choice]}: ").strip()
            try:
                val = float(val_str)
                coeffs[keys[choice]] = val
                save_rating_config(rating_config)
                print(f"{C_GREEN}Updated successfully.{C_RESET}")
            except ValueError:
                print(f"{C_RED}Invalid number.{C_RESET}")