"""
Decision Support System - MCDM Orchestrator
Handles validation, execution, saving/loading runs, snapshot generation, and result viewing/comparing.
"""

import os
import json
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from tabulate import tabulate

from src.factors_manager import load_factors_config
from src.evaluations import get_evaluations_filepath
from src.mcdm_methods import METHOD_REGISTRY

# ==========================================
# FILE PATHS & CONSTANTS
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RUNS_DIR = os.path.join(DATA_DIR, 'runs')
ENGINE_CONFIG_FILE = os.path.join(DATA_DIR, 'mcdm_config.json')
WEIGHTS_FILE = os.path.join(DATA_DIR, 'weights.json')

C_RED = '\033[91m'
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_BLUE = '\033[94m'
C_BOLD = '\033[1m'
C_RESET = '\033[0m'

# ==========================================
# CONFIG & STATE MANAGEMENT
# ==========================================
def _ensure_dirs():
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    if not os.path.exists(RUNS_DIR): os.makedirs(RUNS_DIR)

def load_engine_config() -> Dict[str, Any]:
    _ensure_dirs()
    default_config = {"parameters": {"WASPAS_lambda": 0.5}}
    if os.path.exists(ENGINE_CONFIG_FILE):
        with open(ENGINE_CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            default_config.update(data)
            return default_config
    with open(ENGINE_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=4)
    return default_config

def save_engine_config(config: Dict[str, Any]):
    with open(ENGINE_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

# ==========================================
# ENGINE DATA BUILDER & VALIDATION
# ==========================================
def _validate_and_build_matrices() -> Dict[str, Any]:
    factors_config = load_factors_config()
    factors = factors_config.get("factors", [])
    if not factors: raise ValueError("No factors defined in factors_config.json")
    
    if not os.path.exists(WEIGHTS_FILE):
        raise ValueError("Weights data missing. Please go to Option 2 and complete the Weight Engine.")
    with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
        weights_data = json.load(f)
    global_weights = weights_data.get("global_weights", {})
    
    # Extract domain percentages for the visual chart
    domain_map = {d["id"]: d["name"] for d in factors_config.get("domains", [])}
    cat_weights_display = {}
    for d_id, w in weights_data.get("category_weights", {}).items():
        if d_id in domain_map:
            cat_weights_display[domain_map[d_id]] = w * 100
        elif d_id in domain_map.values():
            cat_weights_display[d_id] = w * 100
    
    eval_path = get_evaluations_filepath()
    if not os.path.exists(eval_path):
        raise ValueError("Evaluations missing. Please go to Option 3 and complete the Rating System.")
    with open(eval_path, 'r', encoding='utf-8') as f:
        evaluations = json.load(f)
        
    countries = list(set([e["country"] for e in evaluations]))
    countries.sort()
    if len(countries) < 2:
        raise ValueError("Need evaluations for at least two countries to run MCDM.")

    c_ids = [f["id"] for f in factors]
    matrix = np.zeros((len(countries), len(factors)))
    weights_arr = np.zeros(len(factors))
    types_arr = np.zeros(len(factors))
    fuzzy_data = {c: {} for c in countries} 
    
    for j, fid in enumerate(c_ids):
        weights_arr[j] = global_weights.get(fid, 0.0)
        types_arr[j] = next((f["type"] for f in factors if f["id"] == fid), 1)
        
        for i, country in enumerate(countries):
            ev = next((e for e in evaluations if e["criterion_id"] == fid and e["country"] == country), None)
            if not ev:
                raise ValueError(f"Missing evaluation for {country} on criterion {fid}.")
            
            val = ev["rating"]
            matrix[i, j] = val if val > 0 else 0.0001
            fuzzy_data[country][fid] = ev.get("trapezoid", [val]*4)
            
    w_sum = np.sum(weights_arr)
    if w_sum <= 0:
        raise ValueError("Weights sum to 0. Please recalculate weights in Option 2.")
    weights_arr = weights_arr / w_sum
            
    return {
        "countries": countries,
        "criteria_ids": c_ids,
        "matrix": matrix,
        "weights": weights_arr,
        "types": types_arr,
        "fuzzy_data": fuzzy_data,
        "cat_weights_display": cat_weights_display,
        "raw_factors_config": factors_config,
        "raw_weights": weights_data,
        "raw_evaluations": evaluations
    }

# ==========================================
# EXECUTION (WITH FULL SNAPSHOT PERSISTENCE)
# ==========================================
def execute_run(method_names: List[str], run_name: str):
    print(f"\n{C_YELLOW}Validating inputs and building decision matrices...{C_RESET}")
    try:
        data = _validate_and_build_matrices()
    except ValueError as e:
        print(f"{C_RED}Validation Failed: {e}{C_RESET}")
        return
        
    config = load_engine_config()
    parameters = config.get("parameters", {})
    
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    results = {}
    for name in method_names:
        if name not in METHOD_REGISTRY: continue
        method = METHOD_REGISTRY[name]
        
        try:
            res = method.execute(
                matrix=data["matrix"], 
                weights=data["weights"], 
                types=data["types"], 
                parameters=parameters
            )
            res["method_type"] = method.method_type
            results[name] = res
        except Exception as e:
            results[name] = {"status": "error", "warnings": [str(e)], "method_type": method.method_type}

    # Filter evaluations to only the countries included in this run
    active_countries = data["countries"]
    filtered_evals = [e for e in data["raw_evaluations"] if e.get("country") in active_countries]

    run_snapshot = {
        "run_id": run_id,
        "name": run_name,
        "timestamp": datetime.now().isoformat(),
        "countries": active_countries,
        "category_weights": data["cat_weights_display"],
        "methods_executed": method_names,
        "parameters": parameters,
        "results": results,
        # --- Complete Historical Snapshot ---
        "snapshot": {
            "weights": data["raw_weights"],
            "evaluations": filtered_evals,
            "factors_config": data["raw_factors_config"],
            "criteria_ids": data["criteria_ids"],
            "types": [int(t) for t in data["types"]]
        }
    }
    
    _ensure_dirs()
    save_path = os.path.join(RUNS_DIR, f"{run_id}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(run_snapshot, f, indent=4)
        
    print(f"{C_GREEN}Run completed successfully! Saved with full snapshot as '{run_name}'.{C_RESET}")
    return run_snapshot

# ==========================================
# RESULT VIEWING & COMPARING
# ==========================================
def display_run_results(run_data: Dict[str, Any]):
    print(f"\n{C_BLUE}" + "="*80)
    print(f" RESULTS: {C_BOLD}{run_data['name']}{C_RESET} ({run_data['timestamp'][:16].replace('T', ' ')})")
    print("="*80 + f"{C_RESET}")
    
    cat_weights = run_data.get("category_weights", {})
    if cat_weights:
        print(f"\n{C_BLUE}--- CATEGORY WEIGHTS (W_cat) USED ---{C_RESET}")
        cat_list = sorted(cat_weights.items(), key=lambda x: x[1], reverse=True)
        max_name_len = max(len(name) for name, _ in cat_list)
        BAR_WIDTH = 30
        for name, pct in cat_list:
            blocks = int((pct / 100.0) * BAR_WIDTH)
            bar = "█" * blocks
            print(f" {name:<{max_name_len}}   {C_GREEN}{bar:<{BAR_WIDTH}}{C_RESET}  {pct:>4.1f}%")
            
    countries = run_data["countries"]
    results = run_data["results"]
    error_messages = []
    
    print(f"\n{C_BOLD}1. Method Scores & Rankings{C_RESET}")
    rank_table = []
    win_counts = {c: 0 for c in countries}
    
    for m_name, res in results.items():
        if res.get("status") == "success":
            scores = res["scores"]
            ranks = res["ranking"]
            
            winner_idx = np.argmin(ranks) 
            winner_name = countries[winner_idx]
            
            row = [m_name]
            for i, c in enumerate(countries):
                rank = int(ranks[i])
                if rank == 1: win_counts[c] += 1
                row.append(f"{scores[i]:.4f}\n(Rank {rank})")
                
            row.append(f"{C_GREEN}{C_BOLD}{winner_name}{C_RESET}")
            rank_table.append(row)
            
        elif res.get("status") == "not_implemented":
            row = [f"{C_YELLOW}{m_name}{C_RESET}"] + [f"{C_YELLOW}Pending{C_RESET}"] * len(countries) + ["-"]
            rank_table.append(row)
        else:
            row = [f"{C_RED}{m_name}{C_RESET}"] + [f"{C_RED}Error{C_RESET}"] * len(countries) + ["-"]
            rank_table.append(row)
            err_text = res.get("warnings", ["Unknown error"])[0]
            error_messages.append(f"{m_name}: {err_text}")
            
    headers = ["Method"] + countries + ["Winner"]
    print(tabulate(rank_table, headers=headers, tablefmt="grid"))
    
    if error_messages:
        print(f"\n{C_RED}{C_BOLD}Execution Errors:{C_RESET}")
        for msg in error_messages:
            print(f"{C_RED}- {msg}{C_RESET}")
    
    print(f"\n{C_BOLD}2. Agreement Summary (First Place Votes){C_RESET}")
    summary_table = []
    total_successful = sum(1 for r in results.values() if r.get("status") == "success")
    
    if total_successful > 0:
        for c in countries:
            wins = win_counts[c]
            pct = (wins / total_successful) * 100
            summary_table.append([c, wins, f"{pct:.1f}%"])
        summary_table.sort(key=lambda x: x[1], reverse=True)
        print(tabulate(summary_table, headers=["Country", "1st Place Ranks", "% Agreement"], tablefmt="simple"))
    else:
        print(f"{C_YELLOW}No deterministic runs completed successfully.{C_RESET}")
    print()

def compare_multiple_runs(runs_to_compare: List[Dict[str, Any]]):
    print(f"\n{C_BLUE}" + "="*90)
    print(f" COMPARING {len(runs_to_compare)} SAVED RUNS")
    print("="*90 + f"{C_RESET}")
    
    countries = runs_to_compare[0]["countries"]
    all_methods = set()
    for r in runs_to_compare:
        for m_name, res in r["results"].items():
            if res.get("status") == "success":
                all_methods.add(m_name)
                
    all_methods = sorted(list(all_methods))
    
    if not all_methods:
        print(f"{C_YELLOW}No successful method results to compare.{C_RESET}")
        return

    table = []
    for m_name in all_methods:
        first_in_group = True
        for run in runs_to_compare:
            res = run["results"].get(m_name, {})
            display_m_name = m_name if first_in_group else "" 
            
            if res.get("status") == "success":
                scores = res["scores"]
                ranks = res["ranking"]
                winner_idx = np.argmin(ranks)
                winner_name = countries[winner_idx]
                
                row = [display_m_name, run["name"]]
                for c in countries:
                    try:
                        c_idx = run["countries"].index(c)
                        row.append(f"{scores[c_idx]:.4f}\n(R{int(ranks[c_idx])})")
                    except ValueError:
                        row.append("N/A")
                row.append(f"{C_GREEN}{winner_name}{C_RESET}")
                table.append(row)
            else:
                row = [display_m_name, run["name"]] + ["N/A"] * len(countries) + ["-"]
                table.append(row)
            
            first_in_group = False
            
        table.append([""] * (3 + len(countries)))
        
    if table: table.pop()
                
    headers = ["Method", "Run Name"] + countries + ["Winner"]
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print()

# ==========================================
# CLI MENUS
# ==========================================
def configure_parameters_cli():
    config = load_engine_config()
    while True:
        print(f"\n{C_BLUE}" + "="*50)
        print(" ENGINE PARAMETERS")
        print("="*50 + f"{C_RESET}")
        print(f"1. WASPAS Lambda: {config['parameters']['WASPAS_lambda']} (0=WPM, 1=WSM, 0.5=Equal)")
        print("b. Back")
        
        choice = input("\nEnter choice: ").strip().lower()
        if choice == 'b': break
        if choice == '1':
            try:
                val = float(input("Enter new Lambda (0.0 to 1.0): "))
                if 0.0 <= val <= 1.0:
                    config['parameters']['WASPAS_lambda'] = val
                    save_engine_config(config)
                    print(f"{C_GREEN}Updated successfully.{C_RESET}")
                else:
                    print(f"{C_RED}Must be between 0.0 and 1.0.{C_RESET}")
            except ValueError:
                print(f"{C_RED}Invalid number.{C_RESET}")

def run_models_cli():
    while True:
        print(f"\n{C_BLUE}" + "="*50)
        print(" RUN MCDM MODELS")
        print("="*50 + f"{C_RESET}")
        print("1. Run ALL Deterministic Methods")
        print("2. Run ALL Fuzzy Methods (Status Check)")
        print("3. Run EVERYTHING")
        print("4. Run a SINGLE specific method")
        print("b. Back")
        
        choice = input("\nEnter choice: ").strip().lower()
        if choice == 'b': break
        
        det_methods = [m for m, obj in METHOD_REGISTRY.items() if obj.method_type == "deterministic"]
        fuz_methods = [m for m, obj in METHOD_REGISTRY.items() if obj.method_type == "fuzzy"]
        
        if choice == '1':
            to_run = det_methods
        elif choice == '2':
            to_run = fuz_methods
        elif choice == '3':
            to_run = det_methods + fuz_methods
        elif choice == '4':
            methods_list = list(METHOD_REGISTRY.keys())
            print(f"\n{C_BLUE}--- AVAILABLE METHODS ---{C_RESET}")
            for i, m in enumerate(methods_list):
                print(f" {i+1}. {m}")
            
            m_choice = input(f"\nSelect method [1-{len(methods_list)}]: ").strip()
            try:
                m_idx = int(m_choice) - 1
                if 0 <= m_idx < len(methods_list):
                    to_run = [methods_list[m_idx]]
                else:
                    print(f"{C_RED}Invalid method number.{C_RESET}")
                    continue
            except ValueError:
                print(f"{C_RED}Invalid input.{C_RESET}")
                continue
        else:
            continue
            
        run_name = input("\nEnter a name to save this run (e.g., 'baseline'): ").strip()
        if not run_name: run_name = "unnamed_run"
        
        snapshot = execute_run(to_run, run_name)
        if snapshot:
            display_run_results(snapshot)

def saved_runs_cli():
    _ensure_dirs()
    while True:
        runs = []
        for file in os.listdir(RUNS_DIR):
            if file.endswith('.json'):
                with open(os.path.join(RUNS_DIR, file), 'r', encoding='utf-8') as f:
                    runs.append(json.load(f))
                    
        runs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        print(f"\n{C_BLUE}" + "="*80)
        print(" SAVED RUNS DATABASE")
        print("="*80 + f"{C_RESET}")
        
        if not runs:
            print(f"{C_YELLOW}No saved runs found.{C_RESET}")
            return
            
        table = []
        for i, r in enumerate(runs):
            t_str = r['timestamp'][:16].replace('T', ' ')
            methods_count = len(r.get('methods_executed', []))
            has_snap = "📸 Snapshot" if "snapshot" in r else "📄 Summary"
            table.append([i+1, r['name'], t_str, methods_count, has_snap, r['run_id'][:12]])
            
        print(tabulate(table, headers=["#", "Run Name", "Date/Time", "Methods", "Format", "ID Snippet"], tablefmt="simple"))
        
        print("\nOptions:")
        print(" [1-X] Load and View Run")
        print(" [comp X Y] Compare multiple runs side-by-side (e.g., 'comp 1 2')")
        print(" [del X] Delete a run (e.g., 'del 2')")
        print(" [b] Back")
        
        choice = input("\nEnter command: ").strip().lower()
        if choice == 'b': break
        
        if choice.startswith('del '):
            try:
                idx = int(choice.split()[1]) - 1
                if 0 <= idx < len(runs):
                    target_id = runs[idx]["run_id"]
                    os.remove(os.path.join(RUNS_DIR, f"{target_id}.json"))
                    print(f"{C_GREEN}Run deleted.{C_RESET}")
            except:
                print(f"{C_RED}Invalid delete command.{C_RESET}")
        elif choice.startswith('comp '):
            parts = choice.split()[1:]
            runs_to_compare = []
            for p in parts:
                try:
                    idx = int(p) - 1
                    if 0 <= idx < len(runs):
                        runs_to_compare.append(runs[idx])
                except ValueError:
                    pass
            if len(runs_to_compare) >= 2:
                compare_multiple_runs(runs_to_compare)
            else:
                print(f"{C_YELLOW}Please provide at least 2 valid run numbers to compare (e.g., 'comp 1 2').{C_RESET}")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(runs):
                    display_run_results(runs[idx])
            except ValueError:
                pass

def run_mcdm_cli():
    while True:
        print(f"\n{C_BLUE}" + "="*50)
        print("  MCDM ENGINE & ORCHESTRATOR")
        print("="*50 + f"{C_RESET}")
        print("1. Configure Engine Parameters")
        print("2. Run MCDM Models")
        print("3. View Saved Runs & Diagnostics")
        print("4. Back to Main Menu")
        print(f"{C_BLUE}" + "="*50 + f"{C_RESET}")
        
        choice = input("Select an option [1-4]: ").strip()
        
        if choice == '1':
            configure_parameters_cli()
        elif choice == '2':
            run_models_cli()
        elif choice == '3':
            saved_runs_cli()
        elif choice == '4':
            break
        else:
            print(f"{C_YELLOW}Invalid choice.{C_RESET}")