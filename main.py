"""
Decision Support System - Main CLI Entry Point
"""

import sys
from src.factors_manager import display_factor_overview
from src.weights import run_weights_cli
from src.evaluations import run_evaluations_cli
from src.mcdm_engine import run_mcdm_cli

def main_menu():
    """Main interactive terminal loop."""
    while True:
        print("\n\033[94m" + "="*55)
        print("   COUPLE LIFE-DECISION SUPPORT SYSTEM (DSS)")
        print("="*55 + "\033[0m")
        print("1. View System Factors & Domains Overview")
        print("2. Hybrid Weight Engine (AHP Matrices & Ratings)")
        print("3. Criteria Rating / Evaluation System (Fuzzy Data)")
        print("4. Multi-Criteria Decision Engine (MCDM)")
        print("5. Exit")
        print("\033[94m" + "="*55 + "\033[0m")
        
        choice = input("Select an option [1-5]: ").strip()
        
        if choice == '1':
            display_factor_overview()
        elif choice == '2':
            run_weights_cli()
        elif choice == '3':
            run_evaluations_cli()
        elif choice == '4':
            run_mcdm_cli()
        elif choice == '5':
            print("\nExiting Decision Support System. Goodbye!\n")
            sys.exit(0)
        else:
            print("\033[93mInvalid choice. Please select 1 through 5.\033[0m")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting gracefully...")
        sys.exit(0)