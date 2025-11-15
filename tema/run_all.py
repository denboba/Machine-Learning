#!/usr/bin/env python3
"""
Tema Învățare Automată - Script Master
Rulează ambele părți ale temei (Bike + Autovit)

Usage:
    python3 run_all.py              # Rulează ambele părți
    python3 run_all.py --bike       # Doar Bike rental
    python3 run_all.py --autovit    # Doar Autovit
"""

import sys
import subprocess
import os

def run_script(script_name, description):
    """Run a Python script and display its output"""
    print("="*80)
    print(f"RULARE: {description}")
    print("="*80)
    print(f"Script: {script_name}\n")
    
    result = subprocess.run([sys.executable, script_name], 
                          capture_output=False, 
                          text=True)
    
    if result.returncode == 0:
        print(f"\n✅ {description} - SUCCES")
    else:
        print(f"\n❌ {description} - EROARE (cod: {result.returncode})")
    
    return result.returncode

def main():
    """Main function"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Parse arguments
    run_bike = '--bike' in sys.argv or len(sys.argv) == 1
    run_autovit = '--autovit' in sys.argv or len(sys.argv) == 1
    
    print("\n" + "="*80)
    print("TEMA ÎNVĂȚARE AUTOMATĂ - IMPLEMENTARE COMPLETĂ")
    print("="*80)
    print("\nCerințe tema.pdf:")
    print("  4.1 - EDA (minimum 4 analize per dataset)")
    print("  4.2 - Feature Engineering & Preprocessing")
    print("  4.3 - ML Models (6 modele cu hyperparameter tuning)")
    print("\n" + "="*80)
    
    results = {}
    
    # Run Bike rental
    if run_bike:
        print("\n\n📊 PARTEA 1: DATASET ÎNCHIRIERE BICICLETE")
        results['bike'] = run_script('tema_complete_implementation.py', 
                                     'Partea 1: Închiriere Biciclete')
    
    # Run Autovit
    if run_autovit:
        print("\n\n🚗 PARTEA 2: DATASET AUTOVIT (Prețuri Mașini)")
        results['autovit'] = run_script('tema_autovit_implementation.py',
                                       'Partea 2: Autovit')
    
    # Summary
    print("\n\n" + "="*80)
    print("REZUMAT EXECUȚIE")
    print("="*80)
    
    for part, code in results.items():
        status = "✅ SUCCES" if code == 0 else "❌ EROARE"
        print(f"{part.upper()}: {status}")
    
    print("\nFișiere generate:")
    print("  Bike: bike_*.png, predictii_biciclete_final.csv")
    print("  Autovit: autovit_*.png, predictii_autovit_final.csv")
    
    print("\nDocumentație: README_IMPLEMENTARE.md")
    print("="*80)
    
    # Exit with error if any script failed
    if any(code != 0 for code in results.values()):
        sys.exit(1)

if __name__ == '__main__':
    main()
