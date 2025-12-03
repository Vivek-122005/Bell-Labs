#!/usr/bin/env python3
"""
Main Data Processing Pipeline
Runs all preprocessing steps in the correct order to create the final master panel.
"""

import sys
import subprocess
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

def run_step(step_name, script_path):
    """Run a processing step and handle errors"""
    print("\n" + "=" * 70)
    print(f"STEP: {step_name}")
    print("=" * 70)
    
    # script_path can be a string (relative) or Path object
    if isinstance(script_path, str):
        script_path = SCRIPTS_DIR / script_path
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        # Run the script as a subprocess
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=False
        )
        
        print(f"✅ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {step_name}: Script exited with code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error in {step_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the complete data processing pipeline"""
    print("=" * 70)
    print("NUTRITION & OBESITY DATA PROCESSING PIPELINE")
    print("=" * 70)
    print("\nThis pipeline will:")
    print("1. Preprocess raw FAO data")
    print("2. Preprocess obesity data")
    print("3. Create food group mapping")
    print("4. Create panel datasets")
    print("5. Create master panel")
    print("\n" + "=" * 70)
    
    # Define pipeline steps
    steps = [
        ("Preprocessing FAO Data", "preprocessing/preprocess_fao_data.py"),
        ("Preprocessing Obesity Data", "preprocessing/preprocess_obesity_data.py"),
        ("Creating Food Group Mapping", "preprocessing/preprocess_food_group_mapping.py"),
        ("Creating Panel Datasets", "panels/create_panel_datasets.py"),
        ("Creating Master Panel", "panels/create_master_panel.py"),
    ]
    
    # Run each step
    for step_name, script_name in steps:
        success = run_step(step_name, script_name)
        if not success:
            print(f"\n❌ Pipeline failed at: {step_name}")
            print("Please fix the error and run again.")
            sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nFinal output: data/processed/final/master_panel_final.csv")
    print("\nNext steps:")
    print("1. Run EDA: python scripts/analysis/perform_eda.py")
    print("2. Create visualizations: python scripts/analysis/interactive_plot.py")
    print("3. Explore in notebooks: jupyter notebook notebooks/01_eda_visualization.ipynb")


if __name__ == "__main__":
    main()

