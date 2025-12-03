#!/usr/bin/env python3
"""
Optional cleanup script to remove intermediate processed files.
Only keeps essential files needed for analysis.
Run this if you want to clean up intermediate files (they can be regenerated).
"""

from pathlib import Path
import shutil

PROCESSED_DIR = Path("data/processed")

# Files to KEEP (essential)
ESSENTIAL_FILES = {
    "Cleaned_FAO_Nutrients.csv",
    "Cleaned_FAO_Population.csv", 
    "Cleaned_Obesity.csv",
    "Item_to_FoodGroup.csv",
    "master_panel_final.csv",
    "README.md"
}

# Files to REMOVE (intermediate, can be regenerated)
INTERMEDIATE_FILES = {
    "Cleaned_Population.csv",  # Redundant (same as Cleaned_FAO_Population.csv)
    "integrated_nutrition_data.csv",  # Old format
    "nutrient_panel.csv",
    "nutrient_panel_with_pop.csv",
    "nutrient_panel_filtered.csv",
    "obesity_panel_filtered.csv",
    "foodgroup_energy_panel.csv",
    "foodgroup_fat_g_day_panel.csv",
    "foodgroup_protein_g_day_panel.csv",
    "master_panel_before_impute.csv",
    "master_panel_sample.csv",
    "master_panel_with_shares.csv",
}

def main():
    print("=" * 70)
    print("CLEANUP INTERMEDIATE FILES")
    print("=" * 70)
    print("\nThis will remove intermediate files that can be regenerated.")
    print("Essential files will be kept.")
    print("\nFiles to KEEP:")
    for f in sorted(ESSENTIAL_FILES):
        print(f"  ✓ {f}")
    
    print("\nFiles to REMOVE:")
    removed_count = 0
    for f in sorted(INTERMEDIATE_FILES):
        file_path = PROCESSED_DIR / f
        if file_path.exists():
            print(f"  ✗ {f}")
            removed_count += 1
        else:
            print(f"  - {f} (not found)")
    
    if removed_count == 0:
        print("\n✅ No intermediate files to remove.")
        return
    
    response = input(f"\nRemove {removed_count} intermediate file(s)? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Remove files
    print("\nRemoving files...")
    for f in INTERMEDIATE_FILES:
        file_path = PROCESSED_DIR / f
        if file_path.exists():
            file_path.unlink()
            print(f"  ✓ Removed: {f}")
    
    print(f"\n✅ Cleanup complete! Removed {removed_count} file(s).")
    print("\nNote: You can regenerate these files by running:")
    print("  python run_pipeline.py")

if __name__ == "__main__":
    main()

