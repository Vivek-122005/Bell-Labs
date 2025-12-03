"""
Create Master Panel Dataset
Merges nutrients, food groups, population, and obesity into final master panel
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed")
CLEANED_DIR = DATA_DIR / "cleaned"
PANELS_DIR = DATA_DIR / "panels"
FINAL_DIR = DATA_DIR / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CREATING MASTER PANEL")
print("=" * 60)

# STEP 6 — merge nutrients + foodgroups + obesity into master panel
print("\n🔗 STEP 6: Merging datasets into master panel...")

# Load filtered panels
nut_filt = pd.read_csv(PANELS_DIR / "nutrient_panel_filtered.csv")
fg = pd.read_csv(PANELS_DIR / "foodgroup_energy_panel.csv")
ob_filt = pd.read_csv(PANELS_DIR / "obesity_panel_filtered.csv")
pop = pd.read_csv(CLEANED_DIR / "Cleaned_FAO_Population.csv")

# Ensure year is integer
for df in [nut_filt, fg, ob_filt, pop]:
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

print(f"   Loaded nutrient panel: {len(nut_filt):,} rows")
print(f"   Loaded food group panel: {len(fg):,} rows")
print(f"   Loaded obesity panel: {len(ob_filt):,} rows")
print(f"   Loaded population: {len(pop):,} rows")

# Merge nutrient panel and foodgroup panel (left join to keep nutrient rows)
print("\n   Merging nutrient and food group panels...")
master = nut_filt.merge(
    fg, 
    on=['country', 'year'], 
    how='left', 
    suffixes=('', '_fg')
)
print(f"   After nutrient+foodgroup merge: {len(master):,} rows")

# Merge obesity
print("   Merging obesity data...")
master = master.merge(
    ob_filt[['country', 'year', 'obesity_pct']], 
    on=['country', 'year'], 
    how='left'
)
print(f"   After obesity merge: {len(master):,} rows")

# Merge population if not already merged
if 'population' not in master.columns:
    print("   Merging population data...")
    master = master.merge(
        pop[['country', 'year', 'population']], 
        on=['country', 'year'], 
        how='left'
    )
    print(f"   After population merge: {len(master):,} rows")
else:
    print("   Population already in dataset")

# Basic sanity checks
print("\n   Basic sanity checks:")
print(f"   Rows: {len(master):,}")
print(f"   Countries: {master['country'].nunique()}")
print(f"   Years: {master['year'].min()} - {master['year'].max()}")
print(f"   Obesity missing: {master['obesity_pct'].isna().sum():,} ({master['obesity_pct'].isna().sum()/len(master)*100:.1f}%)")
print(f"   Energy missing: {master['energy_kcal_day'].isna().sum():,} ({master['energy_kcal_day'].isna().sum()/len(master)*100:.1f}%)")
print(f"   Population missing: {master['population'].isna().sum():,} ({master['population'].isna().sum()/len(master)*100:.1f}%)")

# Save before imputation
master_before_impute = master.copy()
master_before_impute.to_csv(DATA_DIR / "master_panel_before_impute.csv", index=False)
print(f"\n   ✅ Saved master_panel_before_impute.csv")

# STEP 7 — handle missing values and flags
print("\n📊 STEP 7: Handling missing values...")
print("   Documenting missing values:")

missing_summary = pd.DataFrame({
    'variable': ['obesity_pct', 'energy_kcal_day', 'protein_g_day', 'fat_g_day', 'population'],
    'missing_count': [
        master['obesity_pct'].isna().sum(),
        master['energy_kcal_day'].isna().sum(),
        master['protein_g_day'].isna().sum(),
        master['fat_g_day'].isna().sum(),
        master['population'].isna().sum()
    ],
    'missing_pct': [
        master['obesity_pct'].isna().sum() / len(master) * 100,
        master['energy_kcal_day'].isna().sum() / len(master) * 100,
        master['protein_g_day'].isna().sum() / len(master) * 100,
        master['fat_g_day'].isna().sum() / len(master) * 100,
        master['population'].isna().sum() / len(master) * 100
    ]
})
print(missing_summary.to_string(index=False))

# For visualization: we'll keep all rows but document missing
# For regression: users can decide on imputation strategy

# Optional: Linear interpolation for small gaps in nutrients per country
print("\n   Applying conservative linear interpolation for nutrients (max gap: 2 years)...")
master = master.sort_values(['country', 'year'])

# Interpolate nutrients (only for small gaps)
nutrient_cols = ['energy_kcal_day', 'protein_g_day', 'fat_g_day']
for col in nutrient_cols:
    if col in master.columns:
        master[col] = master.groupby('country')[col].apply(
            lambda g: g.interpolate(method='linear', limit=2, limit_direction='both')
        ).values

print(f"   After interpolation:")
for col in nutrient_cols:
    if col in master.columns:
        missing_after = master[col].isna().sum()
        print(f"     {col}: {missing_after:,} missing ({missing_after/len(master)*100:.1f}%)")

# STEP 8 — add derived columns & quality checks
print("\n🔍 STEP 8: Adding derived columns and quality checks...")

# Compute total energy from food groups
fg_cols = [c for c in master.columns if c not in [
    'country', 'year', 'energy_kcal_day', 'protein_g_day', 'fat_g_day', 
    'sugar_g_day', 'obesity_pct', 'population'
]]

print(f"   Food group columns: {len(fg_cols)}")
print(f"   Food groups: {fg_cols[:5]}...")

# Compute sum of foodgroup columns
master['fg_energy_sum'] = master[fg_cols].sum(axis=1)

# Compare with total energy
master['total_energy_check'] = master['energy_kcal_day']
master['fg_energy_diff'] = master['fg_energy_sum'] - master['total_energy_check']

# Check the difference
diff_stats = master['fg_energy_diff'].describe()
print(f"\n   Food group energy sum vs total energy:")
print(f"     Mean difference: {diff_stats['mean']:.2f} kcal")
print(f"     Std difference: {diff_stats['std']:.2f} kcal")
print(f"     Max absolute difference: {master['fg_energy_diff'].abs().max():.2f} kcal")

# Flag large discrepancies (more than 5% difference)
master['energy_check_flag'] = (master['fg_energy_diff'].abs() > master['total_energy_check'] * 0.05).astype(int)
large_diff_count = master['energy_check_flag'].sum()
print(f"     Rows with >5% difference: {large_diff_count:,} ({large_diff_count/len(master)*100:.1f}%)")

# Compute food group shares (as percentage of total energy)
for col in fg_cols:
    share_col = f"{col}_share"
    master[share_col] = (master[col] / master['total_energy_check'] * 100).round(2)
    # Handle division by zero
    master[share_col] = master[share_col].replace([np.inf, -np.inf], np.nan)

print(f"   Created food group share columns ({len(fg_cols)} shares)")

# STEP 9 — final save and lightweight documentation
print("\n💾 STEP 9: Final save and documentation...")

# Save final master panel
master.to_csv(FINAL_DIR / "master_panel_final.csv", index=False)
print(f"   ✅ Saved master_panel_final.csv ({len(master):,} rows)")

# Save a trimmed sample for quick loading
master_sample = master.sample(min(1000, len(master)), random_state=42)
master_sample.to_csv(DATA_DIR / "master_panel_sample.csv", index=False)
print(f"   ✅ Saved master_panel_sample.csv ({len(master_sample):,} rows)")

# Create README
readme_content = f"""# Processed Data Files

This directory contains cleaned and processed datasets ready for analysis.

## Master Panel Dataset

**File**: `master_panel_final.csv`

**Description**: Complete panel dataset with nutrients, food groups, population, and obesity data merged at country-year level.

**Rows**: {len(master):,}
**Countries**: {master['country'].nunique()}
**Years**: {int(master['year'].min())} - {int(master['year'].max())}

### Variables

#### Core Identifiers
- `country`: Country name (standardized to FAO naming)
- `year`: Year (integer, {int(master['year'].min())}-{int(master['year'].max())})

#### Nutrient Variables (per capita per day)
- `energy_kcal_day`: Total energy intake (kcal/capita/day)
- `protein_g_day`: Protein intake (g/capita/day)
- `fat_g_day`: Fat intake (g/capita/day)
- `sugar_g_day`: Sugar intake (g/capita/day) - **Note**: Mostly missing, not available in source data

#### Food Group Energy (kcal/capita/day)
The following columns represent energy intake from each food group:
{f''.join([f'- `{col}`: Energy from {col} (kcal/capita/day)\n' for col in fg_cols[:10]])}
... and {len(fg_cols) - 10} more food groups

#### Food Group Shares (%)
Each food group has a corresponding share column (e.g., `Cereals_share`) representing the percentage of total energy from that food group.

#### Demographic Variables
- `population`: Total population (integer, actual numbers, not thousands)
- `obesity_pct`: Prevalence of obesity among adults (BMI ≥ 30) as percentage

#### Quality Check Variables
- `fg_energy_sum`: Sum of all food group energies (should approximate `energy_kcal_day`)
- `fg_energy_diff`: Difference between food group sum and total energy
- `energy_check_flag`: Flag (0/1) indicating if difference > 5% of total energy

### Data Processing Notes

1. **Source Data**:
   - Nutrients: FAO Food Balance Sheets (2010-2023)
   - Food Groups: Aggregated from FAO items using Item_to_FoodGroup mapping
   - Population: FAO Population data (2010-2023)
   - Obesity: WHO Global Health Observatory (1990-2022)

2. **Year Coverage**: 
   - Common years used: {int(master['year'].min())}-{int(master['year'].max())} (intersection of nutrient and obesity data)

3. **Country Coverage**:
   - {master['country'].nunique()} countries in final panel
   - Filtered to countries present in both nutrient and obesity datasets

4. **Missing Data**:
   - Obesity missing: {master['obesity_pct'].isna().sum():,} rows ({master['obesity_pct'].isna().sum()/len(master)*100:.1f}%)
   - Energy missing: {master['energy_kcal_day'].isna().sum():,} rows ({master['energy_kcal_day'].isna().sum()/len(master)*100:.1f}%)
   - Population missing: {master['population'].isna().sum():,} rows ({master['population'].isna().sum()/len(master)*100:.1f}%)

5. **Imputation**:
   - Applied conservative linear interpolation for nutrients (max gap: 2 years) per country
   - No imputation applied to obesity or population
   - Original data before imputation saved as `master_panel_before_impute.csv`

6. **Food Group Mapping**:
   - Item-to-food-group mapping coverage: ~92%
   - Unmapped items assigned to "Other" category
   - Mapping file: `Item_to_FoodGroup.csv`

## Other Processed Files

### Panel Datasets
- `nutrient_panel.csv`: Nutrient data pivoted to columns (country-year level)
- `foodgroup_energy_panel.csv`: Food group energy aggregated by country-year
- `foodgroup_protein_g_day_panel.csv`: Food group protein aggregated by country-year
- `foodgroup_fat_g_day_panel.csv`: Food group fat aggregated by country-year
- `nutrient_panel_with_pop.csv`: Nutrient panel with population attached

### Filtered Panels
- `nutrient_panel_filtered.csv`: Nutrients filtered to common years/countries
- `obesity_panel_filtered.csv`: Obesity filtered to common years/countries

### Intermediate Files
- `master_panel_before_impute.csv`: Master panel before any imputation
- `master_panel_sample.csv`: Random sample (1000 rows) for quick exploration

## Usage Notes

- For regression analysis: Consider country fixed effects to handle country-level heterogeneity
- For visualization: May want to drop rows with missing key variables depending on analysis
- Food group shares sum to ~100% (check `fg_energy_sum` vs `energy_kcal_day` for quality)
- Sugar data (`sugar_g_day`) is mostly missing - consider using Sugar food group energy as proxy

## Data Quality

- All nutrient values are per-capita per-day (no need to divide by population)
- Units: kcal/capita/day for energy, g/capita/day for protein/fat/sugar
- Population values are actual numbers (converted from thousands in source)
- Obesity values are percentages (0-100 scale)

## Last Updated

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Save README
readme_path = DATA_DIR / "README.md"
with open(readme_path, 'w') as f:
    f.write(readme_content)
print(f"   ✅ Saved README.md")

# Final summary
print("\n" + "=" * 60)
print("✅ MASTER PANEL CREATION COMPLETE")
print("=" * 60)
print(f"\n📋 Final Dataset Summary:")
print(f"   File: master_panel_final.csv")
print(f"   Rows: {len(master):,}")
print(f"   Columns: {len(master.columns)}")
print(f"   Countries: {master['country'].nunique()}")
print(f"   Years: {int(master['year'].min())} - {int(master['year'].max())}")
print(f"\n📊 Key Variables:")
print(f"   Nutrients: energy_kcal_day, protein_g_day, fat_g_day, sugar_g_day")
print(f"   Food Groups: {len(fg_cols)} groups with energy and share columns")
print(f"   Demographics: population, obesity_pct")
print(f"\n📁 Files Created:")
print(f"   1. master_panel_final.csv - Complete master panel")
print(f"   2. master_panel_before_impute.csv - Before imputation")
print(f"   3. master_panel_sample.csv - Sample for quick loading")
print(f"   4. README.md - Documentation")
print("\n✅ Ready for analysis!")

