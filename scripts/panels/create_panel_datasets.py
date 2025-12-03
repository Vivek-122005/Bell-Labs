"""
Create Panel Datasets from Cleaned Data
Creates nutrient_panel.csv, foodgroup_panel.csv, and prepares for master_panel.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# STEP 0 — quick imports and helper functions
DATA_DIR = Path("data/processed")
CLEANED_DIR = DATA_DIR / "cleaned"
MAPPINGS_DIR = DATA_DIR / "mappings"
PANELS_DIR = DATA_DIR / "panels"
PANELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CREATING PANEL DATASETS")
print("=" * 60)

# STEP 1 — load and inspect files
print("\n📂 STEP 1: Loading and inspecting files...")
nut = pd.read_csv(CLEANED_DIR / "Cleaned_FAO_Nutrients.csv")
pop = pd.read_csv(CLEANED_DIR / "Cleaned_FAO_Population.csv")
mapping = pd.read_csv(MAPPINGS_DIR / "Item_to_FoodGroup.csv")
ob = pd.read_csv(CLEANED_DIR / "Cleaned_Obesity.csv")

print(f"   Loaded nutrients: {len(nut):,} rows")
print(f"   Loaded population: {len(pop):,} rows")
print(f"   Loaded food group mapping: {len(mapping):,} rows")
print(f"   Loaded obesity: {len(ob):,} rows")

# Quick checks
print("\n   Column checks:")
print(f"   Nutrients columns: {list(nut.columns)}")
print(f"   Population columns: {list(pop.columns)}")
print(f"   Mapping columns: {list(mapping.columns)}")
print(f"   Obesity columns: {list(ob.columns)}")

# Ensure types
print("\n   Converting year columns to integer...")
nut['year'] = pd.to_numeric(nut['year'], errors='coerce').astype('Int64')
pop['year'] = pd.to_numeric(pop['year'], errors='coerce').astype('Int64')
ob['year'] = pd.to_numeric(ob['year'], errors='coerce').astype('Int64')

# Checks to perform
print("\n   Data quality checks:")
print(f"   Unique elements in nutrients: {sorted(nut['element'].unique())}")
print(f"   Unique units in nutrients: {sorted(nut['unit_standard'].unique())}")
print(f"   Countries in nutrients: {nut['country'].nunique()}")
print(f"   Countries in population: {pop['country'].nunique()}")
print(f"   Countries in obesity: {ob['country'].nunique()}")
print(f"   Year range in nutrients: {nut['year'].min()} - {nut['year'].max()}")
print(f"   Year range in population: {pop['year'].min()} - {pop['year'].max()}")
print(f"   Year range in obesity: {ob['year'].min()} - {ob['year'].max()}")

# STEP 2 — create nutrient panel (pivot elements to columns)
print("\n📊 STEP 2: Creating nutrient panel...")
# Keep only needed columns from nut
keep = ['country', 'year', 'element', 'value_standard', 'item']
df = nut[keep].copy()

# For nutrients, prefer "Grand Total" rows when available
# If Grand Total exists, use it; otherwise use mean of individual items
print("   Prioritizing 'Grand Total' rows for aggregation...")

# Separate Grand Total and individual items
grand_total = df[df['item'] == 'Grand Total'].copy()
individual_items = df[df['item'] != 'Grand Total'].copy()

# For country-year-element combinations with Grand Total, use Grand Total
# For others, use mean of individual items
nutrient_panel_list = []

for (country, year, element), group in df.groupby(['country', 'year', 'element']):
    grand_total_rows = group[group['item'] == 'Grand Total']
    if len(grand_total_rows) > 0:
        # Use Grand Total value
        value = grand_total_rows['value_standard'].iloc[0]
    else:
        # Use mean of individual items
        value = group['value_standard'].mean()
    
    nutrient_panel_list.append({
        'country': country,
        'year': year,
        'element': element,
        'value_standard': value
    })

nutrient_panel_df = pd.DataFrame(nutrient_panel_list)

# Pivot so each element becomes a column
nutrient_panel = nutrient_panel_df.pivot_table(
    index=['country', 'year'],
    columns='element',
    values='value_standard',
    aggfunc='first'  # Should be unique now
).reset_index()

# Rename columns for clarity (remove any extra spaces)
nutrient_panel.columns.name = None  # Remove the name from columns index

print(f"   Created nutrient panel: {len(nutrient_panel):,} rows")
print(f"   Columns: {list(nutrient_panel.columns)}")
print(f"   Sample data:")
print(nutrient_panel.head())

# Check for missing elements
expected_elements = ['energy_kcal_day', 'protein_g_day', 'fat_g_day', 'sugar_g_day']
missing_elements = [e for e in expected_elements if e not in nutrient_panel.columns]
if missing_elements:
    print(f"   ⚠️  Warning: Missing elements: {missing_elements}")
    # Add missing columns as NaN
    for elem in missing_elements:
        nutrient_panel[elem] = np.nan

# Save nutrient panel
nutrient_panel.to_csv(PANELS_DIR / "nutrient_panel.csv", index=False)
print(f"   ✅ Saved to: {DATA_DIR / 'nutrient_panel.csv'}")

# STEP 3 — build food-group panel
print("\n🍎 STEP 3: Building food-group panel...")
# Join mapping on item or item_code
nut_items = nut.copy()

# Clean item names for better matching
nut_items['item_clean'] = nut_items['item'].str.strip()
mapping['item_clean'] = mapping['item'].str.strip()

# Prefer matching on item_code if present
if 'item_code' in nut_items.columns and nut_items['item_code'].notna().any():
    nut_items['item_code'] = pd.to_numeric(nut_items['item_code'], errors='coerce')
    mapping['item_code'] = pd.to_numeric(mapping['item_code'], errors='coerce')
    print("   Merging on item_code first...")
    merged = nut_items.merge(
        mapping[['item_code', 'food_group']], 
        on='item_code', 
        how='left', 
        suffixes=('', '_mapping')
    )
    
    # For rows that didn't match on item_code, try matching on item name
    unmatched_mask = merged['food_group'].isna()
    matched_on_code = (~unmatched_mask).sum()
    print(f"   Rows matched on item_code: {matched_on_code:,}")
    print(f"   Rows unmatched on item_code: {unmatched_mask.sum():,}")
    
    if unmatched_mask.sum() > 0:
        print("   Trying item name matching for unmatched rows...")
        # Create a mapping dictionary from item name to food_group
        item_to_fg = dict(zip(mapping['item_clean'], mapping['food_group']))
        
        # Apply mapping to unmatched rows
        unmatched_indices = merged[unmatched_mask].index
        merged.loc[unmatched_indices, 'food_group'] = merged.loc[unmatched_indices, 'item_clean'].map(item_to_fg)
        
        matched_on_name = merged.loc[unmatched_indices, 'food_group'].notna().sum()
        print(f"   Additional rows matched on item name: {matched_on_name:,}")
else:
    print("   Merging on item name...")
    merged = nut_items.merge(
        mapping[['item_clean', 'food_group']],
        left_on='item_clean',
        right_on='item_clean',
        how='left',
        suffixes=('', '_mapping')
    )

print(f"   Merged dataset: {len(merged):,} rows")
print(f"   Rows with food_group: {merged['food_group'].notna().sum():,}")
print(f"   Rows without food_group: {merged['food_group'].isna().sum():,}")

# Focus on energy rows
energy_rows = merged[merged['element'] == 'energy_kcal_day'].copy()
print(f"   Energy rows: {len(energy_rows):,}")

# Exclude "Grand Total" items from food group aggregation
# (Grand Total is the sum, not a food group)
energy_rows = energy_rows[energy_rows['item'] != 'Grand Total'].copy()
print(f"   Energy rows (excluding Grand Total): {len(energy_rows):,}")

# Remove rows without a food_group (assign 'Other' if missing)
energy_rows['food_group'] = energy_rows['food_group'].fillna('Other')
print(f"   Food groups: {sorted(energy_rows['food_group'].unique())}")

# Aggregate: sum energy by country-year-food_group
fg_energy = energy_rows.groupby(['country', 'year', 'food_group'])['value_standard'].sum().reset_index()
fg_energy = fg_energy.rename(columns={'value_standard': 'energy_kcal_day'})
print(f"   Aggregated food group rows: {len(fg_energy):,}")

# Pivot food_group to columns
foodgroup_energy_panel = fg_energy.pivot_table(
    index=['country', 'year'],
    columns='food_group',
    values='energy_kcal_day',
    aggfunc='sum'
).reset_index().fillna(0)

# Clean column names (remove any extra spaces)
foodgroup_energy_panel.columns.name = None

print(f"   Created food group panel: {len(foodgroup_energy_panel):,} rows")
print(f"   Food group columns: {[c for c in foodgroup_energy_panel.columns if c not in ['country', 'year']]}")
print(f"   Sample data:")
print(foodgroup_energy_panel.head())

# Save food group panel
foodgroup_energy_panel.to_csv(PANELS_DIR / "foodgroup_energy_panel.csv", index=False)
print(f"   ✅ Saved to: {PANELS_DIR / 'foodgroup_energy_panel.csv'}")

# Also create grams-based food group aggregates for protein, fat, sugar
print("\n   Creating grams-based food group aggregates...")
for element in ['protein_g_day', 'fat_g_day', 'sugar_g_day']:
    if element not in merged['element'].values:
        print(f"   ⚠️  Skipping {element} (not found in data)")
        continue
    
    element_rows = merged[merged['element'] == element].copy()
    element_rows['food_group'] = element_rows['food_group'].fillna('Other')
    
    # Aggregate by food group
    fg_element = element_rows.groupby(['country', 'year', 'food_group'])['value_standard'].sum().reset_index()
    fg_element = fg_element.rename(columns={'value_standard': element})
    
    # Pivot to columns
    fg_panel = fg_element.pivot_table(
        index=['country', 'year'],
        columns='food_group',
        values=element,
        aggfunc='sum'
    ).reset_index().fillna(0)
    fg_panel.columns.name = None
    
    # Save separately or merge into main foodgroup panel
    output_file = PANELS_DIR / f"foodgroup_{element}_panel.csv"
    fg_panel.to_csv(output_file, index=False)
    print(f"   ✅ Saved {element} panel to: {output_file}")

# STEP 4 — attach population
print("\n👥 STEP 4: Attaching population to nutrient panel...")
nutrient_panel_with_pop = nutrient_panel.merge(
    pop[['country', 'year', 'population']],
    on=['country', 'year'],
    how='left'
)

# Check missing population
missing_pop = nutrient_panel_with_pop['population'].isna().sum()
print(f"   Missing population rows: {missing_pop:,} ({missing_pop/len(nutrient_panel_with_pop)*100:.1f}%)")

if missing_pop > 0:
    print("   Sample rows with missing population:")
    print(nutrient_panel_with_pop[nutrient_panel_with_pop['population'].isna()][['country', 'year']].head(10))

nutrient_panel_with_pop.to_csv(PANELS_DIR / "nutrient_panel_with_pop.csv", index=False)
print(f"   ✅ Saved to: {DATA_DIR / 'nutrient_panel_with_pop.csv'}")

# STEP 5 — align years and countries
print("\n🔗 STEP 5: Aligning years and countries...")
# Which years overlap between nutrient_panel and obesity?
years_nut = set(nutrient_panel['year'].dropna().unique())
years_ob = set(ob['year'].dropna().unique())
common_years = sorted(list(years_nut & years_ob))
print(f"   Common years: {min(common_years)} to {max(common_years)} ({len(common_years)} years)")

# Which countries overlap?
countries_nut = set(nutrient_panel['country'].dropna().unique())
countries_ob = set(ob['country'].dropna().unique())
common_countries = sorted(list(countries_nut & countries_ob))
print(f"   Common countries: {len(common_countries)}")
print(f"   Countries only in nutrients: {len(countries_nut - countries_ob)}")
print(f"   Countries only in obesity: {len(countries_ob - countries_nut)}")

# Show some country name mismatches
if len(countries_nut - countries_ob) > 0:
    print("\n   Sample countries only in nutrients:")
    print(f"   {sorted(list(countries_nut - countries_ob))[:10]}")
if len(countries_ob - countries_nut) > 0:
    print("\n   Sample countries only in obesity:")
    print(f"   {sorted(list(countries_ob - countries_nut))[:10]}")

# Create filtered panels
nut_filt = nutrient_panel[
    (nutrient_panel['year'].isin(common_years)) & 
    (nutrient_panel['country'].isin(common_countries))
].copy()

ob_filt = ob[
    (ob['year'].isin(common_years)) & 
    (ob['country'].isin(common_countries))
].copy()

print(f"\n   Filtered nutrient panel: {len(nut_filt):,} rows")
print(f"   Filtered obesity panel: {len(ob_filt):,} rows")

# Save filtered panels
nut_filt.to_csv(PANELS_DIR / "nutrient_panel_filtered.csv", index=False)
ob_filt.to_csv(PANELS_DIR / "obesity_panel_filtered.csv", index=False)
print(f"   ✅ Saved filtered panels")

# Check if we can use ISO3 for better matching
print("\n   Checking ISO3 matching potential...")
if 'iso3' in ob.columns:
    print(f"   Obesity dataset has ISO3 codes")
    # Check if we can get ISO3 for nutrients from AreaCodes
    try:
        from pathlib import Path
        area_codes = pd.read_csv(Path("data/raw/FoodBalanceSheet_data/FoodBalanceSheets_E_AreaCodes.csv"))
        area_codes.columns = area_codes.columns.str.strip()
        if 'Area' in area_codes.columns and 'Area Code (M49)' in area_codes.columns:
            # M49 codes can be converted to ISO3, but this is complex
            # For now, we'll note that country name matching is the primary method
            print(f"   AreaCodes available for potential ISO3 mapping")
    except:
        print(f"   Could not load AreaCodes for ISO3 mapping")

print("\n✅ Panel dataset creation complete!")
print("=" * 60)
print("\n📋 Summary of created files:")
print(f"   1. nutrient_panel.csv - {len(nutrient_panel):,} rows")
print(f"   2. foodgroup_energy_panel.csv - {len(foodgroup_energy_panel):,} rows")
print(f"   3. nutrient_panel_with_pop.csv - {len(nutrient_panel_with_pop):,} rows")
print(f"   4. Filtered panels (common years/countries) saved")
print(f"\n   Ready for master_panel.csv creation in next step!")

