"""
Data Integration Script
Integrates FoodBalanceSheet, FoodSecurity, and Population datasets
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_food_balance_sheet():
    """Load FoodBalanceSheet data from raw directory"""
    print("Loading FoodBalanceSheet data...")
    
    # Try to find the file
    fbs_path = RAW_DATA_DIR / "FoodBalanceSheet_data" / "FoodBalanceSheets_E_All_Data_(Normalized).csv"
    
    if not fbs_path.exists():
        print(f"Warning: {fbs_path} not found. Creating sample structure...")
        return None
    
    try:
        df = pd.read_csv(fbs_path, low_memory=False)
        print(f"Loaded {len(df):,} rows from FoodBalanceSheet")
        return df
    except Exception as e:
        print(f"Error loading FoodBalanceSheet: {e}")
        return None


def load_food_security():
    """Load FoodSecurity data from raw directory"""
    print("Loading FoodSecurity data...")
    
    fsec_path = RAW_DATA_DIR / "FoodSecurity_data" / "Food_Security_Data_E_All_Data_(Normalized).csv"
    
    if not fsec_path.exists():
        print(f"Warning: {fsec_path} not found. Creating sample structure...")
        return None
    
    try:
        df = pd.read_csv(fsec_path, low_memory=False)
        print(f"Loaded {len(df):,} rows from FoodSecurity")
        return df
    except Exception as e:
        print(f"Error loading FoodSecurity: {e}")
        return None


def load_population():
    """Load Population data from raw directory and transform to long format"""
    print("Loading Population data...")
    
    pop_path = RAW_DATA_DIR / "Population_data" / "Population_E_All_Area_Groups_NOFLAG.csv"
    
    if not pop_path.exists():
        print(f"Warning: {pop_path} not found. Creating sample structure...")
        return None
    
    try:
        df = pd.read_csv(pop_path, low_memory=False)
        print(f"Loaded {len(df)} rows from Population (wide format)")
        
        # Transform from wide to long format
        # Identify year columns (Y1950, Y1951, etc.)
        year_cols = [col for col in df.columns if col.startswith('Y')]
        
        # Melt the dataframe
        id_vars = [col for col in df.columns if col not in year_cols]
        df_long = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=year_cols,
            var_name='Year_Str',
            value_name='Population'
        )
        
        # Extract year from column name (Y1950 -> 1950)
        df_long['Year'] = df_long['Year_Str'].str.replace('Y', '').astype(int)
        df_long = df_long.drop('Year_Str', axis=1)
        
        # Filter for total population (both sexes) - Element Code 511
        df_long = df_long[df_long['Element Code'] == 511].copy()
        
        # Convert population from thousands to actual numbers
        df_long['Population'] = df_long['Population'] * 1000
        
        # Standardize country name
        df_long['Country'] = df_long['Area'].str.strip().str.title()
        
        print(f"Transformed to {len(df_long):,} rows (long format)")
        return df_long[['Area', 'Country', 'Year', 'Population']]
        
    except Exception as e:
        print(f"Error loading Population: {e}")
        return None


def extract_nutritional_elements(fbs_df):
    """Extract relevant nutritional elements from FoodBalanceSheet"""
    if fbs_df is None:
        return None
    
    print("Extracting nutritional elements...")
    
    # Define element codes for key nutrients
    element_codes = {
        684: 'Fat_g_per_capita_day',  # Fat supply quantity (g/capita/day)
        674: 'Protein_g_per_capita_day',  # Protein supply quantity (g/capita/day)
        664: 'Calories_kcal_per_capita_day',  # Food supply (kcal/capita/day)
        645: 'Food_Quantity_kg_per_capita_yr',  # Food supply quantity (kg/capita/yr)
    }
    
    # Filter for relevant elements
    fbs_filtered = fbs_df[fbs_df['Element Code'].isin(element_codes.keys())].copy()
    
    # Map element codes to nutrient types
    fbs_filtered['Nutrient_Type'] = fbs_filtered['Element Code'].map(element_codes)
    
    # Standardize country name
    fbs_filtered['Country'] = fbs_filtered['Area'].str.strip().str.title()
    
    # Rename Value to Consumption_Value
    fbs_filtered = fbs_filtered.rename(columns={
        'Value': 'Consumption_Value',
        'Unit': 'Consumption_Unit',
        'Item': 'Food_Item'
    })
    
    # Select relevant columns
    cols = ['Country', 'Year', 'Nutrient_Type', 'Food_Item', 
            'Consumption_Value', 'Consumption_Unit', 'Element']
    fbs_filtered = fbs_filtered[cols].copy()
    
    # Ensure Year is integer
    fbs_filtered['Year'] = fbs_filtered['Year'].astype(int)
    
    print(f"Extracted {len(fbs_filtered):,} rows of nutritional data")
    return fbs_filtered


def integrate_datasets(fbs_df, pop_df, fsec_df=None):
    """Integrate all datasets"""
    print("\nIntegrating datasets...")
    
    # Start with nutritional data
    nutrition_df = extract_nutritional_elements(fbs_df)
    
    if nutrition_df is None or pop_df is None:
        print("Error: Missing required datasets")
        return None
    
    # Ensure Year is integer in both dataframes before merge
    nutrition_df['Year'] = nutrition_df['Year'].astype(int)
    pop_df['Year'] = pop_df['Year'].astype(int)
    
    # Merge nutrition data with population data
    integrated_df = pd.merge(
        nutrition_df,
        pop_df,
        on=['Country', 'Year'],
        how='inner'
    )
    
    # Ensure Year is integer after merge
    integrated_df['Year'] = integrated_df['Year'].astype(int)
    
    print(f"After merging with population: {len(integrated_df):,} rows")
    
    # Optionally merge with food security data
    if fsec_df is not None:
        fsec_df = fsec_df.copy()
        fsec_df['Country'] = fsec_df['Area'].str.strip().str.title()
        # Ensure Year is integer
        fsec_df['Year'] = pd.to_numeric(fsec_df['Year'], errors='coerce').astype('Int64')
        # Drop rows where Year conversion failed
        fsec_df = fsec_df.dropna(subset=['Year'])
        fsec_df['Year'] = fsec_df['Year'].astype(int)
        
        integrated_df = pd.merge(
            integrated_df,
            fsec_df[['Country', 'Year', 'Element', 'Value']],
            on=['Country', 'Year'],
            how='left',
            suffixes=('', '_fsec')
        )
        print(f"After merging with food security: {len(integrated_df):,} rows")
    
    return integrated_df


def save_integrated_data(df, output_path):
    """Save integrated dataset to CSV"""
    if df is None:
        print("No data to save")
        return
    
    print(f"\nSaving integrated dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")
    print(f"Columns: {', '.join(df.columns)}")


def main():
    """Main integration function"""
    print("=" * 60)
    print("Data Integration Pipeline")
    print("=" * 60)
    
    # Load datasets
    fbs_df = load_food_balance_sheet()
    pop_df = load_population()
    fsec_df = load_food_security()
    
    # Integrate datasets
    integrated_df = integrate_datasets(fbs_df, pop_df, fsec_df)
    
    # Save integrated dataset
    output_path = PROCESSED_DATA_DIR / "integrated_nutrition_data.csv"
    save_integrated_data(integrated_df, output_path)
    
    # Print summary statistics
    if integrated_df is not None:
        print("\n" + "=" * 60)
        print("Integration Summary")
        print("=" * 60)
        print(f"Total rows: {len(integrated_df):,}")
        print(f"Total columns: {len(integrated_df.columns)}")
        print(f"\nCountries: {integrated_df['Country'].nunique()}")
        print(f"Years: {integrated_df['Year'].min()} - {integrated_df['Year'].max()}")
        print(f"Nutrient types: {integrated_df['Nutrient_Type'].nunique()}")
        print(f"\nNutrient types: {', '.join(integrated_df['Nutrient_Type'].unique())}")
        print("\n" + "=" * 60)
    
    print("\nIntegration complete!")


if __name__ == "__main__":
    main()

