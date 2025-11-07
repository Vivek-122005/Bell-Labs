"""
Dataset Analysis Script
Analyzes data quality and completeness for research questions
Enhanced version with better insights
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Regional aggregations to identify (common patterns)
REGIONAL_PATTERNS = [
    'world', 'africa', 'asia', 'europe', 'americas', 'oceania',
    'eastern', 'western', 'northern', 'southern', 'middle', 'central',
    'union', 'countries', 'lifdcs', 'nfidcs', 'ldcs', 'sids',
    'caribbean', 'south america', 'north america', 'central america',
    'melanesia', 'micronesia', 'polynesia', 'australia and new zealand',
    'south-eastern', 'south eastern', 'sub-saharan', 'sub saharan'
]

# Known individual countries (to override region detection)
KNOWN_COUNTRIES = [
    'afghanistan', 'albania', 'algeria', 'argentina', 'australia', 'austria',
    'bangladesh', 'belgium', 'brazil', 'canada', 'china', 'france', 'germany',
    'india', 'indonesia', 'italy', 'japan', 'mexico', 'nigeria', 'pakistan',
    'russia', 'south africa', 'south korea', 'spain', 'thailand', 'turkey',
    'united kingdom', 'united states', 'vietnam'
]


def is_region(country_name):
    """Check if a country name is actually a regional aggregation"""
    if pd.isna(country_name):
        return False
    country_lower = str(country_name).lower().strip()
    
    # Check if it's a known individual country first
    if any(known_country in country_lower for known_country in KNOWN_COUNTRIES):
        # But exclude if it contains regional patterns
        if any(pattern in country_lower for pattern in ['eastern', 'western', 'northern', 'southern', 'central', 'middle']):
            return True
        return False
    
    # Check for regional patterns
    return any(pattern in country_lower for pattern in REGIONAL_PATTERNS)


def analyze_data_quality(df, dataset_name):
    """Analyze data quality for a dataset"""
    print(f"\n{'='*60}")
    print(f"Data Quality Analysis: {dataset_name}")
    print(f"{'='*60}")
    
    if df is None:
        print("Dataset not available")
        return
    
    print(f"\nBasic Statistics:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values analysis
    print(f"\nMissing Values Analysis:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    if len(missing_df) > 0:
        print(missing_df.to_string())
        print(f"\n  Total missing cells: {missing.sum():,}")
        print(f"  Percentage of missing data: {(missing.sum() / (len(df) * len(df.columns)) * 100):.2f}%")
    else:
        print("  No missing values")
    
    # Data types summary
    print(f"\nData Types Summary:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Year coverage
    if 'Year' in df.columns:
        print(f"\nYear Coverage:")
        # Try to convert Year to numeric for calculations
        year_numeric = pd.to_numeric(df['Year'], errors='coerce')
        if year_numeric.notna().sum() > 0:
            print(f"  Min year: {year_numeric.min():.0f}")
            print(f"  Max year: {year_numeric.max():.0f}")
            print(f"  Unique years: {df['Year'].nunique()}")
            print(f"  Year range: {year_numeric.max() - year_numeric.min() + 1:.0f} years")
        else:
            print(f"  Min year: {df['Year'].min()}")
            print(f"  Max year: {df['Year'].max()}")
            print(f"  Unique years: {df['Year'].nunique()}")
    
    # Country/Area coverage with region identification
    if 'Country' in df.columns or 'Area' in df.columns:
        country_col = 'Country' if 'Country' in df.columns else 'Area'
        unique_areas = df[country_col].unique()
        
        # Separate countries from regions
        regions = [area for area in unique_areas if is_region(area)]
        countries = [area for area in unique_areas if not is_region(area)]
        
        print(f"\nGeographic Coverage:")
        print(f"  Total unique areas: {len(unique_areas)}")
        print(f"  Individual countries: {len(countries)}")
        print(f"  Regional aggregations: {len(regions)}")
        
        if len(countries) > 0:
            print(f"\n  Sample individual countries ({min(10, len(countries))}):")
            for country in sorted(countries)[:10]:
                print(f"    - {country}")
        
        if len(regions) > 0:
            print(f"\n  Regional aggregations ({min(10, len(regions))}):")
            for region in sorted(regions)[:10]:
                print(f"    - {region}")
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric Columns Summary:")
        for col in numeric_cols[:5]:  # Show first 5 numeric columns
            if df[col].notna().sum() > 0:
                print(f"  {col}:")
                print(f"    Mean: {df[col].mean():.2f}")
                print(f"    Median: {df[col].median():.2f}")
                print(f"    Std: {df[col].std():.2f}")
                print(f"    Min: {df[col].min():.2f}")
                print(f"    Max: {df[col].max():.2f}")


def check_research_question_feasibility(integrated_df):
    """Check which research questions can be answered with available data"""
    print(f"\n{'='*60}")
    print("Research Question Feasibility Assessment")
    print(f"{'='*60}")
    
    if integrated_df is None:
        print("Integrated dataset not available")
        return
    
    # Check for sugar in Food_Item
    has_sugar = False
    has_fiber = False
    if 'Food_Item' in integrated_df.columns:
        food_items_str = ' '.join(integrated_df['Food_Item'].dropna().astype(str).unique())
        has_sugar = any(term in food_items_str.lower() for term in ['sugar', 'sweetener', 'sweet'])
        has_fiber = any(term in food_items_str.lower() for term in ['fiber', 'fibre', 'dietary fiber'])
    
    questions = {
        "Question 1 - Sugar, Fat, Protein, Fiber Consumption": {
            "Fat": "Fat_g_per_capita_day" in integrated_df['Nutrient_Type'].values if 'Nutrient_Type' in integrated_df.columns else False,
            "Protein": "Protein_g_per_capita_day" in integrated_df['Nutrient_Type'].values if 'Nutrient_Type' in integrated_df.columns else False,
            "Sugar": has_sugar,
            "Fiber": has_fiber,
        },
        "Question 3 - Dietary Homogenization": {
            "Food consumption by country": 'Country' in integrated_df.columns,
            "Food items by country": 'Food_Item' in integrated_df.columns,
            "Temporal data": 'Year' in integrated_df.columns,
        },
        "Question 4 - Food Groups Consumption Changes": {
            "Food groups available": 'Food_Item' in integrated_df.columns,
            "Temporal data": 'Year' in integrated_df.columns,
        },
        "Question 6 - Interactive Bar Plot": {
            "Nutrient consumption": 'Nutrient_Type' in integrated_df.columns,
            "Population data": 'Population' in integrated_df.columns,
            "Country and year data": 'Country' in integrated_df.columns and 'Year' in integrated_df.columns,
        }
    }
    
    for question, checks in questions.items():
        print(f"\n{question}:")
        all_available = True
        for check, available in checks.items():
            status = "✅" if available else "❌"
            print(f"  {status} {check}")
            if not available:
                all_available = False
        
        overall_status = "✅ Can be answered" if all_available else "⚠️ Partially available"
        print(f"  Overall: {overall_status}")


def analyze_data_completeness(integrated_df):
    """Analyze data completeness across countries and years"""
    print(f"\n{'='*60}")
    print("Data Completeness Analysis")
    print(f"{'='*60}")
    
    if integrated_df is None:
        print("Integrated dataset not available")
        return
    
    if 'Country' in integrated_df.columns and 'Year' in integrated_df.columns:
        # Separate countries from regions
        integrated_df['Is_Region'] = integrated_df['Country'].apply(is_region)
        countries_df = integrated_df[~integrated_df['Is_Region']].copy()
        regions_df = integrated_df[integrated_df['Is_Region']].copy()
        
        print(f"\nIndividual Countries vs Regions:")
        print(f"  Individual countries: {countries_df['Country'].nunique()}")
        print(f"  Regional aggregations: {regions_df['Country'].nunique()}")
        print(f"  Total rows (countries): {len(countries_df):,}")
        print(f"  Total rows (regions): {len(regions_df):,}")
        
        # Country-year coverage for individual countries
        if len(countries_df) > 0:
            country_year_coverage = countries_df.groupby('Country')['Year'].agg(['min', 'max', 'count', 'nunique'])
            country_year_coverage.columns = ['First_Year', 'Last_Year', 'Total_Records', 'Unique_Years']
            country_year_coverage = country_year_coverage.sort_values('Total_Records', ascending=False)
            
            print(f"\nTop 10 Individual Countries by Data Coverage:")
            print(country_year_coverage.head(10).to_string())
            
            if len(country_year_coverage) > 10:
                print(f"\nBottom 10 Individual Countries by Data Coverage:")
                print(country_year_coverage.tail(10).to_string())
        
        # Regional aggregations coverage
        if len(regions_df) > 0:
            region_year_coverage = regions_df.groupby('Country')['Year'].agg(['min', 'max', 'count', 'nunique'])
            region_year_coverage.columns = ['First_Year', 'Last_Year', 'Total_Records', 'Unique_Years']
            region_year_coverage = region_year_coverage.sort_values('Total_Records', ascending=False)
            
            print(f"\nTop 10 Regional Aggregations by Data Coverage:")
            print(region_year_coverage.head(10).to_string())
        
        # Year coverage
        year_coverage = integrated_df.groupby('Year')['Country'].nunique().sort_index()
        print(f"\nYear Coverage (Number of Countries/Regions per Year):")
        print(year_coverage.to_string())
        
        # Year coverage for individual countries only
        if len(countries_df) > 0:
            year_coverage_countries = countries_df.groupby('Year')['Country'].nunique().sort_index()
            print(f"\nYear Coverage - Individual Countries Only:")
            print(year_coverage_countries.to_string())
    
    if 'Nutrient_Type' in integrated_df.columns:
        nutrient_coverage = integrated_df.groupby('Nutrient_Type').size().sort_values(ascending=False)
        print(f"\nNutrient Type Coverage:")
        print(nutrient_coverage.to_string())
        
        # Nutrient coverage by country type
        if 'Is_Region' in integrated_df.columns:
            print(f"\nNutrient Coverage by Geographic Type:")
            nutrient_by_type = integrated_df.groupby(['Is_Region', 'Nutrient_Type']).size().unstack(fill_value=0)
            print(nutrient_by_type.to_string())


def analyze_nutrient_distribution(integrated_df):
    """Analyze distribution of nutrient values"""
    print(f"\n{'='*60}")
    print("Nutrient Value Distribution Analysis")
    print(f"{'='*60}")
    
    if integrated_df is None or 'Consumption_Value' not in integrated_df.columns:
        print("Consumption values not available")
        return
    
    if 'Nutrient_Type' in integrated_df.columns:
        for nutrient in integrated_df['Nutrient_Type'].unique():
            nutrient_data = integrated_df[
                (integrated_df['Nutrient_Type'] == nutrient) & 
                (integrated_df['Consumption_Value'].notna())
            ]['Consumption_Value']
            
            if len(nutrient_data) > 0:
                print(f"\n{nutrient}:")
                print(f"  Count: {len(nutrient_data):,}")
                print(f"  Mean: {nutrient_data.mean():.2f}")
                print(f"  Median: {nutrient_data.median():.2f}")
                print(f"  Std: {nutrient_data.std():.2f}")
                print(f"  Min: {nutrient_data.min():.2f}")
                print(f"  Max: {nutrient_data.max():.2f}")
                print(f"  25th percentile: {nutrient_data.quantile(0.25):.2f}")
                print(f"  75th percentile: {nutrient_data.quantile(0.75):.2f}")
                
                # Outlier detection (using IQR method)
                Q1 = nutrient_data.quantile(0.25)
                Q3 = nutrient_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = nutrient_data[(nutrient_data < lower_bound) | (nutrient_data > upper_bound)]
                print(f"  Potential outliers: {len(outliers):,} ({len(outliers)/len(nutrient_data)*100:.2f}%)")


def generate_quality_report(integrated_df):
    """Generate a comprehensive quality report"""
    print(f"\n{'='*60}")
    print("Comprehensive Data Quality Report")
    print(f"{'='*60}")
    
    if integrated_df is None:
        print("Integrated dataset not available")
        return
    
    # Basic statistics
    print(f"\nDataset Statistics:")
    print(f"  Total rows: {len(integrated_df):,}")
    print(f"  Total columns: {len(integrated_df.columns)}")
    print(f"  Memory usage: {integrated_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    print(f"\nMissing Values Summary:")
    total_cells = len(integrated_df) * len(integrated_df.columns)
    missing_cells = integrated_df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    print(f"  Total missing values: {missing_cells:,} ({missing_pct:.2f}%)")
    
    # Missing values by column
    missing_by_col = integrated_df.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    if len(missing_by_col) > 0:
        print(f"\n  Missing values by column:")
        for col, count in missing_by_col.items():
            pct = (count / len(integrated_df)) * 100
            print(f"    {col}: {count:,} ({pct:.2f}%)")
    
    # Geographic coverage
    if 'Country' in integrated_df.columns:
        print(f"\nGeographic Coverage:")
        unique_countries = integrated_df['Country'].unique()
        regions = [c for c in unique_countries if is_region(c)]
        countries = [c for c in unique_countries if not is_region(c)]
        print(f"  Total unique areas: {len(unique_countries)}")
        print(f"  Individual countries: {len(countries)}")
        print(f"  Regional aggregations: {len(regions)}")
    
    # Temporal coverage
    if 'Year' in integrated_df.columns:
        print(f"\nTemporal Coverage:")
        print(f"  Year range: {integrated_df['Year'].min()} - {integrated_df['Year'].max()}")
        print(f"  Unique years: {integrated_df['Year'].nunique()}")
        print(f"  Years covered: {sorted(integrated_df['Year'].unique())}")
    
    # Nutrient coverage
    if 'Nutrient_Type' in integrated_df.columns:
        print(f"\nNutrient Coverage:")
        print(f"  Unique nutrient types: {integrated_df['Nutrient_Type'].nunique()}")
        print(f"  Nutrient types: {', '.join(sorted(integrated_df['Nutrient_Type'].unique()))}")
        
        # Nutrient coverage statistics
        nutrient_counts = integrated_df['Nutrient_Type'].value_counts()
        print(f"\n  Records per nutrient type:")
        for nutrient, count in nutrient_counts.items():
            print(f"    {nutrient}: {count:,} records")
    
    # Food items coverage
    if 'Food_Item' in integrated_df.columns:
        print(f"\nFood Items Coverage:")
        unique_items = integrated_df['Food_Item'].nunique()
        print(f"  Unique food items: {unique_items}")
        print(f"  Top 10 food items by frequency:")
        top_items = integrated_df['Food_Item'].value_counts().head(10)
        for item, count in top_items.items():
            print(f"    {item}: {count:,} records")


def main():
    """Main analysis function"""
    print("=" * 60)
    print("Enhanced Dataset Analysis and Quality Assessment")
    print("=" * 60)
    
    # Try to load integrated dataset
    integrated_path = PROCESSED_DATA_DIR / "integrated_nutrition_data.csv"
    
    if integrated_path.exists():
        print(f"\nLoading integrated dataset from {integrated_path}...")
        try:
            integrated_df = pd.read_csv(integrated_path, low_memory=False)
            print(f"Loaded {len(integrated_df):,} rows")
            
            # Run enhanced analyses
            analyze_data_quality(integrated_df, "Integrated Dataset")
            analyze_data_completeness(integrated_df)
            analyze_nutrient_distribution(integrated_df)
            check_research_question_feasibility(integrated_df)
            generate_quality_report(integrated_df)
            
        except Exception as e:
            print(f"Error loading integrated dataset: {e}")
            import traceback
            traceback.print_exc()
            print("\nPlease run integrate_datasets.py first to create the integrated dataset.")
    else:
        print(f"\nIntegrated dataset not found at {integrated_path}")
        print("Please run integrate_datasets.py first to create the integrated dataset.")
    
    # Try to analyze raw datasets
    print("\n" + "=" * 60)
    print("Analyzing Raw Datasets")
    print("=" * 60)
    
    # FoodBalanceSheet
    fbs_path = RAW_DATA_DIR / "FoodBalanceSheet_data" / "FoodBalanceSheets_E_All_Data_(Normalized).csv"
    if fbs_path.exists():
        try:
            fbs_df = pd.read_csv(fbs_path, low_memory=False, nrows=10000)  # Sample for analysis
            analyze_data_quality(fbs_df, "FoodBalanceSheet (Sample)")
        except Exception as e:
            print(f"Error loading FoodBalanceSheet: {e}")
    else:
        print(f"FoodBalanceSheet not found at {fbs_path}")
    
    # Population
    pop_path = RAW_DATA_DIR / "Population_data" / "Population_E_All_Area_Groups_NOFLAG.csv"
    if pop_path.exists():
        try:
            pop_df = pd.read_csv(pop_path, low_memory=False)
            analyze_data_quality(pop_df, "Population")
        except Exception as e:
            print(f"Error loading Population: {e}")
    else:
        print(f"Population data not found at {pop_path}")
    
    # FoodSecurity
    fsec_path = RAW_DATA_DIR / "FoodSecurity_data" / "Food_Security_Data_E_All_Data_(Normalized).csv"
    if fsec_path.exists():
        try:
            fsec_df = pd.read_csv(fsec_path, low_memory=False, nrows=10000)  # Sample for analysis
            analyze_data_quality(fsec_df, "FoodSecurity (Sample)")
        except Exception as e:
            print(f"Error loading FoodSecurity: {e}")
    else:
        print(f"FoodSecurity data not found at {fsec_path}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
