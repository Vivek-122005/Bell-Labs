# Nutrition & Obesity Trends Analysis

A data analysis project examining global nutrition patterns, dietary trends, and their relationship with obesity rates across countries.

## 📋 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Run Data Processing Pipeline

```bash
# Run the complete pipeline (preprocesses all data and creates master panel)
python run_pipeline.py
```

This will:
- ✅ Preprocess FAO nutrition data
- ✅ Preprocess obesity data  
- ✅ Create food group mappings
- ✅ Create panel datasets
- ✅ Create final master panel

**Output**: `data/processed/final/master_panel_final.csv`

### 3. Run Analysis

```bash
# Exploratory Data Analysis
python scripts/analysis/perform_eda.py

# Create interactive visualizations
python scripts/analysis/interactive_plot.py
```

### 4. Explore in Jupyter

**Step 0: Understand raw data** (recommended first):
```bash
jupyter notebook notebooks/00_raw_data_exploration.ipynb
```

**Main analysis notebook**:
```bash
jupyter notebook notebooks/01_eda_visualization.ipynb
```

**Note**: See `notebooks/README.md` for complete notebook guide.

## 📁 Project Structure

```
Bell labs/
│
├── run_pipeline.py              # Main pipeline script - run this first!
│
├── data/
│   ├── raw/                     # Raw data files (FAO, WHO datasets)
│   │   ├── FoodBalanceSheet_data/
│   │   ├── Population_data/
│   │   └── data.csv            # Obesity data
│   │
│   └── processed/              # Cleaned and processed data (organized)
│       ├── cleaned/            # Step 1-3: Cleaned raw data
│       │   ├── Cleaned_FAO_Nutrients.csv
│       │   ├── Cleaned_FAO_Population.csv
│       │   └── Cleaned_Obesity.csv
│       ├── mappings/           # Step 3: Mapping files
│       │   └── Item_to_FoodGroup.csv
│       ├── panels/             # Step 4: Intermediate panels
│       └── final/              # Step 5: Final dataset ⭐
│           └── master_panel_final.csv
│
├── scripts/                     # Processing scripts (organized by purpose)
│   ├── preprocessing/          # Step 1-3: Data preprocessing
│   │   ├── preprocess_fao_data.py
│   │   ├── preprocess_obesity_data.py
│   │   └── preprocess_food_group_mapping.py
│   ├── panels/                 # Step 4-5: Panel dataset creation
│   │   ├── create_panel_datasets.py
│   │   └── create_master_panel.py
│   └── analysis/               # Step 6+: Analysis and visualization
│       ├── perform_eda.py
│       ├── extended_eda.py
│       └── interactive_plot.py
│
├── notebooks/                   # Jupyter notebooks for exploration
│   └── 02_eda_visualization.ipynb      # Main analysis notebook
│
├── doc/                         # Documentation (organized)
│   ├── README.md               # Documentation guide
│   ├── guides/                 # How-to guides
│   │   └── methodology.md     # Detailed methodology
│   ├── reference/              # Reference docs
│   │   ├── data_dictionary.md  # Variable descriptions
│   │   ├── data_analysis.md    # Dataset analysis
│   │   └── dataset_analysis.md # Alternative analysis
│   └── notes/                  # Research notes
│       └── research_notes.md   # Research findings
│   └── reseach_notes.md
│
└── requirements.txt             # Python dependencies
```

## 🔄 Data Processing Workflow

### Pipeline Steps

1. **Preprocess FAO Data** (`preprocess_fao_data.py`)
   - Cleans Food Balance Sheet data
   - Extracts nutrients (energy, protein, fat)
   - Extracts population data
   - Output: `Cleaned_FAO_Nutrients.csv`, `Cleaned_FAO_Population.csv`

2. **Preprocess Obesity Data** (`preprocess_obesity_data.py`)
   - Cleans WHO obesity dataset
   - Standardizes country names
   - Output: `Cleaned_Obesity.csv`

3. **Create Food Group Mapping** (`preprocess_food_group_mapping.py`)
   - Maps FAO items to food groups (Cereals, Meat, Dairy, etc.)
   - Output: `Item_to_FoodGroup.csv`

4. **Create Panel Datasets** (`create_panel_datasets.py`)
   - Creates country-year panels for nutrients
   - Aggregates food groups by country-year
   - Output: `nutrient_panel.csv`, `foodgroup_energy_panel.csv`, etc.

5. **Create Master Panel** (`create_master_panel.py`)
   - Merges all datasets into final panel
   - Handles missing data
   - Output: `master_panel_final.csv` ⭐

### Running Individual Steps

If you need to run steps individually:

```bash
python scripts/preprocess_fao_data.py
python scripts/preprocess_obesity_data.py
python scripts/preprocess_food_group_mapping.py
python scripts/create_panel_datasets.py
python scripts/create_master_panel.py
```

## 📊 Final Dataset

**File**: `data/processed/final/master_panel_final.csv`

**Structure**: Country-year panel (171 countries, 2010-2022)

**Key Variables**:
- `country`, `year`: Identifiers
- `energy_kcal_day`, `protein_g_day`, `fat_g_day`: Nutrients (per capita/day)
- `Cereals`, `Meat`, `Dairy & Eggs`, etc.: Food group energy (kcal/capita/day)
- `Cereals_share`, `Meat_share`, etc.: Food group shares (%)
- `population`: Total population
- `obesity_pct`: Obesity prevalence (%)

See `data/processed/README.md` for detailed variable descriptions.

## 🔬 Analysis

### Exploratory Data Analysis

```bash
python scripts/perform_eda.py
```

Generates:
- Summary statistics
- Correlation matrices
- Trend visualizations
- Outputs saved to `data/outputs/`

### Interactive Visualizations

```bash
python scripts/interactive_plot.py
```

Creates interactive Plotly charts for:
- Energy vs Obesity trends
- Food group shares over time
- Country comparisons

### Jupyter Notebooks

Open `notebooks/02_eda_visualization.ipynb` for interactive exploration.

## 📚 Documentation

- **Documentation Guide**: `doc/README.md` - Overview of all documentation
- **Methodology**: `doc/guides/methodology.md` - Detailed methodology
- **Data Dictionary**: `doc/reference/data_dictionary.md` - Variable descriptions
- **Research Notes**: `doc/notes/research_notes.md` - Research findings
- **Processed Data README**: `data/processed/README.md` - Dataset documentation

## 🛠️ Requirements

- Python 3.8+
- See `requirements.txt` for package list

Main dependencies:
- pandas, numpy
- matplotlib, seaborn, plotly
- jupyter
- scikit-learn

## 📝 Notes

- **Data Sources**: FAO Food Balance Sheets, WHO Global Health Observatory
- **Year Coverage**: 2010-2022 (common years across datasets)
- **Country Coverage**: 171 countries
- **Missing Data**: Handled via interpolation (max 2-year gaps)

## 🚀 Next Steps

After running the pipeline:

1. **Explore the data**: Open `notebooks/01_eda_visualization.ipynb`
2. **Run EDA**: `python scripts/analysis/perform_eda.py`
3. **Create visualizations**: `python scripts/analysis/interactive_plot.py`
4. **Build models**: Use `data/processed/final/master_panel_final.csv` for regression/ML analysis

## ❓ Troubleshooting

**Issue**: `ModuleNotFoundError`
- **Solution**: Activate virtual environment: `source venv/bin/activate`

**Issue**: Missing raw data files
- **Solution**: Ensure data files are in `data/raw/` directory

**Issue**: Pipeline fails at a step
- **Solution**: Check error message, fix the issue, and re-run from that step

---

**Last Updated**: 2025-01-20
