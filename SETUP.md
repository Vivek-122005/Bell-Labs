# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Activate Environment
```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 2: Run Pipeline
```bash
python run_pipeline.py
```
**Output**: `data/processed/master_panel_final.csv` ⭐

### Step 3: Run Analysis
```bash
python scripts/perform_eda.py
```

## 📊 Use the Data

**Final Dataset**: `data/processed/final/master_panel_final.csv`

- 171 countries, 2010-2022
- Variables: nutrients, food groups, obesity, population
- Ready for analysis!

## 📚 Documentation

- **README.md** - Full project overview
- **PROJECT_GUIDE.md** - How the pipeline works
- **CLEANUP_SUMMARY.md** - What was cleaned up

## 🔧 Common Commands

```bash
# Run complete pipeline
python run_pipeline.py

# Run individual step
python scripts/preprocessing/preprocess_fao_data.py

# Run EDA
python scripts/analysis/perform_eda.py

# Create visualizations
python scripts/analysis/interactive_plot.py

# Clean up intermediate files (optional)
python cleanup_intermediate_files.py

# Open notebook
jupyter notebook notebooks/02_eda_visualization.ipynb
```

## 📁 Key Files

- `run_pipeline.py` - Main pipeline script
- `master_panel_final.csv` - Final dataset for analysis
- `scripts/perform_eda.py` - Analysis script
- `notebooks/00_raw_data_exploration.ipynb` - **Start here!** Explore raw data
- `notebooks/01_eda_visualization.ipynb` - Main analysis notebook
- `notebooks/README.md` - Notebook guide

---

**That's it! You're ready to go.** 🎉

