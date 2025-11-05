# Nutrition and Obesity Trends

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Datasets](#datasets)
- [Research Questions](#research-questions)
- [Project Objectives](#project-objectives)
- [Methodology](#methodology)
- [Analysis Philosophy](#analysis-philosophy)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Interactive Visualization](#interactive-visualization)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

## Overview

Nutritional habits are deeply linked to public health outcomes, particularly obesity, diabetes, and cardiovascular disease. Over the past 50 years, globalization and industrialization of food production have fundamentally changed how and what we eat. This project investigates how diets are shifting globally, which countries face the fastest growth in obesity, and how dietary diversity, food affordability, and economic status shape nutritional health.

## Problem Statement

Global nutrition patterns have undergone significant transformations due to:
- **Globalization of food production**: Industrialized food systems have expanded across borders
- **Dietary shifts**: Traditional diets are being replaced by processed, energy-dense foods
- **Health implications**: Rising rates of obesity, diabetes, and cardiovascular disease worldwide
- **Economic factors**: Income levels influence both calorie intake and nutritional quality
- **Regional variations**: Different countries and regions exhibit distinct dietary patterns and health outcomes

This project aims to analyze these trends using global nutrition and health datasets to understand the relationship between dietary patterns, economic factors, and public health outcomes.

## Datasets

### Primary Data Source

**Kaggle Dataset**
- **Platform**: Kaggle
- **Method**: Data is fetched directly from Kaggle using the Kaggle API
- **Contains**: 
  - Nutrition and dietary consumption data (originally from FAO)
  - Obesity statistics and health outcomes (originally from USDA)
  - Food balance sheets (food supply, consumption, and utilization)
  - Nutritional indicators (protein, fat, calories, sugar, fiber, etc.) by country and year
  - Population data for per capita calculations
  - Health outcomes (obesity rates, diabetes prevalence)

> **Note**: The datasets hosted on Kaggle are derived from:
> - **FAO (Food and Agriculture Organization)**: https://www.fao.org/faostat/en/#home
> - **USDA Economic Research Service**: https://www.ers.usda.gov/topics/food-choices-health/obesity

### Additional Data Sources (Optional)

- World Bank: Economic indicators (GDP per capita, income levels)
- World Health Organization (WHO): Global health statistics (diabetes prevalence, obesity rates)
- Our World in Data: Health and nutrition visualizations and datasets

## Research Questions

### Exploratory Questions

1. **Dietary Patterns by Region**
   - Which countries consume the highest levels of sugar, fat, protein, and fiber?
   - Are there distinct regional patterns (e.g., Western diets vs. Asian diets)?

2. **Health-Diet Correlation**
   - How do obesity and diabetes rates correlate with national dietary consumption patterns?
   - Are there specific nutrients or food groups that show stronger associations with health outcomes?

3. **Dietary Homogenization**
   - Are global diets becoming more homogenous (same foods across cultures)?
   - Do distinct regional diets persist despite globalization?

4. **Temporal Changes in Food Consumption**
   - Which food groups (processed foods, dairy, cereals, fruits/vegetables) have changed most in consumption over the past 50 years?
   - What are the trends in nutritional diversity over time?

5. **Economic Factors and Nutrition Quality**
   - Does income predict not just calorie intake but also nutritional quality?
   - How does economic status relate to nutritional diversity and the balance between processed and fresh foods?

## Project Objectives

1. **Data Processing**
   - Aggregate nutrient consumption levels on yearly granularity for each country
   - Calculate per capita consumption rates (total consumption / population) for each nutrient, country, and year
   - Clean and integrate data from multiple sources

2. **Interactive Visualization**
   - Create an interactive bar plot allowing users to:
     - Select a country
     - Select a nutrient (protein, fat, sugar, fiber, calories, etc.)
     - View consumption levels across years for the selected country and nutrient

3. **Predictive and Descriptive Analysis**
   - Perform descriptive statistics and exploratory data analysis
   - Build predictive models for obesity trends and health outcomes
   - Identify correlations and patterns in the data
   - Document analysis philosophy, design, and methodology

## Methodology

### Data Processing Pipeline

1. **Data Collection**
   - Fetch datasets from Kaggle using the Kaggle API
   - Extract relevant variables (nutrient consumption, population, health outcomes)
   - Load data into pandas DataFrames for analysis

2. **Data Cleaning**
   - Handle missing values
   - Standardize country names and codes
   - Align temporal coverage (years)
   - Validate data consistency

3. **Feature Engineering**
   - Calculate per capita consumption: `Consumption per capita = Total Nutrient Consumption / Population`
   - Create derived metrics:
     - Nutritional diversity index
     - Processed vs. fresh food ratio
     - Macronutrient balance (protein:fat:carbohydrate ratio)

4. **Data Integration**
   - Merge nutrition data with population data
   - Integrate health outcomes (obesity, diabetes rates)
   - Combine economic indicators (GDP per capita) if available

### Analysis Approach

1. **Descriptive Analysis**
   - Summary statistics by country and region
   - Temporal trends analysis
   - Regional comparisons
   - Cross-sectional analysis by income level

2. **Predictive Analysis**
   - Time series forecasting for obesity trends
   - Regression models predicting health outcomes from dietary patterns
   - Clustering analysis to identify dietary pattern groups

3. **Correlation Analysis**
   - Correlation matrices between nutrients and health outcomes
   - Spatial correlation analysis
   - Temporal correlation with economic indicators

## Analysis Philosophy

### Design Principles

1. **Evidence-Based Approach**: All conclusions are grounded in statistical analysis and validated against multiple data sources
2. **Transparency**: All data processing steps, assumptions, and limitations are documented
3. **Reproducibility**: Code and methodology are structured to enable replication of results
4. **Comprehensive Coverage**: Analysis spans multiple dimensions (temporal, spatial, nutritional, economic)

### Methodological Considerations

- **Causality vs. Correlation**: We acknowledge that correlation does not imply causation; dietary patterns may be confounded by other factors
- **Data Quality**: We account for varying data quality across countries and time periods
- **Missing Data**: We employ appropriate strategies for handling missing values (imputation, exclusion, or sensitivity analysis)
- **Bias**: We recognize potential biases in data collection (underreporting, measurement errors) and discuss their implications

## Features

### Core Functionality

- ✅ Per capita nutrient consumption calculation
- ✅ Interactive visualization (country and nutrient selection)
- ✅ Temporal trend analysis
- ✅ Regional pattern identification
- ✅ Health-diet correlation analysis
- ✅ Predictive modeling for obesity trends

### Interactive Components

- **Country Selection**: Dropdown or searchable list of all countries in the dataset
- **Nutrient Selection**: Choose from available nutrients (protein, fat, sugar, fiber, calories, etc.)
- **Visualization**: Interactive bar plot showing consumption levels across years
- **Additional Features** (optional):
  - Comparison mode (multiple countries)
  - Regional aggregation view
  - Export functionality for charts and data

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Bell labs"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages

The project uses the following Python libraries:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Static plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive visualizations
- `jupyter` - Interactive notebooks
- `scikit-learn` - Machine learning models
- `kaggle` - Kaggle API client for downloading datasets
- `requests` - API/data downloading (if needed)
- `openpyxl` - Excel file handling (for data processing)

### Kaggle API Setup

To download datasets from Kaggle, you need to:

1. **Create a Kaggle account** (if you don't have one)
   - Visit: https://www.kaggle.com/

2. **Get your API credentials**
   - Go to your Kaggle account settings: https://www.kaggle.com/settings
   - Scroll to the "API" section
   - Click "Create New Token" to download `kaggle.json`

3. **Configure Kaggle credentials**
   - Place `kaggle.json` in your home directory:
     - **Linux/Mac**: `~/.kaggle/kaggle.json`
     - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - Set proper permissions:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

4. **Verify installation**
   ```bash
   kaggle datasets list
   ```

## Usage

### Data Preparation

1. **Setup Kaggle API** (if not already done)
   - Ensure `kaggle.json` is in `~/.kaggle/` directory
   - Verify API credentials are working

2. **Download datasets from Kaggle**
   ```bash
   python scripts/download_data.py
   ```
   This script uses the Kaggle API to download the nutrition and obesity datasets.

3. **Process and clean data**
   ```bash
   python scripts/process_data.py
   ```

### Running Analysis

1. **Exploratory Data Analysis**
   ```bash
   jupyter notebook notebooks/01_exploratory_analysis.ipynb
   ```

2. **Interactive Visualization**
   ```bash
   python scripts/interactive_plot.py
   ```
   Or use the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/02_interactive_visualization.ipynb
   ```

3. **Predictive Analysis**
   ```bash
   jupyter notebook notebooks/03_predictive_analysis.ipynb
   ```

### Generating Reports

```bash
python scripts/generate_report.py
```

## Project Structure

```
Bell labs/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore file
│
├── data/                       # Data directory
│   ├── raw/                    # Raw datasets (downloaded)
│   ├── processed/              # Cleaned and processed data
│   └── external/               # External data sources
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_interactive_visualization.ipynb
│   ├── 03_predictive_analysis.ipynb
│   └── 04_report_generation.ipynb
│
├── scripts/                    # Python scripts
│   ├── download_data.py        # Kaggle data download script
│   ├── process_data.py         # Data processing pipeline
│   ├── interactive_plot.py     # Interactive visualization
│   ├── analysis.py             # Core analysis functions
│   └── generate_report.py      # Report generation
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   ├── data_processor.py       # Data processing functions
│   ├── visualizer.py           # Visualization functions
│   ├── analyzer.py             # Analysis functions
│   └── models.py               # Predictive models
│
├── reports/                     # Generated reports and visualizations
│   ├── figures/                # Saved plots
│   └── analysis_report.md      # Final analysis report
│
└── docs/                       # Documentation
    ├── methodology.md          # Detailed methodology
    └── data_dictionary.md      # Data variable descriptions
```

## Key Findings

*This section will be populated with results after analysis completion.*

### Preliminary Findings (To be updated)

- **Dietary Patterns**: [To be filled]
- **Health Correlations**: [To be filled]
- **Temporal Trends**: [To be filled]
- **Regional Differences**: [To be filled]
- **Economic Factors**: [To be filled]

## Interactive Visualization

The interactive bar plot allows users to:

1. **Select a Country**: Choose from a dropdown list of all countries in the dataset
2. **Select a Nutrient**: Choose from available nutrients (protein, fat, sugar, fiber, calories, etc.)
3. **View Trends**: Display consumption levels across years for the selected combination

### Example Usage

```python
from scripts.interactive_plot import create_interactive_plot

# Create and display interactive plot
fig = create_interactive_plot()
fig.show()
```

The visualization includes:
- Hover tooltips showing exact values
- Zoom and pan capabilities
- Export options for images
- Comparison mode for multiple countries (optional)

## Future Work

### Potential Enhancements

1. **Expanded Data Sources**
   - Include more recent years as data becomes available
   - Integrate subnational data for larger countries
   - Add food security and hunger indicators

2. **Advanced Analytics**
   - Machine learning models for obesity prediction
   - Time series forecasting for future trends
   - Causal inference analysis

3. **Enhanced Visualizations**
   - Interactive world maps showing regional patterns
   - Network analysis of food trade and consumption
   - Animated time-lapse visualizations

4. **User Interface**
   - Web application (Flask/Dash/Streamlit)
   - Real-time data updates
   - Customizable dashboard

5. **Policy Analysis**
   - Impact assessment of nutrition policies
   - Scenario modeling for dietary interventions
   - Cost-benefit analysis of public health programs

## Contributors

- [Your Name/Team] - Project Lead

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Kaggle** for hosting and providing easy access to the datasets
- **FAO (Food and Agriculture Organization)** for providing comprehensive global nutrition datasets
- **USDA Economic Research Service** for obesity and food consumption statistics
- **Open Source Community** for the tools and libraries that made this project possible


---

**Note**: This project is part of an academic research initiative. All findings should be interpreted with consideration of data limitations and methodological constraints.

