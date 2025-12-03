# Predictive Analysis Output Files

This directory contains outputs from the complete predictive analysis of obesity ~ diet composition.

## Analysis Overview

Three main analyses were performed:

### (A) Simple Linear Regression: Obesity ~ FatShare
- **Model**: OLS regression of obesity prevalence on fat share (% of total energy)
- **Files**:
  - `simple_lr_summary.txt`: Full model summary and interpretation
  - `simple_lr_predictions.csv`: Predictions and residuals for all observations
  - `simple_lr_residuals.csv`: Residuals only
  - `plots/simple_lr_scatter_fit.png`: Scatter plot with fitted line
  - `plots/simple_lr_resid_vs_fitted.png`: Residuals vs fitted values
  - `plots/simple_lr_qq.png`: Q-Q plot of residuals

### (B) Multiple Regression: Obesity ~ Diet Composition
- **Model**: OLS regression with multiple predictors (fat_share, energy_kcal_day, sugar_share, oil_fats_share, meat_share)
- **Files**:
  - `multi_lr_summary.txt`: Full model summary and interpretation
  - `output_multi_lr_betas.csv`: Coefficients, standardized coefficients, p-values, and VIF
  - `output_multi_lr_test_metrics.json`: Test set performance metrics (R², RMSE, MAE)
  - `plots/multi_lr_obs_vs_pred.png`: Observed vs predicted scatter plot (test set)

### (C) Global Trend Forecast: 2010-2030
- **Model**: Linear trend extrapolation of global population-weighted obesity average
- **Files**:
  - `global_trend_summary.txt`: Model summary and forecast interpretation
  - `output_obesity_forecast_2010_2030.csv`: Forecast table with 95% prediction intervals
  - `plots/global_obesity_forecast_2030.png`: Visualization of trend and forecast

## Data Source

- **Input**: `data/processed/final/master_panel_final.csv`
- **Years**: 2010-2022 (filtered from full dataset)
- **Countries**: 169 countries

## Key Findings

### Simple Regression
- Fat share shows positive association with obesity
- R² = 0.306
- RMSE = 9.53%

### Multiple Regression
- Test R² = 0.356
- Test RMSE = 8.93%
- Strongest predictor: energy_kcal_day

### Global Forecast
- 2030 projection: 21.5% (95% PI: 21.3% - 21.6%)
- Weighting: population-weighted

## Notes

- All models use OLS (Ordinary Least Squares) from statsmodels
- Train/test split for multiple regression: 80/20 (random_state=42)
- Forecast assumes linear trend continuation (no structural breaks)
- Ecological analysis - correlations do not imply causation

## Fallbacks Applied

- Used Sugar_share as sugar_share
- Used Oils & Fats_share as oil_fats_share
- Used Meat_share as meat_share


Generated: 2025-12-03 17:22:18
