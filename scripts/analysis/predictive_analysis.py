"""
Complete Predictive Analysis: Obesity ~ Diet Composition
Performs:
(A) Simple linear regression: Obesity ~ FatShare
(B) Multiple regression with diet composition covariates
(C) Linear trend forecast of global obesity to 2030
"""

try:
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import json
    from pathlib import Path
    import warnings
    warnings.filterwarnings('ignore')
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("\nPlease install required packages:")
    print("  pip install pandas numpy statsmodels scikit-learn matplotlib seaborn scipy")
    print("\nOr install from requirements.txt:")
    print("  pip install -r requirements.txt")
    print("\nNote: You may also need to install statsmodels separately:")
    print("  pip install statsmodels")
    import sys
    sys.exit(1)

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# STEP 0: SETUP PATHS AND LOAD DATA
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "final"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
OUTPUT_PLOTS = OUTPUT_DIR / "plots"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PREDICTIVE ANALYSIS: OBESITY ~ DIET COMPOSITION")
print("="*70)

# Load data
print("\n📂 Loading master panel data...")
df = pd.read_csv(DATA_DIR / "master_panel_final.csv")
print(f"   ✅ Loaded {len(df):,} rows, {len(df.columns)} columns")

# Basic checks
print(f"\n📊 Dataset Overview:")
print(f"   • Total rows: {len(df):,}")
print(f"   • Unique countries: {df['country'].nunique()}")
print(f"   • Year range: {int(df['year'].min())} - {int(df['year'].max())}")
print(f"   • Columns: {list(df.columns)[:10]}... ({len(df.columns)} total)")

# Ensure correct data types
df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
df['obesity_pct'] = pd.to_numeric(df['obesity_pct'], errors='coerce')
df['energy_kcal_day'] = pd.to_numeric(df['energy_kcal_day'], errors='coerce')

# Filter to 2010-2022
print(f"\n🔍 Filtering to years 2010-2022...")
initial_count = len(df)
df = df[(df['year'] >= 2010) & (df['year'] <= 2022)].copy()
print(f"   ✅ Filtered to {len(df):,} rows (removed {initial_count - len(df):,} rows)")

# Calculate fat_share if not present
if 'fat_share' not in df.columns:
    print("\n📐 Calculating fat_share from fat_g_day and energy_kcal_day...")
    df['fat_share'] = (df['fat_g_day'] * 9) / df['energy_kcal_day'] * 100
    print(f"   ✅ Calculated fat_share for {df['fat_share'].notna().sum():,} rows")

# Calculate sugar_share if not present (use Sugar_share if available, else calculate)
if 'sugar_share' not in df.columns:
    if 'Sugar_share' in df.columns:
        print("\n📐 Using Sugar_share as sugar_share...")
        df['sugar_share'] = df['Sugar_share']
    elif 'sugar_g_day' in df.columns:
        print("\n📐 Calculating sugar_share from sugar_g_day and energy_kcal_day...")
        df['sugar_share'] = (df['sugar_g_day'] * 4) / df['energy_kcal_day'] * 100
    else:
        print("\n⚠️  Warning: sugar_share not available and cannot be calculated")
        df['sugar_share'] = np.nan

# Map food group shares
if 'oil_fats_share' not in df.columns and 'Oils & Fats_share' in df.columns:
    df['oil_fats_share'] = df['Oils & Fats_share']
elif 'oil_fats_share' not in df.columns:
    df['oil_fats_share'] = np.nan

if 'meat_share' not in df.columns and 'Meat_share' in df.columns:
    df['meat_share'] = df['Meat_share']
elif 'meat_share' not in df.columns:
    df['meat_share'] = np.nan

# Summary statistics
print("\n📈 Summary Statistics (2010-2022):")
summary_vars = ['obesity_pct', 'fat_share', 'energy_kcal_day', 'sugar_share', 'oil_fats_share', 'meat_share']
summary_stats = {}
for var in summary_vars:
    if var in df.columns:
        stats_dict = {
            'mean': df[var].mean(),
            'std': df[var].std(),
            'min': df[var].min(),
            'max': df[var].max()
        }
        summary_stats[var] = stats_dict
        print(f"   {var:20s}: mean={stats_dict['mean']:.2f}, std={stats_dict['std']:.2f}, "
              f"range=[{stats_dict['min']:.2f}, {stats_dict['max']:.2f}]")

# ============================================================================
# STEP 1: SIMPLE LINEAR REGRESSION - Obesity ~ FatShare
# ============================================================================

print("\n" + "="*70)
print("STEP 1: SIMPLE LINEAR REGRESSION - Obesity ~ FatShare")
print("="*70)

# Build dataset
df_lr = df[['country', 'year', 'obesity_pct', 'fat_share']].copy()
df_lr = df_lr.dropna(subset=['obesity_pct', 'fat_share'])

print(f"\n📊 Simple LR Dataset:")
print(f"   • Observations (N): {len(df_lr):,}")
print(f"   • Unique countries: {df_lr['country'].nunique()}")

# Prepare data
X = df_lr['fat_share'].values
y = df_lr['obesity_pct'].values
X_const = sm.add_constant(X)

# Exploratory plot
print("\n📊 Creating exploratory scatter plot...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y, alpha=0.5, s=20, color='steelblue', edgecolors='black', linewidth=0.5)

# Calculate correlation
corr_coef, p_value_corr = stats.pearsonr(X, y)

# Fit OLS for plot line
z = np.polyfit(X, y, 1)
p = np.poly1d(z)
x_line = np.linspace(X.min(), X.max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'OLS fit: y = {z[0]:.3f}x + {z[1]:.3f}')

ax.set_xlabel('Fat Share (% of total energy)', fontsize=12, fontweight='bold')
ax.set_ylabel('Obesity Prevalence (%)', fontsize=12, fontweight='bold')
ax.set_title('Simple Linear Regression: Obesity ~ FatShare', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Annotate correlation
textstr = f'r = {corr_coef:.3f}\np = {p_value_corr:.3g}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(OUTPUT_PLOTS / "simple_lr_scatter_fit.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_PLOTS / 'simple_lr_scatter_fit.png'}")
plt.close()

# Fit OLS model
print("\n🔧 Fitting OLS model...")
model_simple = sm.OLS(y, X_const).fit()

# Save summary
summary_text = str(model_simple.summary())
with open(OUTPUT_DIR / "simple_lr_summary.txt", 'w') as f:
    f.write("="*70 + "\n")
    f.write("SIMPLE LINEAR REGRESSION: Obesity ~ FatShare\n")
    f.write("="*70 + "\n\n")
    f.write(summary_text)
    f.write("\n\n" + "="*70 + "\n")
    f.write("INTERPRETATION\n")
    f.write("="*70 + "\n\n")

# Extract coefficients
coef = model_simple.params[1]
std_err = model_simple.bse[1]
t_stat = model_simple.tvalues[1]
p_value = model_simple.pvalues[1]
r_squared = model_simple.rsquared
rmse = np.sqrt(model_simple.mse_resid)
mae = np.mean(np.abs(model_simple.resid))

interpretation = (
    f"Coefficient b = {coef:.3f} (SE={std_err:.3f}, t={t_stat:.3f}, p={p_value:.3g}); "
    f"R² = {r_squared:.3f}.\n\n"
    f"Interpretation: Each 1 percentage-point increase in fat_share is associated with "
    f"an estimated {coef:.3f} percentage-point increase in obesity prevalence.\n\n"
    f"Model Performance:\n"
    f"  • RMSE: {rmse:.3f}%\n"
    f"  • MAE: {mae:.3f}%\n"
    f"  • Pearson correlation: r = {corr_coef:.3f} (p = {p_value_corr:.3g})\n"
)

with open(OUTPUT_DIR / "simple_lr_summary.txt", 'a') as f:
    f.write(interpretation)

print(interpretation)

# Diagnostics
print("\n📊 Creating diagnostic plots...")

# Residuals vs Fitted
fitted = model_simple.fittedvalues
residuals = model_simple.resid

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(fitted, residuals, alpha=0.5, s=20, color='steelblue', edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Fitted Values (Predicted Obesity %)', fontsize=12, fontweight='bold')
ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax.set_title('Residuals vs Fitted Values', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PLOTS / "simple_lr_resid_vs_fitted.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_PLOTS / 'simple_lr_resid_vs_fitted.png'}")
plt.close()

# QQ plot
fig, ax = plt.subplots(figsize=(8, 8))
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot of Residuals', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PLOTS / "simple_lr_qq.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_PLOTS / 'simple_lr_qq.png'}")
plt.close()

# Save predictions and residuals
df_lr['obesity_pct_predicted'] = fitted
df_lr['residual'] = residuals
df_lr_output = df_lr[['country', 'year', 'obesity_pct', 'obesity_pct_predicted', 'residual']].copy()
df_lr_output.columns = ['country', 'year', 'obesity_pct_actual', 'obesity_pct_predicted', 'residual']
df_lr_output.to_csv(OUTPUT_DIR / "simple_lr_predictions.csv", index=False)
print(f"   ✅ Saved: {OUTPUT_DIR / 'simple_lr_predictions.csv'}")

df_lr_output[['country', 'year', 'residual']].to_csv(OUTPUT_DIR / "simple_lr_residuals.csv", index=False)
print(f"   ✅ Saved: {OUTPUT_DIR / 'simple_lr_residuals.csv'}")

# ============================================================================
# STEP 2: MULTIPLE REGRESSION
# ============================================================================

print("\n" + "="*70)
print("STEP 2: MULTIPLE REGRESSION - Obesity ~ Diet Composition")
print("="*70)

# Build dataset
predictors = ['fat_share', 'energy_kcal_day', 'sugar_share', 'oil_fats_share', 'meat_share']
df_multi = df[['country', 'year', 'obesity_pct'] + predictors].copy()

# Check which predictors are available
available_predictors = [p for p in predictors if p in df_multi.columns and df_multi[p].notna().sum() > 0]
missing_predictors = [p for p in predictors if p not in available_predictors]

print(f"\n📊 Multiple LR Dataset:")
print(f"   • Available predictors: {available_predictors}")
if missing_predictors:
    print(f"   • Missing predictors: {missing_predictors}")

# Drop rows with NaN in target or selected predictors
df_multi = df_multi.dropna(subset=['obesity_pct'] + available_predictors)
print(f"   • Observations (N): {len(df_multi):,}")
print(f"   • Unique countries: {df_multi['country'].nunique()}")

# Per-country coverage
country_years = df_multi.groupby('country').size()
print(f"   • Median years per country: {country_years.median():.1f}")
print(f"   • Years per country range: {country_years.min()} - {country_years.max()}")

# Prepare data
X_multi = df_multi[available_predictors].copy()
y_multi = df_multi['obesity_pct'].values

# Center energy_kcal_day for numerical stability
if 'energy_kcal_day' in X_multi.columns:
    energy_mean = X_multi['energy_kcal_day'].mean()
    X_multi['energy_kcal_day_centered'] = X_multi['energy_kcal_day'] - energy_mean
    # Use centered version for model
    X_multi = X_multi.drop('energy_kcal_day', axis=1)
    if 'energy_kcal_day_centered' in X_multi.columns:
        X_multi = X_multi.rename(columns={'energy_kcal_day_centered': 'energy_kcal_day'})
        # Update available_predictors
        if 'energy_kcal_day' in available_predictors:
            available_predictors = [p if p != 'energy_kcal_day' else 'energy_kcal_day' for p in available_predictors]

# Standardized version for beta coefficients
X_multi_std = (X_multi - X_multi.mean()) / X_multi.std()

# Train/test split
print("\n🔧 Splitting into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, shuffle=True
)
X_train_std, X_test_std, _, _ = train_test_split(
    X_multi_std, y_multi, test_size=0.2, random_state=42, shuffle=True
)

print(f"   • Training set: {len(X_train):,} observations")
print(f"   • Test set: {len(X_test):,} observations")

# Fit OLS on training set
print("\n🔧 Fitting multiple regression on training set...")
X_train_const = sm.add_constant(X_train)
model_multi = sm.OLS(y_train, X_train_const).fit()

# Save summary
summary_text_multi = str(model_multi.summary())
with open(OUTPUT_DIR / "multi_lr_summary.txt", 'w') as f:
    f.write("="*70 + "\n")
    f.write("MULTIPLE LINEAR REGRESSION: Obesity ~ Diet Composition\n")
    f.write("="*70 + "\n\n")
    f.write(f"Predictors: {', '.join(available_predictors)}\n")
    f.write(f"Training observations: {len(X_train):,}\n")
    f.write(f"Test observations: {len(X_test):,}\n\n")
    f.write(summary_text_multi)

# Evaluate on test set
print("\n📊 Evaluating on test set...")
X_test_const = sm.add_constant(X_test)
y_pred_test = model_multi.predict(X_test_const)

r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

test_metrics = {
    'r2_test': float(r2_test),
    'rmse_test': float(rmse_test),
    'mae_test': float(mae_test),
    'n_test': int(len(y_test))
}

with open(OUTPUT_DIR / "output_multi_lr_test_metrics.json", 'w') as f:
    json.dump(test_metrics, f, indent=2)

print(f"   • Test R²: {r2_test:.3f}")
print(f"   • Test RMSE: {rmse_test:.3f}%")
print(f"   • Test MAE: {mae_test:.3f}%")

# Observed vs Predicted plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_test, y_pred_test, alpha=0.6, s=30, color='steelblue', edgecolors='black', linewidth=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Observed Obesity (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Obesity (%)', fontsize=12, fontweight='bold')
ax.set_title(f'Observed vs Predicted (Test Set)\nR² = {r2_test:.3f}, RMSE = {rmse_test:.2f}%', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PLOTS / "multi_lr_obs_vs_pred.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_PLOTS / 'multi_lr_obs_vs_pred.png'}")
plt.close()

# Standardized coefficients (betas)
print("\n📊 Computing standardized coefficients and VIF...")
X_train_std_const = sm.add_constant(X_train_std)
model_multi_std = sm.OLS(y_train, X_train_std_const).fit()

# VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_train_const.columns[1:]  # Exclude constant
vif_data["VIF"] = [variance_inflation_factor(X_train_const.values, i+1) 
                   for i in range(len(X_train_const.columns)-1)]

# Beta coefficients
beta_data = pd.DataFrame({
    'variable': available_predictors,
    'coef': model_multi.params[1:].values,  # Exclude intercept
    'std_coef': model_multi_std.params[1:].values,  # Standardized
    'p_value': model_multi.pvalues[1:].values
})

# Merge with VIF
beta_data = beta_data.merge(vif_data, left_on='variable', right_on='Variable', how='left')
beta_data = beta_data.drop('Variable', axis=1)

beta_data.to_csv(OUTPUT_DIR / "output_multi_lr_betas.csv", index=False)
print(f"   ✅ Saved: {OUTPUT_DIR / 'output_multi_lr_betas.csv'}")

# Diagnostics: Cook's distance
print("\n📊 Computing Cook's distance...")
influence = model_multi.get_influence()
cooks_d = influence.cooks_distance[0]

# Top 10 influential observations
df_multi_train = df_multi.iloc[X_train.index].copy() if hasattr(X_train, 'index') else df_multi.iloc[:len(X_train)].copy()
df_multi_train['cooks_d'] = cooks_d
top_influential = df_multi_train.nlargest(10, 'cooks_d')[['country', 'year', 'cooks_d']]

print("\n   Top 10 Influential Observations:")
print(top_influential.to_string(index=False))

# Interpretation
print("\n📝 Model Interpretation:")
significant_vars = beta_data[beta_data['p_value'] < 0.05].sort_values('std_coef', key=abs, ascending=False)
if len(significant_vars) > 0:
    strongest = significant_vars.iloc[0]
    interpretation_multi = (
        f"\n{'='*70}\n"
        f"INTERPRETATION\n"
        f"{'='*70}\n\n"
        f"Strongest predictor: {strongest['variable']} (coef={strongest['coef']:.3f}, "
        f"std_coef={strongest['std_coef']:.3f}, p={strongest['p_value']:.3g})\n\n"
    )
    
    if 'fat_share' in significant_vars['variable'].values:
        fat_row = significant_vars[significant_vars['variable'] == 'fat_share'].iloc[0]
        interpretation_multi += (
            f"Fat_share remains {'a strong' if fat_row['p_value'] < 0.05 else 'not significant'} "
            f"predictor (coef={fat_row['coef']:.3f}, p={fat_row['p_value']:.3g}) after controlling "
            f"for other composition variables.\n"
        )
    
    if 'energy_kcal_day' in significant_vars['variable'].values:
        energy_row = significant_vars[significant_vars['variable'] == 'energy_kcal_day'].iloc[0]
        interpretation_multi += (
            f"Energy_kcal_day is {'significant' if energy_row['p_value'] < 0.05 else 'not significant'} "
            f"(coef={energy_row['coef']:.3f}, p={energy_row['p_value']:.3g}) after controlling "
            f"for composition variables.\n"
        )
    
    interpretation_multi += f"\nTest R² = {r2_test:.3f}\n"
    
    with open(OUTPUT_DIR / "multi_lr_summary.txt", 'a') as f:
        f.write(interpretation_multi)
    
    print(interpretation_multi)

# ============================================================================
# STEP 3: GLOBAL TREND FORECAST
# ============================================================================

print("\n" + "="*70)
print("STEP 3: GLOBAL TREND FORECAST TO 2030")
print("="*70)

# Build global dataset
df_global = df[['country', 'year', 'obesity_pct', 'population']].copy()
df_global = df_global.dropna(subset=['obesity_pct'])

# Compute yearly global average
print("\n📊 Computing global obesity averages...")
if 'population' in df_global.columns and df_global['population'].notna().sum() > 0:
    # Population-weighted average
    df_global_weighted = df_global.groupby('year').apply(
        lambda x: np.average(x['obesity_pct'], weights=x['population'])
    ).reset_index()
    df_global_weighted.columns = ['year', 'global_obesity_avg']
    
    # Total population per year
    pop_by_year = df_global.groupby('year')['population'].sum().reset_index()
    pop_by_year.columns = ['year', 'total_population_used']
    
    df_global_agg = df_global_weighted.merge(pop_by_year, on='year')
    weighting_used = "population-weighted"
else:
    # Simple mean
    df_global_agg = df_global.groupby('year')['obesity_pct'].mean().reset_index()
    df_global_agg.columns = ['year', 'global_obesity_avg']
    df_global_agg['total_population_used'] = np.nan
    weighting_used = "unweighted (population data not available)"

print(f"   • Weighting method: {weighting_used}")
print(f"   • Years available: {df_global_agg['year'].min()} - {df_global_agg['year'].max()}")

# Fit OLS: global_obesity_avg ~ 1 + year
print("\n🔧 Fitting linear trend model...")
X_global = sm.add_constant(df_global_agg['year'].values)
y_global = df_global_agg['global_obesity_avg'].values

model_global = sm.OLS(y_global, X_global).fit()

# Save summary
summary_text_global = str(model_global.summary())
with open(OUTPUT_DIR / "global_trend_summary.txt", 'w') as f:
    f.write("="*70 + "\n")
    f.write("GLOBAL OBESITY TREND FORECAST (2010-2030)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Weighting method: {weighting_used}\n")
    f.write(f"Observations: {len(df_global_agg)} years (2010-2022)\n\n")
    f.write(summary_text_global)
    f.write("\n\n" + "="*70 + "\n")
    f.write("LIMITATIONS\n")
    f.write("="*70 + "\n\n")
    f.write("1. This is a simple linear extrapolation assuming current trends persist.\n")
    f.write("2. Does not account for policy changes, economic shocks, or other structural breaks.\n")
    f.write("3. Ecological analysis - correlations do not imply causation.\n")
    f.write("4. Global average masks significant regional and country-level heterogeneity.\n")
    f.write("5. Prediction intervals reflect statistical uncertainty, not model uncertainty.\n")

# Forecast for 2023-2030
print("\n📊 Generating forecast for 2023-2030...")
forecast_years = np.arange(2023, 2031)
X_forecast = sm.add_constant(forecast_years)

# Get predictions with intervals
forecast_results = model_global.get_prediction(X_forecast)
forecast_mean = forecast_results.predicted_mean
forecast_ci = forecast_results.conf_int(alpha=0.05)  # 95% CI

# Combine observed and forecast
df_forecast = pd.DataFrame({
    'year': np.concatenate([df_global_agg['year'].values, forecast_years]),
    'pred': np.concatenate([model_global.fittedvalues, forecast_mean]),
    'pred_low95': np.concatenate([
        model_global.get_prediction(X_global).conf_int(alpha=0.05)[:, 0],
        forecast_ci[:, 0]
    ]),
    'pred_high95': np.concatenate([
        model_global.get_prediction(X_global).conf_int(alpha=0.05)[:, 1],
        forecast_ci[:, 1]
    ]),
    'observed': np.concatenate([
        y_global,
        [np.nan] * len(forecast_years)
    ])
})

df_forecast.to_csv(OUTPUT_DIR / "output_obesity_forecast_2010_2030.csv", index=False)
print(f"   ✅ Saved: {OUTPUT_DIR / 'output_obesity_forecast_2010_2030.csv'}")

# Plot forecast
print("\n📊 Creating forecast visualization...")
fig, ax = plt.subplots(figsize=(14, 8))

# Observed data
observed_mask = df_forecast['observed'].notna()
ax.scatter(df_forecast.loc[observed_mask, 'year'], 
           df_forecast.loc[observed_mask, 'observed'],
           s=100, color='steelblue', marker='o', zorder=3, 
           label='Observed (2010-2022)', edgecolors='black', linewidth=1.5)

# Fitted line (2010-2022)
fitted_mask = (df_forecast['year'] <= 2022) & df_forecast['observed'].notna()
ax.plot(df_forecast.loc[fitted_mask, 'year'], 
        df_forecast.loc[fitted_mask, 'pred'],
        'b-', linewidth=2.5, label='Fitted trend (2010-2022)', zorder=2)

# Forecast line (2023-2030)
forecast_mask = df_forecast['year'] >= 2023
ax.plot(df_forecast.loc[forecast_mask, 'year'], 
        df_forecast.loc[forecast_mask, 'pred'],
        'r--', linewidth=2.5, label='Forecast (2023-2030)', zorder=2)

# Confidence intervals
ax.fill_between(df_forecast.loc[forecast_mask, 'year'],
                df_forecast.loc[forecast_mask, 'pred_low95'],
                df_forecast.loc[forecast_mask, 'pred_high95'],
                alpha=0.3, color='red', label='95% Prediction Interval', zorder=1)

# Historical confidence intervals (lighter)
ax.fill_between(df_forecast.loc[fitted_mask, 'year'],
                df_forecast.loc[fitted_mask, 'pred_low95'],
                df_forecast.loc[fitted_mask, 'pred_high95'],
                alpha=0.2, color='blue', zorder=1)

# Vertical line at 2022
ax.axvline(x=2022, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='End of observed data')

ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Global Obesity Prevalence (%)', fontsize=14, fontweight='bold')
ax.set_title('Global Obesity Trend Forecast: 2010-2030', fontsize=16, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(2009, 2031)

# Add 2030 prediction text
pred_2030 = forecast_mean[-1]
ci_low_2030 = forecast_ci[-1, 0]
ci_high_2030 = forecast_ci[-1, 1]

textstr = f'2030 Forecast:\n{pred_2030:.1f}%\n(95% PI: {ci_low_2030:.1f}% - {ci_high_2030:.1f}%)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(OUTPUT_PLOTS / "global_obesity_forecast_2030.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_PLOTS / 'global_obesity_forecast_2030.png'}")
plt.close()

# Print summary sentence
print(f"\n📝 Forecast Summary:")
summary_sentence = (
    f"Linear trend projection suggests global adult obesity prevalence will reach "
    f"approximately {pred_2030:.1f}% by 2030 (95% PI: {ci_low_2030:.1f}% – {ci_high_2030:.1f}%), "
    f"assuming current trends persist."
)
print(f"   {summary_sentence}")

with open(OUTPUT_DIR / "global_trend_summary.txt", 'a') as f:
    f.write("\n" + "="*70 + "\n")
    f.write("FORECAST SUMMARY\n")
    f.write("="*70 + "\n\n")
    f.write(summary_sentence + "\n")

# ============================================================================
# STEP 4: PACKAGING & README
# ============================================================================

print("\n" + "="*70)
print("STEP 4: CREATING DOCUMENTATION")
print("="*70)

# README
readme_content = f"""# Predictive Analysis Output Files

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
- **Countries**: {df['country'].nunique()} countries

## Key Findings

### Simple Regression
- Fat share shows {'positive' if coef > 0 else 'negative'} association with obesity
- R² = {r_squared:.3f}
- RMSE = {rmse:.2f}%

### Multiple Regression
- Test R² = {r2_test:.3f}
- Test RMSE = {rmse_test:.2f}%
- Strongest predictor: {significant_vars.iloc[0]['variable'] if len(significant_vars) > 0 else 'N/A'}

### Global Forecast
- 2030 projection: {pred_2030:.1f}% (95% PI: {ci_low_2030:.1f}% - {ci_high_2030:.1f}%)
- Weighting: {weighting_used}

## Notes

- All models use OLS (Ordinary Least Squares) from statsmodels
- Train/test split for multiple regression: 80/20 (random_state=42)
- Forecast assumes linear trend continuation (no structural breaks)
- Ecological analysis - correlations do not imply causation

## Fallbacks Applied

{f"- Calculated fat_share from fat_g_day and energy_kcal_day\n" if 'fat_share' not in df.columns else ""}\
{f"- Used Sugar_share as sugar_share\n" if 'Sugar_share' in df.columns else ""}\
{f"- Used Oils & Fats_share as oil_fats_share\n" if 'Oils & Fats_share' in df.columns else ""}\
{f"- Used Meat_share as meat_share\n" if 'Meat_share' in df.columns else ""}\
{f"- Used unweighted mean for global average (population data not available)\n" if weighting_used == "unweighted" else ""}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(OUTPUT_DIR / "README_prediction.txt", 'w') as f:
    f.write(readme_content)
print(f"   ✅ Saved: {OUTPUT_DIR / 'README_prediction.txt'}")

# Environment info
import sys
import importlib

import scipy
env_info = f"""Python Version: {sys.version}

Package Versions:
- pandas: {pd.__version__}
- numpy: {np.__version__}
- statsmodels: {sm.__version__}
- scipy: {scipy.__version__}
- matplotlib: {plt.matplotlib.__version__}
- seaborn: {sns.__version__}

Random Seed: 42 (for reproducibility)
"""

with open(OUTPUT_DIR / "env_info.txt", 'w') as f:
    f.write(env_info)
print(f"   ✅ Saved: {OUTPUT_DIR / 'env_info.txt'}")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print(f"Plots saved to: {OUTPUT_PLOTS}")

