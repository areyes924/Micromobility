import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


'''
[] Script

[Summary]

Notes:
----------------------
- 
===========================================================
'''

# ======================
# Define Constant Parameters
# ======================

HOURLY_PATH = "data/processed/panels/hourly_24-25.csv"
DAILY_PATH = "data/processed/panels/daily_24-25.csv"
REGRESSION_SUMMARY_PATH = "results/OLS/OLS_regression_summary.txt"
ELASTICITY_TABLE_PATH = "results/OLS/OLS_elasticity_table.csv"

PRED_VS_ACTUAL_PATH = "plots/OLS/predicted_vs_actual_hourly.png"

# ======================
# Load Data
# ======================

hourly = pd.read_csv(HOURLY_PATH)
# daily = pd.read_csv(DAILY_PATH)

# ======================
# Minor cleaning
# ======================

hourly["date"] = pd.to_datetime(hourly["date"]).dt.strftime("%Y-%m-%d")

hourly['region'] = hourly['region'].astype('category')
hourly['month'] = hourly['month'].astype('category')
hourly['hour'] = hourly['hour'].astype('category')
hourly['weekend_flag'] = hourly['weekend_flag'].astype(int)

# print(hourly.dtypes)

# Remove missing weather flag entries
hourly_model = hourly.loc[hourly['missing_weather_flag'] == 0].copy()

# ======================
# OLS Model
# ======================

# Fit OLS model with standard errors clustered by date
model = smf.ols(
    "trip_count ~ temperature_c + rain_mm + weekend_flag + C(region) + C(month) + C(hour)",
    data=hourly_model
).fit(cov_type='cluster', cov_kwds={'groups': hourly_model['date']})

# use HAC instead...?
# model = smf.ols(
#     "trip_count ~ temperature_c + rain + weekend_flag + C(region) + C(month) + C(hour)",
#     data=hourly_model
# ).fit(cov_type='HAC', cov_kwds={'maxlags': 3})


# Save summary
summary_text = model.summary().as_text()

# Save for reproducibility
with open(REGRESSION_SUMMARY_PATH, "w") as f:
    f.write(summary_text)

results_table = (
    model.summary2().tables[1][['Coef.', 'Std.Err.', 'P>|z|']]
    .rename(columns={'Coef.': 'coef', 'Std.Err.': 'se', 'P>|z|': 'pval'})
    .loc[['temperature_c', 'rain_mm', 'weekend_flag']]
)

# ======================
# Compute elasticities
# ======================

mean_trips = hourly_model['trip_count'].mean()

# Convert absolute OLS effects (trips per unit) into approximate % changes relative to mean trips
results_table['pct_change_per_unit'] = 100 * results_table['coef'] / mean_trips

# 95% confidence interval bounds for percent change estimates
results_table['ci_lo'] = 100 * (results_table['coef'] - 1.96 * results_table['se']) / mean_trips
results_table['ci_hi'] = 100 * (results_table['coef'] + 1.96 * results_table['se']) / mean_trips

results_table['adj_r2'] = model.rsquared_adj
results_table['rmse'] = np.sqrt(np.mean(model.resid ** 2))
results_table['mae'] = np.mean(np.abs(model.resid))

results_table.to_csv(ELASTICITY_TABLE_PATH, index_label="variable")
print(results_table)

# ======================
# Predicted vs Actual Plot
# ======================

hourly_model['predicted'] = model.fittedvalues

plt.figure(figsize=(6,6))
plt.scatter(hourly_model['predicted'], hourly_model['trip_count'], alpha=0.3, s=10)
mx = hourly_model['trip_count'].max()
plt.plot([0, mx], [0, mx], 'r--', linewidth=2)
plt.xlabel("Predicted trip count")
plt.ylabel("Actual trip count")
plt.title(f"Predicted vs. Actual Hourly Trips (Adj. R² = {model.rsquared_adj:.2f})")
plt.tight_layout()
plt.savefig(PRED_VS_ACTUAL_PATH, dpi=300)
plt.close()