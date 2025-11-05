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

# ======================
# Load Data
# ======================

hourly = pd.read_csv(HOURLY_PATH)
# daily = pd.read_csv(DAILY_PATH)

# ======================
# Minor cleaning
# ======================

hourly["datetime"] = pd.to_datetime(hourly["datetime"])
hourly["date"] = pd.to_datetime(hourly["date"]).dt.strftime("%Y-%m-%d")

hourly['region'] = hourly['region'].astype('category')
hourly['month'] = hourly['month'].astype('category')
hourly['hour'] = hourly['hour'].astype('category')
hourly['weekend_flag'] = hourly['weekend_flag'].astype(int)

# print(hourly.dtypes)

hourly_model = hourly.loc[hourly['missing_weather_flag'] == 0].copy()

# ======================
# Model
# ======================

model = smf.ols(
    "trip_count ~ temperature_c + rain + weekend_flag + C(region) + C(month) + C(hour)",
    data=hourly_model
).fit(cov_type='HC1')

summary_text = model.summary().as_text()
# print(summary_text)

# Save for reproducibility
with open("results/regression_summary.txt", "w") as f:
    f.write(summary_text)

results_table = (
    model.summary2().tables[1][['Coef.', 'Std.Err.', 'P>|z|']]
    .rename(columns={'Coef.': 'coef', 'Std.Err.': 'se', 'P>|z|': 'pval'})
    .loc[['temperature_c', 'rain', 'weekend_flag']]
)

results_table.to_csv("results/elasticity_table.csv", index_label='variable')

# ======================
# Compute elasticities
# ======================

mean_trips = hourly_model['trip_count'].mean()

results_table['elasticity_pct_per_unit'] = 100 * results_table['coef'] / mean_trips
results_table['ci_lo'] = 100 * (results_table['coef'] - 1.96 * results_table['se']) / mean_trips
results_table['ci_hi'] = 100 * (results_table['coef'] + 1.96 * results_table['se']) / mean_trips

results_table['adj_r2'] = model.rsquared_adj

results_table.to_csv("results/elasticity_table.csv", index_label="variable")
print(results_table)

# ======================
# Predicted vs Actual Plot
# ======================

hourly_model['predicted'] = model.fittedvalues

plt.figure(figsize=(6,6))
plt.scatter(hourly_model['predicted'], hourly_model['trip_count'], alpha=0.3, s=10)
plt.plot([0, hourly_model['trip_count'].max()],
         [0, hourly_model['trip_count'].max()], 'r--')
plt.xlabel("Predicted trip count")
plt.ylabel("Actual trip count")
plt.title(f"Predicted vs. Actual Hourly Trips (Adj. R² = {model.rsquared_adj:.2f})")
plt.tight_layout()
plt.savefig("plots/predicted_vs_actual_hourly.png", dpi=300)
plt.close()

# ======================
# Daylight Model
# ======================

# If hour is numeric or a string of numbers
daylight = hourly_model.loc[
    pd.to_numeric(hourly_model['hour'], errors='coerce').between(7, 21)
].copy()
day_model = smf.ols(
    "trip_count ~ temperature_c + rain + weekend_flag + C(region) + C(month) + C(hour)",
    data=daylight
).fit(cov_type='HC1')

# Quick peek at the weather coefficients
print(day_model.summary2().tables[1].loc[['temperature_c', 'rain', 'weekend_flag']])