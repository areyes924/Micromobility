import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm
import patsy
import matplotlib.pyplot as plt


'''
Hourly GLM Poisson / NB2 Ridership Model Script

Fits Poisson and, if overdispersed, Negative Binomial GLMs for hourly Metro Bike
trips using temperature, rain, and weekend effects with fixed effects and
clustered SEs. Produces elasticity tables and predicted-vs-actual plots with
nonnegative predictions.
 
===========================================================
'''

# Change when you're running the model for different cities
city = "LA"
HOURLY_PATH = "data/processed/panels/LA_hourly_24-25.csv"

# city = "NYC"
# HOURLY_PATH = "data/processed/panels/NYC_hourly_2023.csv"

# ======================
# Define Constant Parameters
# ======================

ELASTICITY_TABLE_PATH = f"results/GLM/{city}/GLM_elasticity_table.csv"
POISSON_SUMMARY_PATH = f"results/GLM/{city}/GLM_regression_summary_Poisson.txt"
NB2_SUMMARY_PATH = f"results/GLM/{city}/GLM_regression_summary_NB2.txt"
DISPERSION_SWITCH = 1.5

# ======================
# Load Data
# ======================
hourly = pd.read_csv(HOURLY_PATH)

# date is ISO string in the panel. Parse once for grouping and residual calcs.
hourly["date"] = pd.to_datetime(hourly["date"], errors="coerce")

# Remove missing weather flag entries
hourly_model = hourly.loc[hourly["missing_weather_flag"] == 0].copy()

# Create clusters for day
hourly_model["cluster_day"] = hourly_model["date_id"].astype(int)

# ======================
# GLM Poisson Model
# ======================



# Defining formula for model
formula = ""
using_snow = not hourly_model["snow_mm"].abs().sum() == 0
if not using_snow: # snow is all zero
    formula = "trip_count ~ temperature_c + rain_mm + weekend_flag + C(region) + C(month) + C(hour)"
    print("snow_mm is all zero: dropping from formula")
else:
    formula = "trip_count ~ temperature_c + rain_mm + snow_mm + weekend_flag + C(region) + C(month) + C(hour)"
    print("using snow in formula")

# Fit (clustered) Poisson Model
glm = smf.glm(formula=formula, data=hourly_model, family=sm.families.Poisson())
res = glm.fit(cov_type="cluster", cov_kwds={"groups": hourly_model["cluster_day"]})

dispersion = res.pearson_chi2 / res.df_resid
print(f"Pearson dispersion = {dispersion:.2f}")

with open(POISSON_SUMMARY_PATH, "w") as f:
    f.write(res.summary().as_text())

# Build design matrices once for NB2 fit and NB2 prediction
y, X = patsy.dmatrices(formula, hourly_model, return_type="dataframe")

# Overdispersion check for NB2
if dispersion > DISPERSION_SWITCH:
    print("Overdispersion detected. Fitting NB2 with estimated alpha.")
    nb_mod = dm.NegativeBinomial(y, X)
    nb_res = nb_mod.fit(
        cov_type="cluster",
        cov_kwds={"groups": hourly_model["cluster_day"]},
        maxiter=200,
        disp=False
    )
    with open(NB2_SUMMARY_PATH, "w") as f:
        f.write(nb_res.summary().as_text())
    final_model = nb_res
    model_label = "Negative Binomial (NB2, alpha estimated)"
else:
    final_model = res
    model_label = "Poisson"

print(f"Using {model_label} results for interpretation.\n")

# ======================
# Residual correlation within day
# ======================
hourly_model["resid"] = final_model.resid_pearson
corr_list = []
for _, g in hourly_model.groupby("date"):
    if len(g) > 1:
        c = g["resid"].corr(g["resid"].shift(1))
        if not np.isnan(c):
            corr_list.append(c)
rho = float(np.nanmean(corr_list)) if corr_list else np.nan
print(f"Average within-day residual correlation (rho) ≈ {rho:.2f}")
# ρ = 0.36 (tested on 11/10/25) implies about 2.5 independent hourly observations per day—clustering corrects
# overconfidence while keeping more information than daily aggregation.

# ======================
# Elasticity table
# ======================
keep = ["temperature_c", "rain_mm", "snow_mm", "weekend_flag"] if using_snow else ["temperature_c", "rain_mm", "weekend_flag"]
coefs = final_model.params.loc[keep]
ses = final_model.bse.loc[keep]
pvals = final_model.pvalues.loc[keep]

pct_change = 100.0 * (np.exp(coefs) - 1.0)
ci_lo = 100.0 * (np.exp(coefs - 1.96 * ses) - 1.0)
ci_hi = 100.0 * (np.exp(coefs + 1.96 * ses) - 1.0)

elasticity_table = pd.DataFrame(
    {
        "coef_beta": coefs,
        "se_beta": ses,
        "pval": pvals,
        "pct_change_per_unit": pct_change,
        "ci_lo_pct": ci_lo,
        "ci_hi_pct": ci_hi,
    }
)
elasticity_table.to_csv(ELASTICITY_TABLE_PATH, index_label="variable")

print(f"\n=== {model_label} Elasticities (%Δ per unit) ===")
print(elasticity_table, "\n")

# ======================
# Predicted vs actual plot
# ======================
if model_label.startswith("Negative Binomial"):
    preds = np.asarray(final_model.predict(X))
else:
    preds = np.asarray(final_model.predict(hourly_model))

hourly_model["predicted"] = preds

plt.figure(figsize=(6, 6))
plt.scatter(
    hourly_model["predicted"],
    hourly_model["trip_count"],
    alpha=0.30,
    s=10,
    zorder=1
)

mx = float(np.nanmax([hourly_model["trip_count"].max(), hourly_model["predicted"].max()]))

plt.plot([0, mx], [0, mx], color="red", linestyle="--", linewidth=2, zorder=5) #, label = 

plt.xlim(0, mx)
plt.ylim(0, mx)
plt.xlabel(f"Predicted trip count ({model_label})")
plt.ylabel("Actual trip count")
plt.title(f"Predicted vs. Actual Hourly Trips ({model_label})")
plt.tight_layout()

tag = "NB2" if dispersion > DISPERSION_SWITCH else "Poisson"
out_png = f"plots/GLM/{city}_predicted_vs_actual_hourly_{tag}.png"
plt.savefig(out_png, dpi=300)
plt.close()
